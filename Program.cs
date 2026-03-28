// Program.cs — Backend entry point for PodHead.
// Reads what.md for the full specification, then runs:
//   1. Load private config (generate template + exit if missing).
//   2. Ensure the agent container image exists (build if not).
//   3. Initialise the SQLite database.
//   4. Run the always-on email-polling loop.

using System.Diagnostics;
using System.Net.Mail;
using System.Text.Json;
using System.Text.Json.Serialization;
using MailKit;
using MailKit.Net.Imap;
using MailKit.Search;
using Microsoft.Data.Sqlite;
using MimeKit;

// ── cancellation ──────────────────────────────────────────────────────────────
using var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) =>
{
    e.Cancel = true;
    cts.Cancel();
};
AppDomain.CurrentDomain.ProcessExit += (_, _) => cts.Cancel();

// ── 1. Load private config ────────────────────────────────────────────────────
const string ConfigPath = "backend/system.private.json";
var config = PrivateConfig.Load(ConfigPath);

// ── 2. Ensure agent container image ──────────────────────────────────────────
await EnsureAgentImageAsync();

// ── 3. Initialise database ────────────────────────────────────────────────────
const string DbPath = "backend/state/agent.db";
Directory.CreateDirectory(Path.GetDirectoryName(DbPath)!);
using var db = new SqliteConnection($"Data Source={DbPath}");
db.Open();
Db.InitSchema(db);

// ── 4. Main polling loop ──────────────────────────────────────────────────────
Console.WriteLine($"[PodHead] Started. Polling every {config.Runtime.PollIntervalSec}s.");

while (!cts.IsCancellationRequested)
{
    InboundEvent? evt = null;
    try
    {
        evt = PollInbox(config, db);
        if (evt is not null)
        {
            Console.WriteLine($"[PodHead] Event from {evt.Person} ({evt.SenderHandle}).");
            await AgentLoop.RunTurnAsync(evt, config, db);
        }
    }
    catch (OperationCanceledException) when (cts.IsCancellationRequested)
    {
        break;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"[PodHead] ERROR: {ex}");
        // Best-effort error email — swallow secondary failures.
        try { SendErrorMail(config, ex.Message, evt?.SenderHandle); } catch { }
    }

    try
    {
        await Task.Delay(
            TimeSpan.FromSeconds(config.Runtime.PollIntervalSec),
            cts.Token);
    }
    catch (OperationCanceledException) { break; }
}

Console.WriteLine("[PodHead] Stopped.");

// ── helpers ───────────────────────────────────────────────────────────────────

static async Task EnsureAgentImageAsync()
{
    static async Task<int> RunAsync(string cmd, string args)
    {
        using var p = Process.Start(new ProcessStartInfo(cmd, args)
        {
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
        })!;
        await p.WaitForExitAsync();
        return p.ExitCode;
    }

    int exists = await RunAsync("podman", "image exists agent-image");
    if (exists != 0)
    {
        Console.WriteLine("[PodHead] agent-image not found — building…");
        int build = await RunAsync("podman", "build -t agent-image .");
        if (build != 0)
            throw new InvalidOperationException("podman build failed. Check the Containerfile.");
        Console.WriteLine("[PodHead] agent-image built successfully.");
    }
}

static InboundEvent? PollInbox(PrivateConfig config, SqliteConnection db)
{
    using var client = new ImapClient();
    client.Connect(config.Email.ImapHost, config.Email.ImapPort, true);
    client.Authenticate(config.Email.ImapUser, config.Email.ImapPass);
    client.Inbox.Open(FolderAccess.ReadWrite);

    var uids = client.Inbox.Search(SearchQuery.NotSeen);
    foreach (var uid in uids)
    {
        var summary = client.Inbox.Fetch(new[] { uid },
            MessageSummaryItems.Envelope | MessageSummaryItems.UniqueId)[0];

        string messageId = summary.Envelope.MessageId ?? uid.ToString();
        string from      = summary.Envelope.From.Mailboxes.FirstOrDefault()?.Address ?? "";

        // Resolve person
        string? person = ResolvePerson(from, config);
        if (person is null)
        {
            // Mark seen so we don't keep re-fetching unknown senders.
            client.Inbox.AddFlags(uid, MessageFlags.Seen, true);
            continue;
        }

        // Deduplication
        if (Db.MessageExists(db, "email", messageId))
        {
            client.Inbox.AddFlags(uid, MessageFlags.Seen, true);
            continue;
        }

        // Fetch full message
        var message  = client.Inbox.GetMessage(uid);
        string body  = message.TextBody ?? message.HtmlBody ?? "";

        client.Inbox.AddFlags(uid, MessageFlags.Seen, true);
        client.Disconnect(true);

        return new InboundEvent(
            Person:       person,
            Source:       "email",
            SenderHandle: from,
            EventId:      messageId,
            Subject:      summary.Envelope.Subject ?? "",
            UserPrompt:   body,
            SentTs:       summary.Envelope.Date?.ToUnixTimeSeconds() ?? DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            ReceivedTs:   DateTimeOffset.UtcNow.ToUnixTimeSeconds()
        );
    }

    client.Disconnect(true);
    return null;
}

static string? ResolvePerson(string senderHandle, PrivateConfig config)
{
    foreach (var (person, identities) in config.Persons)
    {
        if (!config.Whitelist.Contains(person)) continue;
        if (identities.Any(id => string.Equals(id, senderHandle, StringComparison.OrdinalIgnoreCase)))
            return person;
    }
    return null;
}

static void SendErrorMail(PrivateConfig config, string errorMessage, string? senderHandle = null)
{
    string recipient = !string.IsNullOrWhiteSpace(senderHandle)
        ? senderHandle
        : config.Email.SmtpUser;

    using var smtp = new System.Net.Mail.SmtpClient(config.Email.SmtpHost)
    {
        Credentials = new System.Net.NetworkCredential(config.Email.SmtpUser, config.Email.SmtpPass),
        EnableSsl   = true,
    };
    var msg = new System.Net.Mail.MailMessage(config.Email.SmtpUser, recipient)
    {
        Subject = "[PodHead] Internal error",
        Body    = $"An unhandled error occurred:\n\n{errorMessage}",
    };
    smtp.Send(msg);
}

// ── domain types ──────────────────────────────────────────────────────────────

record InboundEvent(
    string Person,
    string Source,
    string SenderHandle,
    string EventId,
    string Subject,
    string UserPrompt,
    long   SentTs,
    long   ReceivedTs);

// ── AgentLoop ─────────────────────────────────────────────────────────────────

static class AgentLoop
{
    private const string PreferencesPath = "head_pod/workspace/preferences.md";
    private const string SkillsPath      = "head_pod/workspace/skills.md";

    public static async Task RunTurnAsync(
        InboundEvent  evt,
        PrivateConfig config,
        SqliteConnection db)
    {
        long now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        // 1. Persist inbound user message.
        Db.PersistMessage(db, new DbMessage(
            Person:           evt.Person,
            Source:           evt.Source,
            SenderHandle:     evt.SenderHandle,
            EventId:          evt.EventId,
            Role:             "user",
            Kind:             "message",
            Text:             evt.UserPrompt,
            Timestamp:        evt.ReceivedTs,
            MessageReference: null,
            ToolName:         null,
            ToolCallId:       null,
            IsError:          0));

        // 2. Build context.
        string summary     = Db.GetSummary(db, evt.Person);
        var    history     = Db.GetHistory(db, evt.Person, config.Runtime.MaxPairs, evt.EventId);
        string preferences = File.Exists(PreferencesPath) ? File.ReadAllText(PreferencesPath) : "";
        string skills      = File.Exists(SkillsPath)      ? File.ReadAllText(SkillsPath)      : "";

        // 3. Compose messages for the LLM.
        var messages = new List<LlmMessage>
        {
            new("system",
                "You are PodHead, an always-on personal assistant running on a Raspberry Pi.\n\n" +
                $"## Preferences\n{preferences}\n\n## Skills\n{skills}"),
        };

        // Prior Q/A history (chronological, old to new) comes before the current turn.
        messages.AddRange(history);

        // Current turn: summary prepended to the new user question.
        string summaryBlock = string.IsNullOrWhiteSpace(summary)
            ? ""
            : $"[Conversation summary so far]\n{summary}\n\n";
        messages.Add(new("user", summaryBlock + evt.UserPrompt));

        // 4. Call the LLM (tool loop).
        string assistantText = "";
        List<string> imagePaths = new();

        try
        {
            var result = await Llm.ChatAsync(messages, config);
            assistantText = result.Text;

            // Handle SendImageToUser tool calls.
            foreach (var call in result.ToolCalls)
            {
                if (call.Name == "SendImageToUser")
                {
                    var path = HandleSendImageToUser(call, evt.EventId, db);
                    if (path is not null) imagePaths.Add(path);
                }
            }
        }
        catch (Exception ex)
        {
            assistantText = "I encountered an error and could not complete your request.";
            Db.PersistMessage(db, new DbMessage(
                Person:           evt.Person,
                Source:           config.Llm.ChatEndpoint,
                SenderHandle:     "",
                EventId:          Guid.NewGuid().ToString(),
                Role:             "assistant",
                Kind:             "message",
                Text:             ex.ToString(),
                Timestamp:        DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                MessageReference: evt.EventId,
                ToolName:         null,
                ToolCallId:       null,
                IsError:          1));
        }

        // 5. Persist assistant response.
        Db.PersistMessage(db, new DbMessage(
            Person:           evt.Person,
            Source:           config.Llm.ChatEndpoint,
            SenderHandle:     "",
            EventId:          Guid.NewGuid().ToString(),
            Role:             "assistant",
            Kind:             "message",
            Text:             assistantText,
            Timestamp:        DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            MessageReference: evt.EventId,
            ToolName:         null,
            ToolCallId:       null,
            IsError:          0));

        // 6. Update summary.
        try
        {
            string newSummary = await Llm.SummarizeAsync(
                messages, assistantText, config);
            Db.SetSummary(db, evt.Person, newSummary);
        }
        catch { /* non-fatal */ }

        // 7. Send reply email.
        Mail.Send(config, evt.SenderHandle, evt.Subject, assistantText, imagePaths);
    }

    private static string? HandleSendImageToUser(
        ToolCall call, string triggerEventId, SqliteConnection db)
    {
        try
        {
            string outDir = "head_pod/workspace/out";
            Directory.CreateDirectory(outDir);

            string base64   = call.Arguments.GetValueOrDefault("base64", "");
            string filename = call.Arguments.GetValueOrDefault("filename", "image.png");
            string outPath  = Path.Combine(outDir, filename);

            // Strip data-URL prefix if present.
            int comma = base64.IndexOf(',');
            if (comma >= 0) base64 = base64[(comma + 1)..];

            File.WriteAllBytes(outPath, Convert.FromBase64String(base64));
            return outPath;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[SendImageToUser] {ex.Message}");
            return null;
        }
    }
}

// ── Db ────────────────────────────────────────────────────────────────────────

static class Db
{
    public static void InitSchema(SqliteConnection db)
    {
        db.ExecuteNonQuery("PRAGMA journal_mode=WAL;");
        db.ExecuteNonQuery("""
            CREATE TABLE IF NOT EXISTS summaries (
              person      TEXT    PRIMARY KEY,
              summary     TEXT    NOT NULL DEFAULT '',
              updated_ts  INTEGER NOT NULL
            );
            """);
        db.ExecuteNonQuery("""
            CREATE TABLE IF NOT EXISTS messages (
              id                INTEGER PRIMARY KEY AUTOINCREMENT,
              person            TEXT    NOT NULL,
              source            TEXT    NOT NULL,
              sender_handle     TEXT    NOT NULL,
              event_id          TEXT    NOT NULL,
              role              TEXT    NOT NULL,
              kind              TEXT    NOT NULL,
              text              TEXT    NOT NULL,
              timestamp         INTEGER NOT NULL,
              message_reference TEXT,
              tool_name         TEXT,
              tool_call_id      TEXT,
              is_error          INTEGER NOT NULL DEFAULT 0,
              embedding         BLOB,
              UNIQUE(source, event_id)
            );
            """);
        db.ExecuteNonQuery("""
            CREATE INDEX IF NOT EXISTS idx_messages_person_time
              ON messages(person, timestamp, id);
            """);
        db.ExecuteNonQuery("""
            CREATE INDEX IF NOT EXISTS idx_messages_person_ref
              ON messages(person, message_reference);
            """);
        db.ExecuteNonQuery("""
            CREATE INDEX IF NOT EXISTS idx_messages_tool_call_id
              ON messages(tool_call_id);
            """);
    }

    public static bool MessageExists(SqliteConnection db, string source, string eventId)
    {
        using var cmd = db.CreateCommand();
        cmd.CommandText = "SELECT 1 FROM messages WHERE source=@s AND event_id=@e LIMIT 1;";
        cmd.Parameters.AddWithValue("@s", source);
        cmd.Parameters.AddWithValue("@e", eventId);
        return cmd.ExecuteScalar() is not null;
    }

    public static void PersistMessage(SqliteConnection db, DbMessage msg)
    {
        using var cmd = db.CreateCommand();
        cmd.CommandText = """
            INSERT OR IGNORE INTO messages
              (person, source, sender_handle, event_id, role, kind, text,
               timestamp, message_reference, tool_name, tool_call_id, is_error)
            VALUES
              (@person, @source, @sender_handle, @event_id, @role, @kind, @text,
               @timestamp, @message_reference, @tool_name, @tool_call_id, @is_error);
            """;
        cmd.Parameters.AddWithValue("@person",           msg.Person);
        cmd.Parameters.AddWithValue("@source",           msg.Source);
        cmd.Parameters.AddWithValue("@sender_handle",    msg.SenderHandle);
        cmd.Parameters.AddWithValue("@event_id",         msg.EventId);
        cmd.Parameters.AddWithValue("@role",             msg.Role);
        cmd.Parameters.AddWithValue("@kind",             msg.Kind);
        cmd.Parameters.AddWithValue("@text",             msg.Text);
        cmd.Parameters.AddWithValue("@timestamp",        msg.Timestamp);
        cmd.Parameters.AddWithValue("@message_reference",(object?)msg.MessageReference ?? DBNull.Value);
        cmd.Parameters.AddWithValue("@tool_name",        (object?)msg.ToolName         ?? DBNull.Value);
        cmd.Parameters.AddWithValue("@tool_call_id",     (object?)msg.ToolCallId       ?? DBNull.Value);
        cmd.Parameters.AddWithValue("@is_error",         msg.IsError);
        cmd.ExecuteNonQuery();
    }

    public static string GetSummary(SqliteConnection db, string person)
    {
        using var cmd = db.CreateCommand();
        cmd.CommandText = "SELECT summary FROM summaries WHERE person=@p;";
        cmd.Parameters.AddWithValue("@p", person);
        return cmd.ExecuteScalar() as string ?? "";
    }

    public static void SetSummary(SqliteConnection db, string person, string summary)
    {
        using var cmd = db.CreateCommand();
        cmd.CommandText = """
            INSERT INTO summaries (person, summary, updated_ts)
            VALUES (@p, @s, @ts)
            ON CONFLICT(person) DO UPDATE SET summary=@s, updated_ts=@ts;
            """;
        cmd.Parameters.AddWithValue("@p",  person);
        cmd.Parameters.AddWithValue("@s",  summary);
        cmd.Parameters.AddWithValue("@ts", DateTimeOffset.UtcNow.ToUnixTimeSeconds());
        cmd.ExecuteNonQuery();
    }

    /// <summary>
    /// Returns the last <paramref name="maxPairs"/> Q/A groups for the given person,
    /// formatted as LLM messages (oldest first), excluding the current turn's event.
    /// </summary>
    public static IEnumerable<LlmMessage> GetHistory(
        SqliteConnection db, string person, int maxPairs, string excludeEventId)
    {
        // Fetch root user events (message_reference IS NULL) newest-first, limit to maxPairs,
        // excluding the current inbound event that was just persisted.
        using var rootCmd = db.CreateCommand();
        rootCmd.CommandText = """
            SELECT event_id FROM messages
            WHERE person=@p AND role='user' AND kind='message'
              AND message_reference IS NULL AND event_id != @exclude
            ORDER BY timestamp DESC, id DESC
            LIMIT @limit;
            """;
        rootCmd.Parameters.AddWithValue("@p",       person);
        rootCmd.Parameters.AddWithValue("@exclude", excludeEventId);
        rootCmd.Parameters.AddWithValue("@limit",   maxPairs);

        var rootIds = new List<string>();
        using (var r = rootCmd.ExecuteReader())
            while (r.Read()) rootIds.Add(r.GetString(0));

        // Reverse to chronological order.
        rootIds.Reverse();

        var result = new List<LlmMessage>();
        foreach (var rootId in rootIds)
        {
            // Fetch all rows in this Q/A group in order.
            using var grpCmd = db.CreateCommand();
            grpCmd.CommandText = """
                SELECT role, text FROM messages
                WHERE person=@p AND (event_id=@eid OR message_reference=@eid)
                ORDER BY timestamp ASC, id ASC;
                """;
            grpCmd.Parameters.AddWithValue("@p",   person);
            grpCmd.Parameters.AddWithValue("@eid", rootId);

            using var r = grpCmd.ExecuteReader();
            while (r.Read())
                result.Add(new LlmMessage(r.GetString(0), r.GetString(1)));
        }

        return result;
    }
}

// ── Llm ───────────────────────────────────────────────────────────────────────

static class Llm
{
    // Shared HttpClient — reuse across requests to avoid socket exhaustion.
    private static readonly HttpClient Http = new();

    private static HttpRequestMessage BuildRequest(string endpoint, string apiKey, object payload)
    {
        var req = new HttpRequestMessage(HttpMethod.Post, endpoint)
        {
            Content = new StringContent(
                JsonSerializer.Serialize(payload),
                System.Text.Encoding.UTF8,
                "application/json")
        };
        req.Headers.Add("Authorization", $"Bearer {apiKey}");
        return req;
    }

    public static async Task<LlmResult> ChatAsync(
        IEnumerable<LlmMessage> messages, PrivateConfig config)
    {
        var payload = new
        {
            model    = "gpt-4o-mini",
            messages = messages.Select(m => new { role = m.Role, content = m.Content }),
            tools    = new[]
            {
                new
                {
                    type     = "function",
                    function = new
                    {
                        name        = "SendImageToUser",
                        description = "Send an image attachment to the triggering user.",
                        parameters  = new
                        {
                            type       = "object",
                            properties = new
                            {
                                base64   = new { type = "string", description = "Base64-encoded image data or data URL." },
                                filename = new { type = "string", description = "Filename for the attachment." },
                                mime     = new { type = "string", description = "MIME type, e.g. image/png." }
                            },
                            required = new[] { "base64", "filename", "mime" }
                        }
                    }
                }
            }
        };

        using var req      = BuildRequest(config.Llm.ChatEndpoint, config.Llm.ApiKey, payload);
        var       response = await Http.SendAsync(req);
        response.EnsureSuccessStatusCode();

        using var doc    = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var choice       = doc.RootElement.GetProperty("choices")[0].GetProperty("message");
        string text      = choice.TryGetProperty("content", out var c) ? c.GetString() ?? "" : "";
        var toolCalls    = new List<ToolCall>();

        if (choice.TryGetProperty("tool_calls", out var tcs))
        {
            foreach (var tc in tcs.EnumerateArray())
            {
                string name   = tc.GetProperty("function").GetProperty("name").GetString()!;
                var    args   = JsonSerializer.Deserialize<Dictionary<string, string>>(
                    tc.GetProperty("function").GetProperty("arguments").GetString()!)!;
                toolCalls.Add(new ToolCall(name, args));
            }
        }

        return new LlmResult(text, toolCalls);
    }

    public static async Task<string> SummarizeAsync(
        IEnumerable<LlmMessage> history, string latestReply, PrivateConfig config)
    {
        string historyText = string.Join("\n",
            history.Select(m => $"{m.Role}: {m.Content}"));
        string prompt      = $"{historyText}\nassistant: {latestReply}";

        var payload = new
        {
            model    = "gpt-4o-mini",
            messages = new[]
            {
                new { role = "system", content = "Summarize the following conversation briefly." },
                new { role = "user",   content = prompt }
            }
        };

        string endpoint = string.IsNullOrWhiteSpace(config.Llm.SummarizeEndpoint)
            ? config.Llm.ChatEndpoint
            : config.Llm.SummarizeEndpoint;

        using var req      = BuildRequest(endpoint, config.Llm.ApiKey, payload);
        var       response = await Http.SendAsync(req);
        response.EnsureSuccessStatusCode();

        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        return doc.RootElement
            .GetProperty("choices")[0]
            .GetProperty("message")
            .GetProperty("content")
            .GetString() ?? "";
    }
}

// ── Mail ──────────────────────────────────────────────────────────────────────

static class Mail
{
    public static void Send(
        PrivateConfig  config,
        string         to,
        string         subject,
        string         body,
        IList<string>  attachments)
    {
        using var smtp = new System.Net.Mail.SmtpClient(config.Email.SmtpHost)
        {
            Credentials = new System.Net.NetworkCredential(config.Email.SmtpUser, config.Email.SmtpPass),
            EnableSsl   = true,
        };

        var msg = new System.Net.Mail.MailMessage(config.Email.SmtpUser, to)
        {
            Subject = subject.StartsWith("Re:", StringComparison.OrdinalIgnoreCase)
                ? subject
                : $"Re: {subject}",
            Body = body,
        };

        foreach (var path in attachments)
            msg.Attachments.Add(new Attachment(path));

        smtp.Send(msg);
    }
}

// ── PrivateConfig ─────────────────────────────────────────────────────────────

class PrivateConfig
{
    [JsonPropertyName("email")]   public EmailConfig   Email   { get; set; } = new();
    [JsonPropertyName("llm")]     public LlmConfig     Llm     { get; set; } = new();
    [JsonPropertyName("whitelist")] public List<string> Whitelist { get; set; } = new();
    [JsonPropertyName("persons")] public Dictionary<string, List<string>> Persons { get; set; } = new();
    [JsonPropertyName("runtime")] public RuntimeConfig Runtime { get; set; } = new();

    public static PrivateConfig Load(string path)
    {
        if (!File.Exists(path))
        {
            WriteTemplate(path);
            Console.Error.WriteLine(
                $"[PodHead] '{path}' was not found.\n" +
                $"A template has been created. Fill in all values and restart.");
            Environment.Exit(1);
        }

        var json   = File.ReadAllText(path);
        var config = JsonSerializer.Deserialize<PrivateConfig>(json)
            ?? throw new InvalidOperationException("Failed to parse config.");

        config.Validate();
        return config;
    }

    private void Validate()
    {
        var missing = new List<string>();

        if (string.IsNullOrWhiteSpace(Email.ImapHost)) missing.Add("email.imap_host");
        if (string.IsNullOrWhiteSpace(Email.ImapUser)) missing.Add("email.imap_user");
        if (string.IsNullOrWhiteSpace(Email.ImapPass)) missing.Add("email.imap_pass");
        if (string.IsNullOrWhiteSpace(Email.SmtpHost)) missing.Add("email.smtp_host");
        if (string.IsNullOrWhiteSpace(Email.SmtpUser)) missing.Add("email.smtp_user");
        if (string.IsNullOrWhiteSpace(Email.SmtpPass)) missing.Add("email.smtp_pass");
        if (string.IsNullOrWhiteSpace(Llm.ChatEndpoint)) missing.Add("llm.chat_endpoint");
        if (string.IsNullOrWhiteSpace(Llm.ApiKey))       missing.Add("llm.api_key");

        if (missing.Count > 0)
        {
            Console.Error.WriteLine(
                "[PodHead] The following required config fields are empty:\n  " +
                string.Join("\n  ", missing) +
                "\nPlease fill in backend/system.private.json and restart.");
            Environment.Exit(1);
        }
    }

    private static void WriteTemplate(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var template = new PrivateConfig
        {
            Email = new EmailConfig
            {
                ImapHost = "imap.example.com",
                ImapPort = 993,
                ImapUser = "you@example.com",
                ImapPass = "your-imap-password",
                SmtpHost = "smtp.example.com",
                SmtpUser = "you@example.com",
                SmtpPass = "your-smtp-password",
            },
            Llm = new LlmConfig
            {
                ChatEndpoint      = "https://api.openai.com/v1/chat/completions",
                EmbedEndpoint     = "https://api.openai.com/v1/embeddings",
                SummarizeEndpoint = "",
                ApiKey            = "sk-...",
            },
            Whitelist = new List<string> { "Alice" },
            Persons   = new Dictionary<string, List<string>>
            {
                { "Alice", new List<string> { "alice@example.com" } }
            },
            Runtime = new RuntimeConfig { MaxPairs = 10, PollIntervalSec = 30 },
        };

        var options = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(path, JsonSerializer.Serialize(template, options));
    }
}

class EmailConfig
{
    [JsonPropertyName("imap_host")] public string ImapHost { get; set; } = "";
    [JsonPropertyName("imap_port")] public int    ImapPort { get; set; } = 993;
    [JsonPropertyName("imap_user")] public string ImapUser { get; set; } = "";
    [JsonPropertyName("imap_pass")] public string ImapPass { get; set; } = "";
    [JsonPropertyName("smtp_host")] public string SmtpHost { get; set; } = "";
    [JsonPropertyName("smtp_user")] public string SmtpUser { get; set; } = "";
    [JsonPropertyName("smtp_pass")] public string SmtpPass { get; set; } = "";
}

class LlmConfig
{
    [JsonPropertyName("chat_endpoint")]      public string ChatEndpoint      { get; set; } = "";
    [JsonPropertyName("embed_endpoint")]     public string EmbedEndpoint     { get; set; } = "";
    [JsonPropertyName("summarize_endpoint")] public string SummarizeEndpoint { get; set; } = "";
    [JsonPropertyName("api_key")]            public string ApiKey            { get; set; } = "";
}

class RuntimeConfig
{
    [JsonPropertyName("max_pairs")]        public int MaxPairs        { get; set; } = 10;
    [JsonPropertyName("poll_interval_sec")] public int PollIntervalSec { get; set; } = 30;
}

// ── Supporting value types ────────────────────────────────────────────────────

record LlmMessage(string Role, string Content);

record LlmResult(string Text, List<ToolCall> ToolCalls);

record ToolCall(string Name, Dictionary<string, string> Arguments);

record DbMessage(
    string  Person,
    string  Source,
    string  SenderHandle,
    string  EventId,
    string  Role,
    string  Kind,
    string  Text,
    long    Timestamp,
    string? MessageReference,
    string? ToolName,
    string? ToolCallId,
    int     IsError);

// ── SqliteConnection extension ────────────────────────────────────────────────

static class SqliteExtensions
{
    public static void ExecuteNonQuery(this SqliteConnection db, string sql)
    {
        using var cmd = db.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }
}
