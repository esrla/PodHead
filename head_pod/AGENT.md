# Agent Workspace Guide

This file is the starting point for the agent workspace located at `/workspace` inside
the container. The agent and user may update this file and extend the workspace structure
over time as new conventions emerge.

## Purpose

The workspace is the agent's personal, persistent environment. It holds skills, scripts,
memories, temporary files, and other reusable resources that survive across sessions.

## Standard Directories

| Directory  | Purpose                                                                 |
|------------|-------------------------------------------------------------------------|
| `skills/`  | Packaged, documented capabilities — each skill lives in its own folder  |
| `tools/`   | General-purpose scripts; `tools.md` is the registry                     |
| `memories/`| Persistent notes and structured recall data                             |
| `tmp/`     | Temporary files; may be cleared between sessions                        |
| `out/`     | Generated artifacts and output files                                    |

## Skills

A skill is a subdirectory under `skills/` that contains at minimum a `SKILL.md` file.

`SKILL.md` must begin with a YAML frontmatter block that includes exactly two standard
fields:

```
---
name: Short human-readable skill name
description: One-sentence description of what the skill does.
---
```

The body of `SKILL.md` (after the closing `---`) may contain full usage documentation,
examples, and notes. Reusable scripts should be packaged and documented as skills rather
than left as bare files in `tools/`.

## Backend Tools

In addition to `run_cli` and `spawn_subagent`, the backend provides the following
tool directly accessible to the agent without going through the container:

### embed_text

```
embed_text(text: str) -> {"ok": true, "embedding": [float, ...]}
```

Returns a normalized semantic embedding vector for `text`. Useful for comparing
texts by cosine similarity (e.g. `sum(a*b for a, b in zip(vec_a, vec_b))`),
finding relevant content, or working alongside the skill search script.

## Finding Skills

Use the skill search script to find the most relevant skill for a task:

```
python /workspace/tools/find_skill.py "your query here"
```

Options:

| Flag           | Default             | Description                  |
|----------------|---------------------|------------------------------|
| `--top N`      | 5                   | Number of results to return  |
| `--skills-dir` | `/workspace/skills` | Skills root directory        |

The script reads the `name` and `description` from each `SKILL.md` frontmatter, generates
one multilingual embedding per skill (stored as `.embed.npy` beside `SKILL.md`), and
returns the top N matches for the query. Embeddings are regenerated automatically when
`SKILL.md` is newer than the cached file.

## Extending the Workspace

- Add a new subdirectory under `skills/` and create a `SKILL.md` to package a capability.
- Add scripts to `tools/` and register them with purpose and invocation examples in
  `tools/tools.md`.
- Update `preferences.md` to adjust behavioral defaults.
- Update this file to document new conventions or directory structures.
