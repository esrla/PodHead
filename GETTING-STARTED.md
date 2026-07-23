1. Create one dedicated email account for PodHead.
2. Enable IMAP and SMTP.
3. Clone the repository to the Linux device.
4. Install Git, Python, pip, and Podman.
5. Install requirements.txt.
6. Open backhead/private_config.py.
7. Replace the demonstration values with your values and do not commit real secrets.
8. Start the configured local OpenAI-compatible model servers with chat completions and embeddings enabled.
9. Start the backend with `python -m backhead.main`.
10. Send a test email from a whitelisted address.
11. Verify the reply, database, and container.
