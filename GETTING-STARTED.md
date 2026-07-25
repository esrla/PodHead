1. Create one dedicated email account for PodHead.
2. Enable IMAP and SMTP.
3. Clone the repository to the Linux device.
4. Install Git, Python, pip, and Podman.
5. Install backend and test dependencies with `python -m pip install -r requirements.txt`.
6. Copy `backhead/secrets.example.py` to `backhead/secrets.py`.
7. Open `backhead/secrets.py` and replace every demonstration value with values for your installation.
8. Keep `backhead/secrets.py` local. It is excluded by `.gitignore` and must never be committed.
9. Configure the local llama.cpp chat server and dedicated embedding server paths, models, hosts, ports, context sizes, and threads in `backhead/secrets.py`.
10. Configure the email account, IMAP, SMTP, sender whitelist, and spam mailbox in `backhead/secrets.py`.
11. Start the backend with `python -m backhead.main`; PodHead will health-check both configured model endpoints and start missing local servers automatically.
12. Send a test email from a whitelisted address.
13. Build or refresh the Podman image through normal backend startup; it installs `container-requirements.txt` inside the container image.
14. Verify the reply, database, and container.
15. If one request fails, confirm PodHead replies with the full traceback on that same email thread and keeps polling for the next request.
