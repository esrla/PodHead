1. Create one dedicated email account for PodHead.
2. Enable IMAP and SMTP.
3. Clone the repository to the Linux device.
4. Install Git, Python, pip, and Podman.
5. Install backend and test dependencies with `python -m pip install -r requirements.txt`.
6. Open backhead/private_config.py.
7. Replace the demonstration values with your values and do not commit real secrets.
8. Configure the local llama.cpp chat server and dedicated embedding server paths, models, hosts, ports, context sizes, and threads in `backhead/private_config.py`.
9. Start the backend with `python -m backhead.main`; PodHead will health-check both configured model endpoints and start missing local servers automatically.
10. Send a test email from a whitelisted address.
11. Build or refresh the Podman image through normal backend startup; it installs `container-requirements.txt` inside the container image.
12. Verify the reply, database, and container.
13. If one request fails, confirm PodHead replies with the full traceback on that same email thread and keeps polling for the next request.
