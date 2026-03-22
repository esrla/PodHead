# PodHead
Llm harness ment for my PI, but could be used on any device probably. Yet another llm thing in a container. I just wanted to build my own so I at least trust the code. And no way its getting access to more than i want it to.

# PodHead

## Key Elements Overview

### Introduction
This repository implements an always-on lightweight agent designed specifically for the Raspberry Pi. Its primary purpose is to manage and facilitate various automated tasks efficiently.

### Isolated Framework
The repository features a clear separation between backend and agent environments. The backend handles persistent data management, while the agent operates in a sandboxed environment utilizing `agent_rootfs`. This design ensures that both processes can function optimally without interference.

### Secrets Management
A private JSON configuration file is utilized to securely manage API keys and credentials within the repository. This practice safeguards sensitive information essential for operation.

### SQLite Database
An SQLite database serves as an append-only log, facilitating coherent and person-centric conversation storage. This structure allows for systematic record-keeping and retrieval of interactions.

### Event Listening
The main script (`Main.py`) continuously polls an email inbox (example provided) to monitor incoming events. The workflow comprises scanning, validating, and triggering deployments based on events that have been authenticated via the backend.

### Tools and CLI
The repository distinguishes between explicit LLM tools, such as those for sending images, and general-purpose scripts, like `run_cli_in_container`. These scripts execute in a containerized environment to ensure security during operation.

### Container Editing
The filesystem of the container (`agent_rootfs`) is accessible for manual adjustments, offering flexibility for users while ensuring that it remains isolated from the backend code. This approach upholds security standards while allowing customization.