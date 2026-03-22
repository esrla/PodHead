# Contains the agent loop

# Expected Functionality:
# The agent loop processes one interaction (per turn) when triggered by main.py. Below is the expected functionality:
# 1. Trigger Implementation:
#    - Receives an event triggered by main.py. Includes details like person, source, sender information, and
#      any attachments or messages.
# 
# 2. Database Operations:
#    - Updates the SQLite database with a log of the message (append-only model).
#    - Associates messages with unique IDs for tracking continuity.
# 
# 3. Context Building for Conversation:
#    - Constructs conversation history for the person involved by:
#      - Summarizing previous interactions using a lightweight model.
#      - Fetching the last few question/answer groups.
#    - Loads preferences and operational skills from agent_rootfs/workspace.
# 
# 4. LLM Call:
#    - Calls the Language Model API (LLM) to:
#      - Generate a response for the current interaction.
#      - Potentially request tool/script execution.
# 
# 5. Tool Execution:
#    - Executes tools/scripts (if requested by the LLM) located in agent_rootfs/workspace/tools/.
#    - Uses run_cli_in_container to securely run these tools in an isolated environment.
# 
# 6. Result Handling:
#    - Outputs the generated assistant response and any tool results back to main.py for further processing, such as sending replies to users.