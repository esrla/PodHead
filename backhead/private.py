# This module provides functionality for reading from the private.json configuration file.
#
# Purpose:
# - Load sensitive configuration data, including:
#   - API keys and integration credentials.
#   - IMAP/SMTP settings.
#   - Whitelist of allowed users.
#   - Endpoint configurations for various services.
#
# Responsibilities:
# - Validate the structure and required fields in private.json.
# - Generate a template private.json file with placeholder values on first use.
#
# All updates, including adding new integrations, models, or email configurations, are performed directly in private.json.