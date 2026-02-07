#!/bin/bash
# Docker entrypoint script for nanobot with restart signal support.
#
# This script wraps nanobot and monitors for restart signals.
# When a restart signal is detected, nanobot is restarted automatically.
# This is used for MCP server installation where nanobot needs to restart
# to load the newly installed server.

set -e

# Use NANOBOT_WORKSPACE if set, otherwise default to /app/workspace
WORKSPACE="${NANOBOT_WORKSPACE:-/app/workspace}"
RESTART_SIGNAL="$WORKSPACE/.restart_signal"

# Extract GITHUB_TOKEN from config for gh CLI auth (fallback if not in env)
if [ -z "$GITHUB_TOKEN" ] && [ -f "/root/.nanobot/config.json" ]; then
    GH_TOKEN_EXTRACTED=$(python3 -c "
import json, sys
try:
    c = json.load(open('/root/.nanobot/config.json'))
    t = c.get('tools',{}).get('mcp',{}).get('servers',{}).get('github',{}).get('env',{}).get('GITHUB_TOKEN','')
    print(t)
except Exception:
    sys.exit(0)
" 2>/dev/null || echo "")
    if [ -n "$GH_TOKEN_EXTRACTED" ]; then
        export GITHUB_TOKEN="$GH_TOKEN_EXTRACTED"
        export GH_TOKEN="$GH_TOKEN_EXTRACTED"
        echo "Extracted GITHUB_TOKEN from config.json"
    fi
fi

echo "Starting nanobot with restart signal support..."
echo "Workspace: $WORKSPACE"

while true; do
    echo "$(date): Starting nanobot..."

    # Run nanobot gateway and capture exit code
    nanobot gateway || exit_code=$?

    # Check for restart signal
    if [ -f "$RESTART_SIGNAL" ]; then
        echo "$(date): Restart signal detected"
        cat "$RESTART_SIGNAL"
        echo ""
        echo "Restarting nanobot..."
        # Signal file is read and cleared by nanobot on startup
        sleep 1
        continue
    fi

    # Normal exit or error
    echo "$(date): Nanobot exited with code: ${exit_code:-0}"

    # Exit if nanobot exited normally (code 0) or with error
    exit ${exit_code:-0}
done
