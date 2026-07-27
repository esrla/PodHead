#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"
source .venv/bin/activate

if pgrep -f 'python.*backhead.main' > /dev/null; then
    echo "PodHead is already running:"
    pgrep -af 'python.*backhead.main'
else
    nohup python -u -m backhead.main >> "$HOME/podhead.log" 2>&1 &
    pid=$!

    echo "PodHead started with PID $pid"
    sleep 1

    if ! kill -0 "$pid" 2>/dev/null; then
        echo "PodHead failed to start. See $HOME/podhead.log"
        exit 1
    fi

    pgrep -af 'python.*backhead.main'
fi

if pgrep -f 'streamlit.*backhead/admin_web.py' > /dev/null; then
    echo "PodHead web is already running:"
    pgrep -af 'streamlit.*backhead/admin_web.py'
else
    nohup streamlit run backhead/admin_web.py \
        --server.address 127.0.0.1 \
        --server.port 8501 \
        --server.headless true \
        >> "$HOME/podhead-web.log" 2>&1 &

    web_pid=$!
    echo "PodHead web started with PID $web_pid"
    sleep 1

    if ! kill -0 "$web_pid" 2>/dev/null; then
        echo "PodHead web failed to start. See $HOME/podhead-web.log"
    fi
fi

if command -v tailscale > /dev/null 2>&1 && tailscale status > /dev/null 2>&1; then
    if sudo tailscale serve --bg --yes http://127.0.0.1:8501; then
        echo "PodHead web is available through Tailscale:"
        sudo tailscale serve status
    else
        echo "Could not configure Tailscale Serve. PodHead continues without web access."
    fi
else
    echo "Tailscale is not installed or connected. PodHead continues without web access."
fi

echo
echo "Following $HOME/podhead.log. Ctrl+C stops log viewing, not PodHead."
tail -n 100 -f "$HOME/podhead.log"
