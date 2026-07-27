#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"
source .venv/bin/activate

if pgrep -f 'python.*backhead.main' > /dev/null; then
    echo "PodHead is already running:"
    pgrep -af 'python.*backhead.main'
    exit 0
fi

nohup python -u -m backhead.main >> "$HOME/podhead.log" 2>&1 &
pid=$!

echo "PodHead started with PID $pid"
sleep 1

pgrep -af 'python.*backhead.main'
