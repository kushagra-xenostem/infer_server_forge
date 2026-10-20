#!/bin/bash
# Run a command inside the no-network sandbox that executes model-generated code.
# Usage: sandbox.sh <workdir-to-mount-rw> <command...>
# - host venv site-packages mounted read-only as the only Python deps
# - evalplus dataset cache mounted (rw: evaluate writes ground-truth cache)
# - no network, capped memory/pids, auto-removed
set -eu
WORK=$(realpath "$1"); shift
SITE=/data/projects/llm-serving/.venv/lib/python3.12/site-packages
CACHE=$HOME/.cache/evalplus

exec docker run --rm --network none \
    --memory 6g --memory-swap 6g --pids-limit 512 --cpus 4 \
    --security-opt no-new-privileges --cap-drop ALL \
    -u "$(id -u):$(id -g)" \
    -v "$SITE":/deps:ro \
    -v "$CACHE":/home/runner/.cache/evalplus \
    -v "$WORK":/work \
    -e PYTHONPATH=/deps -e HOME=/home/runner -e PATH=/deps/bin:/usr/local/bin:/usr/bin:/bin \
    -w /work llm-eval-sandbox:1 "$@"
