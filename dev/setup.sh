#!/usr/bin/env sh
# Install uv (if needed) and sync the project environment.
set -eu

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "Error: need curl or wget to install uv." >&2
        exit 1
    fi
    # Make uv available in this shell without restarting
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$(dirname "$0")/.."

if [ ! -d packages/InSpice ]; then
    git clone https://github.com/samueljamesbader/InSpice packages/InSpice
fi

uv sync --extra dev
