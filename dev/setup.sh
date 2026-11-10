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

uv python install 3.14

if [ ! -d packages/InSpice ]; then
    git clone https://github.com/samueljamesbader/InSpice packages/InSpice
fi

if [ ! -d packages/VACASK ]; then
    git clone https://codeberg.org/arpadbuermen/VACASK.git packages/VACASK
fi

# VACASK is installed from a prebuilt wheel (see [tool.uv.sources] in pyproject.toml)
if ! ls wheelhouse/repaired/vacask-*.whl >/dev/null 2>&1; then
    echo "Error: no VACASK wheel found in wheelhouse/repaired/." >&2
    echo "Obtain one for your platform and place it there." >&2
    exit 1
fi

uv sync --python 3.14 --extra dev
