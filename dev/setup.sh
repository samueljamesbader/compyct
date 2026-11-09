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

if [ ! -d packages/VACASK ]; then
    git clone https://codeberg.org/arpadbuermen/VACASK.git packages/VACASK
fi

# OpenVAF-Reloaded Verilog-A compiler, needed to build VACASK
OPENVAF_VERSION="osdi_0.5_20260820-7-gde9c640b"
OPENVAF_URL="https://fides.fe.uni-lj.si/openvaf/download/openvaf-reloaded-$OPENVAF_VERSION"
OPENVAF_DIR="$PWD/.tools/openvaf"
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) OPENVAF_BIN="$OPENVAF_DIR/openvaf-r.exe" ;;
    *)                    OPENVAF_BIN="$OPENVAF_DIR/openvaf-r" ;;
esac
if [ ! -f "$OPENVAF_BIN" ]; then
    echo "Downloading OpenVAF-Reloaded $OPENVAF_VERSION..."
    mkdir -p "$OPENVAF_DIR"
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            curl -fsSL -o "$OPENVAF_DIR/openvaf.zip" "$OPENVAF_URL-win_x64.zip"
            unzip -o -q "$OPENVAF_DIR/openvaf.zip" -d "$OPENVAF_DIR"
            rm "$OPENVAF_DIR/openvaf.zip"
            ;;
        Linux)
            curl -fsSL "$OPENVAF_URL-linux_x64.tar.gz" | tar -xz -C "$OPENVAF_DIR"
            ;;
        *)
            echo "Error: no prebuilt OpenVAF for $(uname -s); build it from source." >&2
            exit 1
            ;;
    esac
fi
export SKBUILD_CMAKE_DEFINE="OPENVAF_DIR=$OPENVAF_DIR"

uv sync --extra dev
