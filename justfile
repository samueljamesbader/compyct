# Developer tasks. Run `just` (or `uv run just`) to list them.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# List available recipes
default:
    @just --list

# Build the HTML docs into docs/build/html
docs:
    uv run sphinx-build -b html docs/source docs/build/html
