# Developers

## Set up the environment

Compyct uses [uv](https://docs.astral.sh/uv/) for project and dependency
management. The setup scripts in `dev/` install uv if it is not already on your
`PATH`, clone [InSpice](https://github.com/samueljamesbader/InSpice) into
`packages/InSpice` if it is not already there, then synchronize the project
environment. InSpice is a uv workspace member, so it is installed in editable
mode (with its `dev` extra) and local changes to it take effect immediately.

On Linux or macOS:

```console
sh dev/setup.sh
```

On Windows (PowerShell):

```console
powershell -ExecutionPolicy Bypass -File dev\setup.ps1
```

If you already have uv and have cloned InSpice, you can instead run this from
the repository root:

```console
uv sync --extra dev
```

Either way, this installs the project and its development dependencies,
including Sphinx, MyST Parser, and [just](https://just.systems/).

## Run developer tasks

Common developer tasks are defined as recipes in the `justfile` at the
repository root. Because just is installed into the project environment, run it
through uv:

```console
uv run just
```

With no arguments, this lists the available recipes. If you activate the
virtual environment, or install just globally with `uv tool install rust-just`,
you can drop the `uv run` prefix.

## Build the documentation

```console
uv run just docs
```

This runs the Sphinx HTML builder
(`uv run sphinx-build -b html docs/source docs/build/html`). Open
`docs/build/html/index.html` to view the generated site.

## Project layout

- `src/compyct/` contains the Python package.
- `docs/source/` contains the Sphinx documentation, written in Markdown.
- `dev/` contains the environment setup scripts.
- `packages/InSpice/` is a local clone of InSpice, used as a uv workspace member.
- `justfile` defines developer tasks.
- `pyproject.toml` defines package metadata and development dependencies.
