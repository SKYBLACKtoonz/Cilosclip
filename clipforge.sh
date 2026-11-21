#!/usr/bin/env bash
# Launch ClipForge from anywhere (desktop shortcut, terminal, file manager).
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
cd "$HERE"

# The package is not pip-installed system-wide, so make it importable here.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

if [ -x "$HERE/.venv/bin/python" ]; then
    PY="$HERE/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    echo "Python 3 was not found. Install Python 3.10+ first." >&2
    exit 1
fi

exec "$PY" -m clipforge gui "$@"
