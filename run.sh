#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$ROOT_DIR/venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "❌ Missing virtual environment. Run ./setup.sh first."
  exit 1
fi

exec "$VENV_PYTHON" -m streamlit run "$ROOT_DIR/app.py"
