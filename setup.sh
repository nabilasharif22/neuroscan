#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/venv"

pick_python() {
  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
    return
  fi
  if command -v python3.12 >/dev/null 2>&1; then
    echo "python3.12"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  echo ""
}

PY_CMD="$(pick_python)"
if [[ -z "$PY_CMD" ]]; then
  echo "❌ Python 3 not found. Install Python 3.11+ first."
  exit 1
fi

echo "➡️ Using Python: $PY_CMD"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "➡️ Creating virtual environment at $VENV_DIR"
  "$PY_CMD" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null

echo "➡️ Installing core dependencies"
if ! "$VENV_DIR/bin/pip" install -r "$ROOT_DIR/requirements.txt"; then
  echo "⚠️ Standard install failed. Applying compatibility fallback (skip pyarrow build)."

  "$VENV_DIR/bin/pip" install --no-deps streamlit==1.56.0 plotly openai python-dotenv
  "$VENV_DIR/bin/pip" install \
    altair blinker cachetools click gitpython numpy packaging pandas pillow pydeck protobuf \
    requests tenacity toml tornado typing-extensions anyio distro httpx jiter pydantic \
    sniffio tqdm jinja2 jsonschema narwhals gitdb python-dateutil charset-normalizer \
    idna urllib3 certifi httpcore h11 annotated-types pydantic-core typing-inspection \
    attrs jsonschema-specifications referencing rpds-py six smmap markupsafe
fi

echo "\n✅ Setup complete"
echo "Run the app with:"
echo "  ./run.sh"
