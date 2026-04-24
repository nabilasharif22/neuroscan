# NeuroScan

NeuroScan is a Streamlit app that extracts structured experiment/model information from neuroscience papers and visualizes it as an interpretable graph.

## What this project does

Given pasted text or an uploaded paper (`.pdf`/`.txt`), NeuroScan:

- Identifies experimental manipulations and measured variables
- Infers tested novel and established model context
- Extracts model-related links (input → model → outcome)
- Validates and filters extracted relationships by type/confidence
- Renders an interactive graph with controls for readability and inspection

## How to use it

1. Open the app in your browser.
2. Upload a paper file or paste manuscript text.
3. Choose relationship filters and confidence threshold.
4. Tune extraction controls (speed, ML relevance threshold, top-K chunks) as needed.
5. Click **Run Analysis**.
6. Review extracted experiments and diagrams.

## Run locally

### Quick start (recommended)

```zsh
./setup.sh
./run.sh
```

The app will be available at:

- `http://localhost:8501`

### What `setup.sh` does

- Creates a local `venv` virtual environment (if missing)
- Upgrades `pip`
- Installs dependencies from `requirements.txt`
- Applies a compatibility fallback install path if standard install fails on your machine

### Manual setup (optional)

```zsh
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m streamlit run app.py
```

### Optional extras

```zsh
./venv/bin/pip install -r requirements-ml.txt
```

### Quick sanity check

```zsh
./venv/bin/python sanity_check.py
```

## Secrets and API keys

NeuroScan supports API-based extraction via environment variables.

Expected keys:

- `GROQ_API_KEY`
- `OPENAI_API_KEY`

Optional model overrides:

- `GROQ_MODEL`
- `GROQ_MODEL_FALLBACK`
- `OPENAI_MODEL`

### Recommended local setup

Create a local `.env` file in the project root (not committed):

```zsh
cat > .env << 'EOF'
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
EOF
```

### Security notes

- Do **not** hardcode API keys in source files.
- Do **not** commit `.env` to version control.
- If keys are absent or API calls fail, app logic may fall back to mock extraction mode.

