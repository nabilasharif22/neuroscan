# Neuro Model Mapper

## Quick Start (Shareable)
```zsh
./setup.sh
./run.sh
```

## What `setup.sh` does
- Creates `venv` automatically
- Installs dependencies
- Applies a compatibility fallback if your machine cannot build `pyarrow`

## Manual Install (optional)
```zsh
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m streamlit run app.py
```

## Optional ML Extras
```zsh
./venv/bin/pip install -r requirements-ml.txt
```

## Quick Robustness Check
```zsh
./venv/bin/python sanity_check.py
```

## Features
- Extract experiment-model relationships
- Interactive filtering
- Highlight causal paths
- End-to-end bug checks for malformed LLM output
- Stable graph rendering even with incomplete links

## Next Steps
- Add OpenAI API in llm.py
- Improve dataset
- Add PDF upload