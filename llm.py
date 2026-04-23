
# ==========================================
# llm.py
# ==========================================

# 1. Load environment variables
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
from groq import Groq

load_dotenv()

# 2. Get API keys from .env
groq_api_key   = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Model names (configurable via .env)
groq_model_name        = os.getenv("GROQ_MODEL",         "llama-3.3-70b-versatile")
groq_model_fallback    = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")
openai_model_name      = os.getenv("OPENAI_MODEL",        "gpt-4o-mini")

# 3. Create clients — Groq preferred, OpenAI as fallback
groq_client   = Groq(api_key=groq_api_key)    if groq_api_key   else None
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Backwards-compat alias used in older code paths
client     = openai_client
model_name = openai_model_name

if groq_client:
    _runtime_mode    = "api"
    _runtime_message = f"Groq API is configured and will be used (model: {groq_model_name})."
elif openai_client:
    _runtime_mode    = "api"
    _runtime_message = f"OpenAI API is configured and will be used (model: {openai_model_name})."
else:
    _runtime_mode    = "mock"
    _runtime_message = "Mock extraction is active. No API key is configured."

# ==========================================
# LLM EXTRACTION (MOCK VERSION)
# ==========================================

def get_llm_status():
    _api_configured = bool(groq_api_key or openai_api_key)
    if groq_api_key:
        _model = groq_model_name
    elif openai_api_key:
        _model = openai_model_name
    else:
        _model = None
    return {
        "mode": _runtime_mode,
        "api_configured": _api_configured,
        "message": _runtime_message,
        "model": _model,
    }


def _mock_output():
    return {
        "experiments": [
            {
                "name": "Example Experiment",
                "manipulated_variables": ["dopamine"],
                "measured_variables": ["learning"],
                "model_links": [
                    {
                        "experiment_variable": "dopamine",
                        "model_component": "prediction error",
                        "relationship": "tests",
                        "confidence": 0.9,
                    },
                    {
                        "experiment_variable": "neural activity",
                        "model_component": "belief",
                        "relationship": "correlates",
                        "confidence": 0.7,
                    },
                ],
                "outcome_links": [
                    {
                        "model_component": "prediction error",
                        "measured_variable": "learning",
                        "relationship": "causes",
                        "confidence": 0.82,
                    }
                ],
            }
        ]
    }


def _extract_with_openai(text, paper_context=""):
    response = openai_client.chat.completions.create(
        model=openai_model_name,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a precise information extraction system."},
            {"role": "user",   "content": _build_prompt(text, paper_context)},
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content if response and response.choices else "{}"
    return json.loads(content or "{}")

def _build_prompt(text, paper_context=""):
    context_block = ""
    if paper_context:
        context_block = (
            "Paper context (title + abstract — for reference only, do NOT extract experiments from this block):\n"
            f"{paper_context}\n\n"
            "--- End of context block ---\n\n"
        )
    return (
        "You extract experiment-model structure from neuroscience text.\n\n"
        "Focus on computational-neuroscience and cognitive-neuroscience experiments.\n"
        "Prioritize causal/mechanistic statements over background narrative.\n\n"
        + context_block
        + "Return ONLY valid JSON with this exact top-level schema:\n"
        "{\n"
        '  "experiments": [\n'
        "    {\n"
        '      "name": "string",\n'
        '      "manipulated_variables": ["string"],\n'
        '      "measured_variables": ["string"],\n'
        '      "model_links": [\n'
        "        {\n"
        '          "experiment_variable": "string",\n'
        '          "model_component": "string",\n'
        '          "relationship": "tests|correlates|controls|modulates|causes|unknown",\n'
        '          "confidence": 0.0\n'
        "        }\n"
        "      ],\n"
        '      "outcome_links": [\n'
        "        {\n"
        '          "model_component": "string",\n'
        '          "measured_variable": "string",\n'
        '          "relationship": "tests|correlates|controls|modulates|causes|unknown",\n'
        '          "confidence": 0.0\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Extract specific variables, not generic words like 'study', 'group', or 'task'.\n"
        "- Keep variable names concise and canonical (e.g., 'dopamine', 'prediction error', 'choice accuracy').\n"
        "- Treat interventions/stimuli/drug doses as manipulated variables when explicit.\n"
        "- Treat behavioral/neural readouts as measured variables when explicit.\n"
        "- Use outcome_links to connect model components to measured variables whenever evidence exists.\n"
        "- If multiple experiments exist, create separate experiment objects.\n"
        "- If uncertain, still return best-effort fields.\n"
        "- Keep confidence in [0.0, 1.0].\n"
        "- No markdown fences, no commentary.\n\n"
        f"Text:\n{text}"
    )


def _extract_with_groq(text, model=None, paper_context=""):
    use_model = model or groq_model_name
    response = groq_client.chat.completions.create(
        model=use_model,
        messages=[
            {"role": "system", "content": "You are a precise information extraction system. Return only valid JSON."},
            {"role": "user",   "content": _build_prompt(text, paper_context)},
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content if response and response.choices else "{}"
    # Strip markdown fences if model wraps output anyway
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(content or "{}")


def extract_experiment_model(text, paper_context=""):
    """
    Provider priority: Groq → OpenAI → mock.
    Falls back automatically on any error.
    paper_context: optional title+abstract string prepended to every prompt
    for cross-chunk continuity.
    """
    global _runtime_mode, _runtime_message

    # --- Try Groq first ---
    if groq_client:
        try:
            data = _extract_with_groq(text, paper_context=paper_context)
            _runtime_mode    = "api"
            _runtime_message = f"Groq extraction succeeded (model: {groq_model_name})."
            return data
        except Exception as exc:
            exc_name = type(exc).__name__
            groq_err = f"{exc_name}: {exc}"

            # On rate limit, transparently retry with smaller model
            if exc_name == "RateLimitError" and groq_model_fallback != groq_model_name:
                try:
                    data = _extract_with_groq(text, model=groq_model_fallback, paper_context=paper_context)
                    _runtime_mode    = "api"
                    _runtime_message = f"Groq rate limit hit on {groq_model_name}; succeeded with fallback model {groq_model_fallback}."
                    return data
                except Exception as exc2:
                    groq_err = f"{exc_name} on both Groq models: {exc2}"

        # Try OpenAI as fallback if configured
        if openai_client:
            try:
                data = _extract_with_openai(text, paper_context=paper_context)
                _runtime_mode    = "api"
                _runtime_message = f"Groq failed ({groq_err}); OpenAI fallback succeeded."
                return data
            except Exception as exc2:
                _runtime_mode    = "mock"
                _runtime_message = f"Both APIs failed. Groq: {groq_err} | OpenAI: {type(exc2).__name__}"
                return _mock_output()

        _runtime_mode    = "mock"
        _runtime_message = f"Groq failed ({groq_err}); no OpenAI key configured. Using mock."
        return _mock_output()

    # --- Try OpenAI only ---
    if openai_client:
        try:
            data = _extract_with_openai(text, paper_context=paper_context)
            _runtime_mode    = "api"
            _runtime_message = f"OpenAI extraction succeeded (model: {openai_model_name})."
            return data
        except Exception as exc:
            _runtime_mode    = "mock"
            _runtime_message = f"OpenAI API failed; using mock output. ({type(exc).__name__})"
            return _mock_output()

    _runtime_mode    = "mock"
    _runtime_message = "Mock extraction is active. No API key is configured."
    return _mock_output()