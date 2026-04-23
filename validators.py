# ==========================================
# validators.py
# Central validation + bug prevention layer
# ==========================================

import re


# ------------------------------------------
# 1. Normalize text safely
# ------------------------------------------
def normalize_var(v):
    if v is None:
        return None
    return str(v).strip().lower().replace(" ", "_")


# ------------------------------------------
# 2. Safe getter (prevents KeyError)
# ------------------------------------------
def safe_get(d, key, default=None):
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


# ------------------------------------------
# 3. Validate a single link
# ------------------------------------------
def validate_link(link):
    """
    Ensures link has required structure.
    Returns cleaned link or None if invalid.
    """

    if not isinstance(link, dict):
        return None

    exp_var = normalize_var(safe_get(link, "experiment_variable"))
    model_comp = normalize_var(safe_get(link, "model_component"))
    rel = safe_get(link, "relationship", "unknown")
    conf = safe_get(link, "confidence", 0.5)

    # Fix confidence
    try:
        conf = float(conf)
    except:
        conf = 0.5

    conf = max(0.0, min(1.0, conf))  # clamp to [0,1]

    # Skip totally empty links
    if not exp_var and not model_comp:
        return None

    return {
        "experiment_variable": exp_var,
        "model_component": model_comp,
        "relationship": rel,
        "confidence": conf
    }


# ------------------------------------------
# 4. Validate experiment structure
# ------------------------------------------
def validate_experiment(exp):
    """
    Cleans and validates a single experiment dict.
    """

    if not isinstance(exp, dict):
        return None

    name = safe_get(exp, "name", "Unnamed Experiment")

    manipulated = safe_get(exp, "manipulated_variables", [])
    measured = safe_get(exp, "measured_variables", [])
    links = safe_get(exp, "model_links", [])

    # Ensure lists
    if not isinstance(manipulated, list):
        manipulated = []
    if not isinstance(measured, list):
        measured = []
    if not isinstance(links, list):
        links = []

    # Normalize variables
    manipulated = [normalize_var(v) for v in manipulated if v]
    measured = [normalize_var(v) for v in measured if v]

    # Validate links
    clean_links = []
    for link in links:
        valid = validate_link(link)
        if valid:
            clean_links.append(valid)

    return {
        "name": name,
        "manipulated_variables": manipulated,
        "measured_variables": measured,
        "model_links": clean_links
    }


# ------------------------------------------
# 5. Validate full LLM output
# ------------------------------------------
def validate_llm_output(data):
    """
    Main entry point.
    Guarantees safe structure for entire pipeline.
    """

    if not isinstance(data, dict):
        print("⚠️ LLM output not a dict")
        return {"experiments": []}

    experiments = data.get("experiments", [])

    if not isinstance(experiments, list):
        print("⚠️ experiments is not a list")
        return {"experiments": []}

    clean_experiments = []

    for exp in experiments:
        valid = validate_experiment(exp)
        if valid:
            clean_experiments.append(valid)

    return {"experiments": clean_experiments}


# ------------------------------------------
# 6. Debug helper (optional)
# ------------------------------------------
def debug_print(data):
    print("\n===== CLEANED DATA =====")
    for exp in data.get("experiments", []):
        print(exp)
    print("========================\n")