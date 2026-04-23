# ==========================================
# pipeline.py (ROBUST VERSION)
# ==========================================

from llm import extract_experiment_model
from segmentation import segment_text
from bug_checks import (
    guard_input_text,
    validate_llm_output,
    filter_experiments,
    collect_all_nodes,
)


# ------------------------------------------
# 1. Main entry point
# ------------------------------------------
def analyze_text(text, rel_filter=None, min_confidence=0.0):
    """
    Full pipeline:
    text → LLM → validation → filtering → output
    """

    cleaned_text, input_issues = guard_input_text(text)
    if not cleaned_text:
        return {"experiments": [], "all_nodes": [], "issues": input_issues}

    # --------------------------------------
    # Step 1: Segment text + call LLM
    # --------------------------------------
    segments = segment_text(cleaned_text) or [cleaned_text]

    all_experiments = []
    issues = list(input_issues)

    for segment in segments:
        raw_output = extract_experiment_model(segment)

        if raw_output is None:
            issues.append("LLM returned None for one segment.")
            continue

        clean_output, validation_issues = validate_llm_output(raw_output)
        issues.extend(validation_issues)
        all_experiments.extend(clean_output.get("experiments", []))

    # --------------------------------------
    # Step 2: Optional filtering
    # --------------------------------------
    filtered_output = filter_experiments(
        {"experiments": all_experiments},
        rel_filter=rel_filter,
        min_confidence=min_confidence,
    )

    experiments = filtered_output.get("experiments", [])
    return {
        "experiments": experiments,
        "all_nodes": collect_all_nodes(experiments),
        "issues": issues,
    }


# ------------------------------------------
# 2. Helper: get first experiment (UI friendly)
# ------------------------------------------
def get_first_experiment(data):
    """
    Safely returns first experiment for visualization
    """

    if not data or "experiments" not in data:
        return None

    experiments = data["experiments"]

    if not experiments:
        return None

    return experiments[0]


# ------------------------------------------
# 3. Helper: flatten all experiments (optional)
# ------------------------------------------
def merge_experiments(data):
    """
    Combines multiple experiments into one graph
    """

    all_manipulated = []
    all_measured = []
    all_links = []

    for exp in data.get("experiments", []):

        all_manipulated.extend(exp.get("manipulated_variables", []))
        all_measured.extend(exp.get("measured_variables", []))
        all_links.extend(exp.get("model_links", []))

    return {
        "name": "Merged Experiments",
        "manipulated_variables": list(set(all_manipulated)),
        "measured_variables": list(set(all_measured)),
        "model_links": all_links
    }


# ------------------------------------------
# 4. Debug utility
# ------------------------------------------
def debug_pipeline(text):
    """
    Runs pipeline with prints for debugging
    """

    raw = extract_experiment_model(text)
    print("\n===== RAW LLM OUTPUT =====")
    print(raw)

    clean, validation_issues = validate_llm_output(raw)
    print("\n===== CLEANED OUTPUT =====")
    print(clean)
    if validation_issues:
        print("\n===== ISSUES =====")
        print(validation_issues)

    return clean