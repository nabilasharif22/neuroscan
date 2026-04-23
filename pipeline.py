# ==========================================
# pipeline.py (ROBUST VERSION)
# ==========================================

from llm import extract_experiment_model, get_llm_status
from segmentation import segment_text, extract_paper_context
from bug_checks import (
    guard_input_text,
    validate_llm_output,
    filter_experiments,
    collect_all_nodes,
    normalize_var,
)


# ------------------------------------------
# 1. Main entry point
# ------------------------------------------
def analyze_text(
    text,
    rel_filter=None,
    min_confidence=0.0,
    max_segments=None,
    ml_score_threshold=0.38,
    llm_top_k=None,
):
    """
    Full pipeline:
    text → LLM → validation → filtering → output
    """

    initial_llm_status = get_llm_status()

    cleaned_text, input_issues = guard_input_text(text)
    if not cleaned_text:
        return {
            "experiments": [],
            "all_nodes": [],
            "issues": input_issues,
            "llm_status": initial_llm_status,
        }

    # --------------------------------------
    # Step 1: Segment text + call LLM
    # --------------------------------------
    segments = segment_text(
        cleaned_text,
        max_segments=max_segments,
        ml_score_threshold=ml_score_threshold,
        llm_top_k=llm_top_k,
    ) or [cleaned_text]

    # Extract title+abstract once — prepended to every LLM prompt so
    # each chunk knows which paper it belongs to (cross-chunk continuity).
    paper_context = extract_paper_context(cleaned_text)

    all_experiments = []
    issues = list(input_issues)

    for segment in segments:
        raw_output = extract_experiment_model(segment, paper_context=paper_context)

        if raw_output is None:
            issues.append("LLM returned None for one segment.")
            continue

        clean_output, validation_issues = validate_llm_output(raw_output)
        issues.extend(validation_issues)
        all_experiments.extend(clean_output.get("experiments", []))

    all_experiments = _merge_similar_experiments(all_experiments)

    # --------------------------------------
    # Step 2: Optional filtering
    # --------------------------------------
    filtered_output = filter_experiments(
        {"experiments": all_experiments},
        rel_filter=rel_filter,
        min_confidence=min_confidence,
    )

    experiments = filtered_output.get("experiments", [])
    final_llm_status = get_llm_status()
    return {
        "experiments": experiments,
        "all_nodes": collect_all_nodes(experiments),
        "issues": issues,
        "llm_status": final_llm_status,
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


def _experiment_group_key(experiment):
    name = normalize_var(experiment.get("name", "unnamed_experiment")) or "unnamed_experiment"
    tested_model = experiment.get("tested_model", {}) or {}
    model_name = normalize_var(tested_model.get("name")) or "unknown_model"
    manipulated = sorted({normalize_var(item) for item in experiment.get("manipulated_variables", []) if normalize_var(item)})
    measured = sorted({normalize_var(item) for item in experiment.get("measured_variables", []) if normalize_var(item)})
    return (model_name, name, tuple(manipulated), tuple(measured))


def _merge_similar_experiments(experiments):
    grouped = {}

    for experiment in experiments:
        key = _experiment_group_key(experiment)
        if key not in grouped:
            grouped[key] = {
                "name": experiment.get("name", "Unnamed Experiment"),
                "tested_model": dict(experiment.get("tested_model", {}) or {}),
                "manipulated_variables": list(experiment.get("manipulated_variables", [])),
                "measured_variables": list(experiment.get("measured_variables", [])),
                "model_links": list(experiment.get("model_links", [])),
                "outcome_links": list(experiment.get("outcome_links", [])),
            }
            continue

        grouped[key]["manipulated_variables"].extend(experiment.get("manipulated_variables", []))
        grouped[key]["measured_variables"].extend(experiment.get("measured_variables", []))
        grouped[key]["model_links"].extend(experiment.get("model_links", []))
        grouped[key]["outcome_links"].extend(experiment.get("outcome_links", []))

        existing_model = grouped[key].get("tested_model", {})
        incoming_model = experiment.get("tested_model", {}) or {}
        if not existing_model.get("evidence") and incoming_model.get("evidence"):
            grouped[key]["tested_model"] = dict(incoming_model)

    merged = []
    for item in grouped.values():
        unique_manipulated = sorted({value for value in item["manipulated_variables"] if value})
        unique_measured = sorted({value for value in item["measured_variables"] if value})

        model_seen = set()
        model_links = []
        for link in item.get("model_links", []):
            signature = (
                link.get("experiment_variable"),
                link.get("model_component"),
                link.get("relationship"),
                link.get("confidence"),
            )
            if signature in model_seen:
                continue
            model_seen.add(signature)
            model_links.append(link)

        outcome_seen = set()
        outcome_links = []
        for link in item.get("outcome_links", []):
            signature = (
                link.get("model_component"),
                link.get("measured_variable"),
                link.get("relationship"),
                link.get("confidence"),
            )
            if signature in outcome_seen:
                continue
            outcome_seen.add(signature)
            outcome_links.append(link)

        merged.append(
            {
                "name": item.get("name", "Unnamed Experiment"),
                "tested_model": dict(item.get("tested_model", {}) or {}),
                "manipulated_variables": unique_manipulated,
                "measured_variables": unique_measured,
                "model_links": model_links,
                "outcome_links": outcome_links,
            }
        )

    return _merge_by_tested_model(merged)


def _merge_by_tested_model(experiments):
    grouped = {}

    for experiment in experiments:
        tested_model = experiment.get("tested_model", {}) or {}
        model_name = normalize_var(tested_model.get("name"))

        if not model_name or model_name == "unknown_model":
            key = ("unknown", normalize_var(experiment.get("name", "unnamed_experiment")) or "unnamed_experiment")
        else:
            key = ("model", model_name)

        if key not in grouped:
            grouped[key] = {
                "name": experiment.get("name", "Unnamed Experiment"),
                "tested_model": dict(tested_model),
                "manipulated_variables": list(experiment.get("manipulated_variables", [])),
                "measured_variables": list(experiment.get("measured_variables", [])),
                "model_links": list(experiment.get("model_links", [])),
                "outcome_links": list(experiment.get("outcome_links", [])),
            }
            continue

        grouped[key]["manipulated_variables"].extend(experiment.get("manipulated_variables", []))
        grouped[key]["measured_variables"].extend(experiment.get("measured_variables", []))
        grouped[key]["model_links"].extend(experiment.get("model_links", []))
        grouped[key]["outcome_links"].extend(experiment.get("outcome_links", []))

    merged = []
    for item in grouped.values():
        model_name = normalize_var((item.get("tested_model", {}) or {}).get("name"))
        display_name = item.get("name", "Unnamed Experiment")
        if model_name and model_name != "unknown_model":
            display_name = f"{model_name.replace('_', ' ').title()} — consolidated"

        model_seen = set()
        dedup_model_links = []
        for link in item.get("model_links", []):
            signature = (
                link.get("experiment_variable"),
                link.get("model_component"),
                link.get("relationship"),
                link.get("confidence"),
            )
            if signature in model_seen:
                continue
            model_seen.add(signature)
            dedup_model_links.append(link)

        outcome_seen = set()
        dedup_outcome_links = []
        for link in item.get("outcome_links", []):
            signature = (
                link.get("model_component"),
                link.get("measured_variable"),
                link.get("relationship"),
                link.get("confidence"),
            )
            if signature in outcome_seen:
                continue
            outcome_seen.add(signature)
            dedup_outcome_links.append(link)

        merged.append(
            {
                "name": display_name,
                "tested_model": dict(item.get("tested_model", {}) or {}),
                "manipulated_variables": sorted({v for v in item.get("manipulated_variables", []) if v}),
                "measured_variables": sorted({v for v in item.get("measured_variables", []) if v}),
                "model_links": dedup_model_links,
                "outcome_links": dedup_outcome_links,
            }
        )

    return merged


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