# ==========================================
# bug_checks.py
# End-to-end validation + normalization layer
# ==========================================

from __future__ import annotations

import math

ALLOWED_RELATIONSHIPS = {"tests", "correlates", "controls", "modulates", "causes", "unknown"}


def normalize_var(value):
    if value is None:
        return None
    text = str(value).strip().lower().replace(" ", "_")
    return text or None


def normalize_relationship(value):
    rel = normalize_var(value)
    if not rel:
        return "unknown"
    return rel if rel in ALLOWED_RELATIONSHIPS else "unknown"


def normalize_rel_filter(rel_filter):
    if rel_filter is None:
        return set()

    if isinstance(rel_filter, str):
        rel_filter = [rel_filter]

    if not isinstance(rel_filter, (list, tuple, set)):
        return set()

    normalized = {normalize_relationship(item) for item in rel_filter}
    return {item for item in normalized if item != "unknown"}


def sanitize_confidence(value, default=0.5):
    try:
        conf = float(value)
        if math.isnan(conf):
            return default
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, conf))


def guard_input_text(text):
    if text is None:
        return "", ["Input text was None; replaced with empty string."]

    cleaned = str(text).strip()
    issues = []

    if not cleaned:
        issues.append("Input text is empty.")

    return cleaned, issues


def _safe_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def validate_link(link):
    if not isinstance(link, dict):
        return None

    experiment_variable = normalize_var(link.get("experiment_variable"))
    model_component = normalize_var(link.get("model_component"))

    if not experiment_variable and not model_component:
        return None

    return {
        "experiment_variable": experiment_variable,
        "model_component": model_component,
        "relationship": normalize_relationship(link.get("relationship", "unknown")),
        "confidence": sanitize_confidence(link.get("confidence", 0.5)),
    }


def validate_outcome_link(link):
    if not isinstance(link, dict):
        return None

    model_component = normalize_var(link.get("model_component"))
    measured_variable = normalize_var(link.get("measured_variable"))

    if not model_component or not measured_variable:
        return None

    return {
        "model_component": model_component,
        "measured_variable": measured_variable,
        "relationship": normalize_relationship(link.get("relationship", "unknown")),
        "confidence": sanitize_confidence(link.get("confidence", 0.5)),
    }


def validate_experiment(experiment, default_name="Unnamed Experiment"):
    if not isinstance(experiment, dict):
        return None

    name = str(experiment.get("name", default_name)).strip() or default_name

    manipulated = []
    for value in _safe_list(experiment.get("manipulated_variables", [])):
        normalized = normalize_var(value)
        if normalized:
            manipulated.append(normalized)

    measured = []
    for value in _safe_list(experiment.get("measured_variables", [])):
        normalized = normalize_var(value)
        if normalized:
            measured.append(normalized)

    links = []
    seen_links = set()
    for link in _safe_list(experiment.get("model_links", [])):
        cleaned = validate_link(link)
        if not cleaned:
            continue

        signature = (
            cleaned["experiment_variable"],
            cleaned["model_component"],
            cleaned["relationship"],
            cleaned["confidence"],
        )
        if signature in seen_links:
            continue
        seen_links.add(signature)
        links.append(cleaned)

    outcome_links = []
    seen_outcome_links = set()
    for link in _safe_list(experiment.get("outcome_links", [])):
        cleaned = validate_outcome_link(link)
        if not cleaned:
            continue

        signature = (
            cleaned["model_component"],
            cleaned["measured_variable"],
            cleaned["relationship"],
            cleaned["confidence"],
        )
        if signature in seen_outcome_links:
            continue
        seen_outcome_links.add(signature)
        outcome_links.append(cleaned)

    for link in links:
        exp_var = link.get("experiment_variable")
        if exp_var and exp_var not in manipulated and exp_var not in measured:
            manipulated.append(exp_var)

    for link in outcome_links:
        measured_var = link.get("measured_variable")
        if measured_var and measured_var not in measured:
            measured.append(measured_var)

    return {
        "name": name,
        "manipulated_variables": sorted(set(manipulated)),
        "measured_variables": sorted(set(measured)),
        "model_links": links,
        "outcome_links": outcome_links,
    }


def validate_llm_output(payload):
    if not isinstance(payload, dict):
        return {"experiments": []}, ["LLM payload is not a dictionary."]

    experiments = payload.get("experiments", [])
    if not isinstance(experiments, list):
        return {"experiments": []}, ["LLM field 'experiments' is not a list."]

    cleaned_experiments = []
    issues = []

    for item in experiments:
        cleaned = validate_experiment(item)
        if cleaned:
            cleaned_experiments.append(cleaned)
        else:
            issues.append("Dropped an invalid experiment object.")

    return {"experiments": cleaned_experiments}, issues


def filter_experiments(data, rel_filter=None, min_confidence=0.0):
    rel_filter_set = normalize_rel_filter(rel_filter)
    min_conf = sanitize_confidence(min_confidence, default=0.0)

    filtered_experiments = []

    for experiment in data.get("experiments", []):
        kept_links = []
        for link in experiment.get("model_links", []):
            relationship = normalize_relationship(link.get("relationship"))
            confidence = sanitize_confidence(link.get("confidence", 0.5))

            if rel_filter_set and relationship not in rel_filter_set:
                continue
            if confidence < min_conf:
                continue

            kept_links.append(
                {
                    "experiment_variable": normalize_var(link.get("experiment_variable")),
                    "model_component": normalize_var(link.get("model_component")),
                    "relationship": relationship,
                    "confidence": confidence,
                }
            )

        kept_outcome_links = []
        for link in experiment.get("outcome_links", []):
            relationship = normalize_relationship(link.get("relationship"))
            confidence = sanitize_confidence(link.get("confidence", 0.5))

            if rel_filter_set and relationship not in rel_filter_set:
                continue
            if confidence < min_conf:
                continue

            kept_outcome_links.append(
                {
                    "model_component": normalize_var(link.get("model_component")),
                    "measured_variable": normalize_var(link.get("measured_variable")),
                    "relationship": relationship,
                    "confidence": confidence,
                }
            )

        if kept_links or kept_outcome_links:
            experiment_copy = dict(experiment)
            experiment_copy["model_links"] = kept_links
            experiment_copy["outcome_links"] = kept_outcome_links
            filtered_experiments.append(experiment_copy)

    return {"experiments": filtered_experiments}


def collect_all_nodes(experiments):
    nodes = set()
    for experiment in experiments:
        for value in experiment.get("manipulated_variables", []):
            normalized = normalize_var(value)
            if normalized:
                nodes.add(normalized)
        for value in experiment.get("measured_variables", []):
            normalized = normalize_var(value)
            if normalized:
                nodes.add(normalized)
        for link in experiment.get("model_links", []):
            source = normalize_var(link.get("experiment_variable"))
            target = normalize_var(link.get("model_component"))
            if source:
                nodes.add(source)
            if target:
                nodes.add(target)
        for link in experiment.get("outcome_links", []):
            source = normalize_var(link.get("model_component"))
            target = normalize_var(link.get("measured_variable"))
            if source:
                nodes.add(source)
            if target:
                nodes.add(target)
    return sorted(nodes)


def build_safe_graph(experiment, rel_filter=None):
    cleaned = validate_experiment(experiment)
    if not cleaned:
        return {
            "inputs": set(),
            "outputs": set(),
            "all_nodes": set(),
            "edges": [],
        }

    rel_filter_set = normalize_rel_filter(rel_filter)
    inputs = set(cleaned.get("manipulated_variables", []))
    outputs = set(cleaned.get("measured_variables", []))
    all_nodes = set(inputs) | set(outputs)
    model_nodes = set()
    edges = []

    for link in cleaned.get("model_links", []):
        relationship = normalize_relationship(link.get("relationship"))
        if rel_filter_set and relationship not in rel_filter_set:
            continue

        source = normalize_var(link.get("experiment_variable"))
        target = normalize_var(link.get("model_component"))

        if not source or not target:
            continue

        all_nodes.add(source)
        all_nodes.add(target)
        model_nodes.add(target)

        edges.append(
            {
                "source": source,
                "target": target,
                "type": relationship,
                "confidence": sanitize_confidence(link.get("confidence", 0.5)),
                "kind": "input_to_model",
            }
        )

    for link in cleaned.get("outcome_links", []):
        relationship = normalize_relationship(link.get("relationship"))
        if rel_filter_set and relationship not in rel_filter_set:
            continue

        source = normalize_var(link.get("model_component"))
        target = normalize_var(link.get("measured_variable"))

        if not source or not target:
            continue

        all_nodes.add(source)
        all_nodes.add(target)
        model_nodes.add(source)

        edges.append(
            {
                "source": source,
                "target": target,
                "type": relationship,
                "confidence": sanitize_confidence(link.get("confidence", 0.5)),
                "kind": "model_to_output",
            }
        )

    return {
        "inputs": inputs,
        "outputs": outputs,
        "model_nodes": model_nodes,
        "all_nodes": all_nodes,
        "edges": edges,
    }