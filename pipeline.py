# ==========================================
# MAIN PIPELINE LOGIC
# ==========================================

from segmentation import segment_text
from ml_model import load_model, score_text
from llm import extract_experiment_model

# Load ML model once
model = load_model()

def analyze_text(text, mode):

    segments = segment_text(text)

    experiments = []
    all_nodes = set()

    for seg in segments:

        # ----------------------------------
        # ML FILTER (score-based)
        # ----------------------------------
        score = score_text(model, seg)

        if score < 0:   # threshold
            continue

        # ----------------------------------
        # LLM EXTRACTION
        # ----------------------------------
        data = extract_experiment_model(seg)

        for exp in data["experiments"]:
            experiments.append(exp)

            # collect nodes
            all_nodes.update(exp["manipulated_variables"])
            all_nodes.update(exp["measured_variables"])

            for link in exp["model_links"]:
                all_nodes.add(link["model_component"])

    return {
        "experiments": experiments,
        "all_nodes": list(all_nodes)
    }