# ==========================================
# sanity_check.py
# Minimal smoke test for robustness layer
# ==========================================

from pipeline import analyze_text
from visualization import draw_experiment_diagram


def run_sanity_check():
    sample_text = """
    Study 1: We manipulated dopamine and measured learning.
    Experiment 2: Neural activity correlates with belief updates.
    """

    results = analyze_text(sample_text, rel_filter=["tests", "correlates"], min_confidence=0.1)

    print("Experiments:", len(results.get("experiments", [])))
    print("Nodes:", len(results.get("all_nodes", [])))
    print("Issues:", len(results.get("issues", [])))

    malformed_experiment = {
        "name": 123,
        "manipulated_variables": [None, " dopamine "],
        "measured_variables": "learning",
        "model_links": [
            {"experiment_variable": "dopamine", "model_component": "prediction error", "relationship": "TESTS", "confidence": "1.2"},
            {"experiment_variable": None, "model_component": None},
            "invalid-link",
        ],
    }

    fig = draw_experiment_diagram(
        malformed_experiment,
        rel_filter=["tests", "controls"],
        selected_node="dopamine",
    )

    print("Figure traces:", len(fig.data))
    print("Sanity check completed.")


if __name__ == "__main__":
    run_sanity_check()