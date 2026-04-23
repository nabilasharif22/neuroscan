# ==========================================
# SKLEARN MODEL (RELEVANCE DETECTION)
# ==========================================

import re
import math
from functools import lru_cache

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC
    _SKLEARN_AVAILABLE = True
except Exception:
    TfidfVectorizer = None
    FeatureUnion = None
    Pipeline = None
    LinearSVC = None
    _SKLEARN_AVAILABLE = False


MODEL_SIGNATURES = {
    "reinforcement_learning": {
        "family": "reinforcement-learning",
        "terms": [
            "reinforcement learning", "model-free", "model-based", "q-learning",
            "actor-critic", "temporal difference", "td learning", "rescorla-wagner",
            "learning rate", "value update",
        ],
    },
    "bayesian_inference": {
        "family": "bayesian",
        "terms": [
            "bayesian", "hierarchical bayesian", "prior", "posterior", "likelihood",
            "precision", "uncertainty", "volatility", "belief updating", "kalman filter",
        ],
    },
    "active_inference": {
        "family": "active-inference",
        "terms": [
            "active inference", "free energy", "expected free energy", "policy precision",
            "generative model",
        ],
    },
    "drift_diffusion_model": {
        "family": "sequential-sampling",
        "terms": [
            "drift diffusion", "ddm", "drift rate", "boundary separation", "non-decision time",
        ],
    },
    "prospect_theory": {
        "family": "decision-theory",
        "terms": [
            "prospect theory", "loss aversion", "risk preference", "utility curvature",
        ],
    },
}

# ------------------------------------------
# BUILD MODEL
# ------------------------------------------
def build_model():
    """
    Creates TF-IDF + SVM pipeline
    """

    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is not installed; relevance SVM model is unavailable.")

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1,2),
        stop_words="english",
        min_df=1
    )

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3,5)
    )

    features = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec)
    ])

    model = Pipeline([
        ("features", features),
        ("clf", LinearSVC())
    ])

    return model

# ------------------------------------------
# TRAIN MODEL (tiny demo dataset)
# ------------------------------------------
def load_model():
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is not installed; relevance SVM model is unavailable.")

    model = build_model()

    texts = [
        "model predicts behavior",
        "belief updating under uncertainty",
        "reward prediction error drives learning",
        "animals were housed in cages",
        "microscopy imaging was used",
        "subjects were trained for days"
    ]

    labels = [
        1,1,1,   # relevant
        0,0,0    # not relevant
    ]

    model.fit(texts, labels)
    return model

# ------------------------------------------
# SCORE TEXT
# ------------------------------------------
def score_text(model, text):
    """
    Returns decision score (higher = more relevant)
    """
    if not _SKLEARN_AVAILABLE:
        return 0.0
    return model.decision_function([text])[0]


@lru_cache(maxsize=1)
def get_relevance_model():
    if not _SKLEARN_AVAILABLE:
        return None
    try:
        return load_model()
    except Exception:
        return None


def normalize_decision_score(raw_score):
    clipped = max(-8.0, min(8.0, float(raw_score)))
    return 1.0 / (1.0 + math.exp(-clipped))


def get_relevance_score(text, model=None):
    active_model = model if model is not None else get_relevance_model()
    if active_model is None:
        return 0.5
    raw_score = score_text(active_model, text)
    return normalize_decision_score(raw_score)


def identify_candidate_models(text, top_k=4):
    """
    Detect likely tested computational models from paper text.
    Returns ranked candidates with model name, family, score, and evidence terms.
    """
    lower = str(text or "").lower()
    candidates = []

    for model_name, spec in MODEL_SIGNATURES.items():
        evidence = []
        score = 0.0
        for term in spec.get("terms", []):
            count = len(re.findall(rf"\b{re.escape(term)}\b", lower))
            if count > 0:
                evidence.append(term)
                score += 1.0 + min(0.6, 0.15 * (count - 1))

        if score > 0:
            candidates.append(
                {
                    "name": model_name,
                    "family": spec.get("family", "unknown"),
                    "score": round(score, 3),
                    "evidence_terms": evidence[:6],
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:max(1, int(top_k))]


def build_model_primer(text, top_k=3):
    """
    Builds a short model-context block for prompts, e.g.:
    "Candidate tested models: reinforcement_learning (reinforcement-learning; evidence: ...), ..."
    """
    candidates = identify_candidate_models(text, top_k=top_k)
    if not candidates:
        return ""

    entries = []
    for candidate in candidates:
        evidence = ", ".join(candidate.get("evidence_terms", [])[:3])
        entries.append(
            f"{candidate['name']} ({candidate['family']}; evidence: {evidence})"
        )

    return "Candidate tested models: " + "; ".join(entries)