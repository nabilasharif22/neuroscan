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

            "reward prediction error", "rpe", "dopamine", "phasic dopamine",
            "value function", "state value", "action value", "q value",
            "policy", "policy update", "softmax", "inverse temperature",
            "exploration exploitation", "credit assignment",
            "trial-by-trial learning", "prediction error signal",
            "eligibility trace", "discount factor", "gamma",
            "two-step task", "bandit task", "multi-armed bandit",
            "transition model", "state transition", "planning",
            "tree search", "successor representation",
            "computational model fit", "parameter recovery",
            "maximum likelihood", "model comparison", "aic", "bic",
        ],
    },

    "bayesian_inference": {
        "family": "bayesian",
        "terms": [
            "bayesian", "hierarchical bayesian", "prior", "posterior", "likelihood",
            "precision", "uncertainty", "volatility", "belief updating", "kalman filter",

            "bayesian brain", "probabilistic inference",
            "posterior belief", "prior belief", "belief state",
            "sensory uncertainty", "perceptual inference",
            "hidden state", "latent variable", "state-space model",
            "gaussian noise", "observation model", "generative process",
            "prediction error", "bayesian updating", "change point",
            "hazard rate", "adaptive learning rate",
            "hierarchical model", "empirical bayes",
            "variational bayes", "variational inference",
            "sampling", "monte carlo", "mcmc",
            "kalman gain", "particle filter",
        ],
    },

    "active_inference": {
        "family": "active-inference",
        "terms": [
            "active inference", "free energy", "expected free energy", "policy precision",
            "generative model",

            "variational free energy", "surprise minimization",
            "expected surprise", "epistemic value", "pragmatic value",
            "belief propagation",
            "perception action loop", "action selection",
            "policy selection", "precision weighting",
            "generative process", "hidden states", "observations",
            "likelihood mapping", "transition mapping",
            "variational inference", "free energy minimization",
            "gradient descent", "variational density",
            "predictive coding", "prediction error minimization",
        ],
    },

    "drift_diffusion_model": {
        "family": "sequential-sampling",
        "terms": [
            "drift diffusion", "ddm", "drift rate", "boundary separation", "non-decision time",
            "starting point bias", "decision threshold",
            "evidence accumulation", "accumulation to bound",
            "sequential sampling", "race model", "leaky competing accumulator",
            "urgency signal", "time-varying drift",
            "reaction time distribution", "choice probability",
            "speed accuracy tradeoff",
            "ramping activity", "integrator", "neural accumulator",
        ],
    },

    "prospect_theory": {
        "family": "decision-theory",
        "terms": [
            "prospect theory", "loss aversion", "risk preference", "utility curvature",
            "value function", "reference point", "probability weighting",
            "nonlinear probability weighting",
            "risk aversion", "risk seeking", "expected utility",
            "subjective value", "decision weight",
            "gambling task", "lottery choice", "risky decision making",
            "value encoding", "subjective utility", "ventromedial prefrontal cortex",
        ],
    },

    "predictive_coding": {
        "family": "predictive-processing",
        "terms": [
            "predictive coding", "prediction error", "hierarchical prediction",
            "top-down prediction", "bottom-up error", "error unit",
            "prediction error minimization", "hierarchical inference",
        ],
    },

    "efficient_coding": {
        "family": "information-theory",
        "terms": [
            "efficient coding", "redundancy reduction", "information maximization",
            "sparse coding", "population coding", "coding efficiency",
            "mutual information", "entropy", "information bottleneck",
        ],
    },

    "generalized_linear_model": {
        "family": "encoding-model",
        "terms": [
            "generalized linear model", "glm", "linear-nonlinear model",
            "ln model", "poisson glm", "spike train model",
            "stimulus filter", "temporal filter", "coupling filter",
            "log likelihood", "link function",
        ],
    },

    "neural_population_dynamics": {
        "family": "dynamical-systems",
        "terms": [
            "dynamical system", "neural dynamics", "state space",
            "population activity", "latent dynamics", "low dimensional manifold",
            "trajectory", "neural trajectory", "attractor",
            "fixed point", "recurrent dynamics",
        ],
    },

    "recurrent_neural_network": {
        "family": "neural-network",
        "terms": [
            "recurrent neural network", "rnn", "lstm", "gru",
            "hidden state", "sequence learning", "backpropagation through time",
            "trained network", "network model",
        ],
    },

    "hebbian_learning": {
        "family": "synaptic-plasticity",
        "terms": [
            "hebbian learning", "hebb rule", "synaptic plasticity",
            "long term potentiation", "ltp", "long term depression", "ltd",
            "spike timing dependent plasticity", "stdp",
            "synaptic weight change",
        ],
    },

    "attractor_network": {
        "family": "dynamical-systems",
        "terms": [
            "attractor network", "continuous attractor", "point attractor",
            "line attractor", "bump attractor", "stable state",
            "working memory attractor", "persistent activity",
        ],
    },

    "hierarchical_rl": {
        "family": "reinforcement-learning",
        "terms": [
            "hierarchical reinforcement learning", "options framework",
            "temporal abstraction", "subgoal", "option policy",
            "hierarchical policy", "skill learning",
        ],
    },

    "inverse_reinforcement_learning": {
        "family": "reinforcement-learning",
        "terms": [
            "inverse reinforcement learning", "irl",
            "reward inference", "latent reward", "preference inference",
            "behavioral cloning", "imitation learning",
        ],
    },

    "predictive_state_representation": {
        "family": "reinforcement-learning",
        "terms": [
            "predictive state representation", "psr",
            "successor representation", "successor features",
            "future occupancy", "state prediction",
        ],
    },

    "control_theory": {
        "family": "control-theory",
        "terms": [
            "optimal control", "control policy", "cost function",
            "linear quadratic regulator", "lqr",
            "feedback control", "feedforward control",
            "state estimation", "observer",
        ],
    },

    "kalman_control": {
        "family": "control-theory",
        "terms": [
            "kalman control", "lqg", "linear quadratic gaussian",
            "kalman filtering", "state estimator",
            "optimal feedback control",
        ],
    },

    "graphical_models": {
        "family": "probabilistic-graphical-model",
        "terms": [
            "graphical model", "bayesian network", "markov random field",
            "factor graph", "conditional independence",
            "belief propagation", "message passing",
        ],
    },

    "hidden_markov_model": {
        "family": "state-space",
        "terms": [
            "hidden markov model", "hmm", "state transition",
            "emission probability", "viterbi", "latent state",
            "state switching", "discrete states",
        ],
    },

    "switching_linear_dynamical_system": {
        "family": "state-space",
        "terms": [
            "switching linear dynamical system", "slds",
            "linear dynamical system", "lds",
            "state switching dynamics", "piecewise linear dynamics",
        ],
    },

    "energy_based_model": {
        "family": "statistical-physics",
        "terms": [
            "energy-based model", "boltzmann machine",
            "restricted boltzmann machine", "rbm",
            "ising model", "energy landscape",
            "partition function",
        ],
    },

    "deep_learning_model": {
        "family": "neural-network",
        "terms": [
            "deep neural network", "cnn", "convolutional neural network",
            "representation learning", "feature hierarchy",
            "supervised learning", "unsupervised learning",
            "autoencoder",
        ],
    },

    "normative_model": {
        "family": "theoretical",
        "terms": [
            "normative model", "optimality", "ideal observer",
            "rational model", "computational level",
            "optimal solution", "theoretical bound",
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
        # --- RELEVANT (computational / model-based language) ---

        # RL
        "reward prediction error signals were computed on each trial",
        "the agent updated action values using a temporal difference rule",
        "a q-learning model was fit to behavioral choices",
        "dopamine activity tracked prediction error magnitude",
        "subjects learned through reinforcement learning across trials",

        # Bayesian
        "beliefs were updated according to a bayesian inference model",
        "posterior distributions were computed from prior and likelihood",
        "the model inferred hidden states under uncertainty",
        "a hierarchical bayesian model captured subject variability",
        "kalman filtering was used to track latent beliefs",

        # Active inference / predictive coding
        "the agent minimized expected free energy during policy selection",
        "prediction errors were propagated across hierarchical levels",
        "a generative model explained sensory observations",
        "precision weighting modulated belief updates",
        "perception was framed as predictive coding",

        # DDM / decision models
        "choices were modeled using a drift diffusion process",
        "evidence accumulated until reaching a decision boundary",
        "drift rate varied with stimulus strength",
        "reaction times reflected accumulation dynamics",

        # GLM / encoding
        "spike trains were modeled using a poisson glm",
        "neural responses were predicted using a linear-nonlinear model",
        "stimulus filters captured temporal structure in firing rates",

        # Dynamical systems
        "neural population activity evolved in a low dimensional manifold",
        "latent dynamics were inferred from population recordings",
        "trajectories converged to a stable attractor state",
        "recurrent dynamics explained temporal patterns of activity",

        # HMM / latent state
        "a hidden markov model identified discrete brain states",
        "state transitions were inferred from neural data",
        "latent states governed behavioral switching",

        # Control theory
        "behavior was modeled using optimal control theory",
        "the system minimized a cost function over trajectories",
        "state estimation was performed using a kalman filter",

        # General
        "computational modeling explained behavioral variability",
        "model parameters were fit using maximum likelihood estimation",
        "model comparison was performed using bic",
        "latent variables captured internal cognitive states",

        # --- HARD POSITIVES (mixed sentences like real papers) ---
        "we recorded neural activity and fit a reinforcement learning model",
        "behavioral data were analyzed using a bayesian hierarchical model",
        "spike data were collected and modeled with a generalized linear model",
        "subjects performed a task while belief updating was modeled computationally",
        "neural recordings revealed dynamics consistent with an attractor network",

        # --- NEGATIVE (non-computational / methods / generic neuro) ---

        "animals were housed under standard conditions",
        "mice were kept on a 12 hour light dark cycle",
        "brain slices were prepared for electrophysiology",
        "cells were visualized using fluorescence microscopy",
        "immunohistochemistry was performed using standard protocols",
        "neurons were patched using whole cell recording",
        "data were collected using a neuropixels probe",
        "signals were amplified and filtered",
        "behavioral training lasted several days",
        "subjects received water rewards during training",
        "the apparatus consisted of a virtual maze",
        "lick responses were recorded during the task",
        "video tracking was used to monitor movement",
        "statistical significance was assessed using a t test",
        "error bars represent standard error of the mean",

        # --- HARD NEGATIVES (technical but NOT model-based) ---
        "firing rates increased during stimulus presentation",
        "neurons in prefrontal cortex showed sustained activity",
        "population responses were averaged across trials",
        "we computed correlations between neurons",
        "principal component analysis was applied to the data",
        "variance explained by the first component was high",
        "neural activity was aligned to stimulus onset",
        "responses differed significantly across conditions",
        "trial averages revealed consistent patterns",
    ]

    labels = [
        # relevant
        *([1] * 45),

        # not relevant
        *([0] * 25),
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
    Detect likely tested models from paper text.
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