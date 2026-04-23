# ==========================================
# TEXT SEGMENTATION
# ==========================================

import re

try:
    from ml_model import (
        identify_candidate_models,
        build_model_primer,
        get_relevance_model,
        get_relevance_score,
    )
except Exception:
    def identify_candidate_models(text, top_k=4):
        return []

    def build_model_primer(text, top_k=3):
        return ""

    def get_relevance_model():
        return None

    def get_relevance_score(text, model=None):
        return 0.5


SECTION_HEADERS = {
    "abstract",
    "introduction",
    "background",
    "methods",
    "materials and methods",
    "experimental methods",
    "experimental procedure",
    "results",
    "results and discussion",
    "discussion",
    "general discussion",
    "conclusion",
    "conclusions",
    "study 1",
    "study 2",
    "study 3",
    "experiment 1",
    "experiment 2",
    "experiment 3",
    "participants",
    "procedure",
    "behavioral results",
    "neuroimaging results",
    "computational modeling",
    "model fitting",
    "model comparison",
}

LOW_SIGNAL_HEADERS = {
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "supplementary",
    "supplementary material",
    "supplementary methods",
    "supplementary figures",
    "author contributions",
    "conflict of interest",
    "funding",
    "data availability",
    "ethics statement",
}

# Keyword synonym clusters — covers diverse neuroscience terminology
KEYWORD_WEIGHTS = {
    # --- Experimental design ---
    "experiment": 2.0,
    "study": 1.5,
    "manipulated": 2.0,
    "manipulation": 1.8,
    "randomized": 1.5,
    "intervention": 2.0,
    "stimulus": 1.5,
    "stimuli": 1.5,
    "measured": 2.0,
    "condition": 1.3,
    "trial": 1.3,
    "task": 1.2,
    "participant": 1.2,
    "subject": 1.0,
    "group": 0.8,

    # --- Behavioral outcomes ---
    "behavior": 1.5,
    "behaviour": 1.5,
    "reaction time": 1.2,
    "response time": 1.2,
    "accuracy": 1.2,
    "performance": 1.2,
    "choice": 1.3,
    "decision": 1.3,
    "error rate": 1.4,

    # --- Learning & memory ---
    "learning": 1.8,
    "memory": 1.8,
    "recall": 1.5,
    "encoding": 1.5,
    "consolidation": 1.5,
    "extinction": 1.6,
    "generalization": 1.5,

    # --- Prediction error / Bayesian cluster ---
    "prediction error": 2.2,
    "reward prediction error": 2.4,
    "surprise": 1.8,          # prediction error synonym
    "unexpected": 1.5,
    "mismatch": 1.5,
    "uncertainty": 1.8,        # belief/entropy synonym
    "volatility": 1.8,
    "entropy": 1.6,
    "precision": 1.6,
    "prior": 1.5,
    "posterior": 1.5,
    "likelihood": 1.4,
    "belief": 1.8,
    "inference": 1.6,
    "expectation": 1.5,

    # --- Value & RL cluster ---
    "policy": 1.6,
    "value": 1.4,
    "reward": 1.8,
    "punishment": 1.5,
    "feedback": 1.5,
    "reinforcement": 1.8,
    "habit": 1.6,
    "goal-directed": 1.8,
    "model-based": 1.8,
    "model-free": 1.8,
    "temporal difference": 2.0,
    "rescorla-wagner": 2.0,
    "td learning": 2.0,
    "q-learning": 1.8,
    "actor-critic": 1.8,

    # --- Neuromodulators ---
    "dopamine": 2.0,
    "serotonin": 2.0,
    "norepinephrine": 1.8,
    "noradrenaline": 1.8,
    "acetylcholine": 1.8,
    "gaba": 1.5,
    "glutamate": 1.5,
    "oxytocin": 1.6,

    # --- Computational models ---
    "computational model": 2.3,
    "reinforcement learning": 2.3,
    "bayesian": 2.1,
    "active inference": 2.2,
    "free energy": 2.0,
    "kalman filter": 1.8,
    "drift diffusion": 1.8,
    "parameter": 1.4,
    "model fit": 1.8,
    "model comparison": 1.8,
    "aic": 1.5,
    "bic": 1.5,
    "log likelihood": 1.7,

    # --- Brain regions ---
    "striatum": 1.8,
    "caudate": 1.7,
    "putamen": 1.7,
    "nucleus accumbens": 1.9,
    "prefrontal": 1.8,
    "prefrontal cortex": 2.0,
    "orbitofrontal": 1.8,
    "hippocampus": 1.8,
    "amygdala": 1.7,
    "insula": 1.6,
    "basal ganglia": 1.8,
    "anterior cingulate": 1.8,
    "vmPFC": 1.8,
    "dlPFC": 1.8,
    "cortex": 1.4,

    # --- Neuroimaging ---
    "eeg": 1.8,
    "fmri": 1.8,
    "bold": 1.5,
    "meg": 1.7,
    "erp": 1.6,
    "neural": 1.4,
    "activation": 1.3,
    "connectivity": 1.5,
    "resting state": 1.5,
    "functional connectivity": 1.7,
}


# Matches leading numbering prefixes in section headers, e.g.:
#   "2. Methods"  "2.1 Results"  "Section 2:"  "Part IV."  "ii. "
_NUMBERED_PREFIX_RE = re.compile(
    r"^"
    r"(?:"
    r"(?:section|part|chapter)\s+[\divxlcdmIVXLCDM]+[:\.)\s]+"  # Section 2: / Part IV.
    r"|[ivxlcdmIVXLCDM]{1,6}[\.)\s]+"                            # Roman: ii. / IV)
    r"|\d+(?:\.\d+)*[\.)\s]+"                                    # 2. / 2.1 / 2.1.
    r")",
    re.IGNORECASE,
)


def _normalize_text(text):
    normalized = str(text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _looks_like_header(line):
    stripped = line.strip()
    if not stripped:
        return False

    # --- All-caps lines (common in PDF extraction) ---
    # e.g. "METHODS", "RESULTS AND DISCUSSION"
    if stripped.isupper() and 2 <= len(stripped.split()) <= 6 and len(stripped) >= 3:
        candidate = stripped.lower()
        if candidate in SECTION_HEADERS or candidate in LOW_SIGNAL_HEADERS:
            return True
        if re.fullmatch(r"[A-Z][A-Z\s\-/&]+", stripped):
            return True

    # --- Strip leading numeric/roman prefix, then check remainder ---
    # e.g. "2. Methods" → "Methods", "Section 2: Results" → "Results"
    remainder = _NUMBERED_PREFIX_RE.sub("", stripped).strip().strip(":").lower()
    if remainder and remainder != stripped.strip(":").lower():
        if remainder in SECTION_HEADERS or remainder in LOW_SIGNAL_HEADERS:
            return True
        # Short alphabetic phrase after a number prefix is almost certainly a header
        if 1 <= len(remainder.split()) <= 6 and re.fullmatch(r"[a-z][a-z\s\-/&]+", remainder):
            return True

    # --- Plain lowercase match ---
    candidate = stripped.strip(":").lower()
    if candidate in SECTION_HEADERS or candidate in LOW_SIGNAL_HEADERS:
        return True

    if re.fullmatch(r"(experiment|study)\s+\d+", candidate):
        return True

    return False


def _split_sections(text):
    lines = text.split("\n")
    sections = []
    current_header = ""
    current_lines = []

    for raw_line in lines:
        line = raw_line.strip()
        if _looks_like_header(line):
            if current_lines:
                sections.append((current_header, "\n".join(current_lines).strip()))
            current_header = line.strip().strip(":").lower()
            current_lines = []
            continue

        current_lines.append(raw_line)

    if current_lines:
        sections.append((current_header, "\n".join(current_lines).strip()))

    return [(header, body) for header, body in sections if body]


def _chunk_with_overlap(text, max_chars=2600, overlap_chars=220):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(0, end - overlap_chars)

    return chunks


def _score_segment(text, header=""):
    lower = text.lower()
    score = 0.0

    # Count-based scoring: reward repeated keywords but with diminishing returns.
    # 1 hit = 1.0×weight, 2 hits = 1.3×weight, 3+ hits = 1.6×weight.
    for keyword, weight in KEYWORD_WEIGHTS.items():
        count = lower.count(keyword)
        if count > 0:
            multiplier = 1.0 + 0.3 * min(count - 1, 2)
            score += weight * multiplier

    # Bonus for explicit causal/mechanistic language
    causal_phrases = [
        "we manipulated", "we measured", "we found", "we observed",
        "significant effect", "was associated with", "predicted",
        "model predicts", "results show", "results indicate",
        "regression", "correlation", "main effect", "interaction",
    ]
    for phrase in causal_phrases:
        if phrase in lower:
            score += 1.0

    # Section-type bonus
    high_value_headers = {
        "results", "behavioral results", "neuroimaging results",
        "discussion", "results and discussion", "general discussion",
        "experiment 1", "experiment 2", "experiment 3",
        "study 1", "study 2", "study 3",
        "computational modeling", "model fitting", "model comparison",
    }
    if (header or "").lower() in high_value_headers:
        score += 2.0

    length_bonus = min(len(text) / 2200.0, 1.0)
    return score + length_bonus


def _is_low_signal_section(header, body):
    header = (header or "").lower()
    if header in LOW_SIGNAL_HEADERS:
        return True

    lower = body.lower()
    if "doi" in lower and lower.count(";") > 5 and "et al" in lower:
        return True

    return False


def _extract_focus_model_terms(text, max_terms=10):
    candidates = identify_candidate_models(text, top_k=4)
    ranked_terms = []
    for candidate in candidates:
        for term in candidate.get("evidence_terms", []):
            if term and term not in ranked_terms:
                ranked_terms.append(term)
    return ranked_terms[:max_terms]


def _resolve_max_segments(text_length, requested_max_segments=None):
    if requested_max_segments is not None:
        try:
            value = int(requested_max_segments)
        except (TypeError, ValueError):
            value = 5
        return max(1, min(12, value))

    if text_length < 6000:
        return 3
    if text_length < 15000:
        return 5
    if text_length < 30000:
        return 6
    return 7


def _token_signature(text, max_tokens=120):
    tokens = re.findall(r"[a-z0-9_\-]+", text.lower())[:max_tokens]
    return set(token for token in tokens if len(token) > 2)


def _is_near_duplicate(chunk_a, chunk_b, threshold=0.88):
    sig_a = _token_signature(chunk_a)
    sig_b = _token_signature(chunk_b)
    if not sig_a or not sig_b:
        return False

    overlap = len(sig_a & sig_b)
    min_len = min(len(sig_a), len(sig_b))
    if min_len == 0:
        return False

    return (overlap / min_len) >= threshold


def segment_text(text, max_segments=None, ml_score_threshold=0.38, llm_top_k=None):
    """
    Splits paper text into neuroscience-relevant chunks for LLM extraction.
    Uses section detection, overlap chunking, relevance scoring, and pruning.
    """

    normalized = _normalize_text(text)
    if not normalized:
        return []

    segment_cap = _resolve_max_segments(len(normalized), requested_max_segments=max_segments)

    sections = _split_sections(normalized)
    if not sections:
        sections = [("", normalized)]

    focus_model_terms = _extract_focus_model_terms(normalized)
    relevance_model = get_relevance_model()

    ranked = []
    position = 0
    for header, body in sections:
        if _is_low_signal_section(header, body):
            continue

        for chunk in _chunk_with_overlap(body):
            rule_score = _score_segment(chunk, header)
            lower_chunk = chunk.lower()
            if focus_model_terms:
                term_hits = sum(1 for term in focus_model_terms if term in lower_chunk)
                rule_score += min(3.0, term_hits * 0.6)

            ml_score = get_relevance_score(chunk, model=relevance_model)
            combined_score = (0.7 * rule_score) + (0.3 * (ml_score * 10.0))

            ranked.append((position, combined_score, ml_score, chunk))
            position += 1

    if not ranked:
        return [normalized]

    ranked.sort(key=lambda item: item[1], reverse=True)
    candidate_pool_size = min(len(ranked), max(segment_cap * 3, segment_cap))
    selected = ranked[:candidate_pool_size]
    selected.sort(key=lambda item: item[0])

    deduped_entries = []
    for position, combined_score, ml_score, chunk in selected:
        if not chunk.strip():
            continue
        if any(_is_near_duplicate(chunk, existing[3]) for existing in deduped_entries):
            continue
        deduped_entries.append((position, combined_score, ml_score, chunk))
        if len(deduped_entries) >= segment_cap:
            break

    gate_threshold = max(0.0, min(1.0, float(ml_score_threshold)))
    threshold_pass = [entry for entry in deduped_entries if entry[2] >= gate_threshold]
    gated_pool = threshold_pass or deduped_entries

    top_k = llm_top_k if llm_top_k is not None else segment_cap
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = segment_cap
    top_k = max(1, min(12, top_k))

    top_ranked = sorted(gated_pool, key=lambda item: item[1], reverse=True)[:top_k]
    top_ranked.sort(key=lambda item: item[0])

    deduped_segments = [entry[3] for entry in top_ranked if entry[3].strip()]

    # Keep at least one chunk even when scoring is weak
    return deduped_segments or [normalized]


def extract_paper_context(text):
    """
    Extracts a compact anchor string (title + abstract) from the paper.
    Prepended to every LLM prompt so each chunk has global context about
    which paper it belongs to and what experiments are described overall.
    Returns a string of at most ~700 chars, or empty string if nothing found.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return ""

    lines = normalized.split("\n")

    # Title heuristic: first non-empty line that is not a known header,
    # not a URL/DOI, and has a plausible title length (10–200 chars).
    title = ""
    for line in lines[:15]:
        stripped = line.strip()
        if (
            stripped
            and 10 <= len(stripped) <= 200
            and not _looks_like_header(stripped)
            and not stripped.lower().startswith(("doi", "http", "journal", "vol.", "pp."))
        ):
            title = stripped
            break

    # Abstract body: prefer an explicit "Abstract" section.
    abstract_text = ""
    sections = _split_sections(normalized)
    for header, body in sections:
        if "abstract" in (header or "").lower():
            abstract_text = body.strip()[:600]
            break

    # Fallback: if no abstract section detected, use the opening 600 chars.
    if not abstract_text:
        abstract_text = normalized[:600].strip()

    parts = []
    if title:
        parts.append(f"Paper title: {title}")
    if abstract_text:
        parts.append(f"Abstract: {abstract_text}")

    model_primer = build_model_primer(normalized, top_k=3)
    if model_primer:
        parts.append(model_primer)

    return "\n".join(parts)