# ==========================================
# TEXT SEGMENTATION
# ==========================================

def segment_text(text):
    """
    Splits text into sections using simple heuristics.
    Helps separate experiments/models.
    """

    sections = []
    current = []

    for line in text.split("\n"):

        if any(k in line.lower() for k in ["experiment", "study"]):
            if current:
                sections.append(" ".join(current))
                current = []

        current.append(line)

    if current:
        sections.append(" ".join(current))

    return sections