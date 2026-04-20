# ==========================================
# SKLEARN MODEL (RELEVANCE DETECTION)
# ==========================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

# ------------------------------------------
# BUILD MODEL
# ------------------------------------------
def build_model():
    """
    Creates TF-IDF + SVM pipeline
    """

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
    return model.decision_function([text])[0]