
"""
06_ranking_enhanced.py
======================

Candidate ranking using the combined trained model.
Used by the Streamlit frontend.

Combined model files:
- best_model_combined.pkl
- label_encoder_combined.pkl
- tfidf_vectorizer_combined.pkl
- bert_model_combined.pkl
"""

import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import importlib.util
from sklearn.metrics.pairwise import cosine_similarity


def load_optional_module(filename):
    module_path = os.path.join(
        os.path.dirname(__file__),
        filename
    )

    spec = importlib.util.spec_from_file_location(
        filename.replace(".py", ""),
        module_path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


_explainability_module = load_optional_module("08_explainability.py")
_skill_gap_module = load_optional_module("07_skill_gap.py")

build_explainer = _explainability_module.build_explainer
explain_candidate = _explainability_module.explain_candidate
analyze_skill_gap = _skill_gap_module.analyze_skill_gap


# ============================================================
# 1. LOAD COMBINED MODEL COMPONENTS
# ============================================================

def load_ranking_components():
    """
    Load the combined XGBoost model,
    label encoder, TF-IDF vectorizer
    and Sentence-BERT model.
    """

    # Project root = parent of app folder
    base_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    models_dir = os.path.join(
        base_dir,
        "models"
    )

    # Enhanced XGBoost model
    best_model = joblib.load(
        os.path.join(
            models_dir,
            "best_model_combined.pkl"
        )
    )

    # Enhanced label encoder
    le = joblib.load(
        os.path.join(
            models_dir,
            "label_encoder_combined.pkl"
        )
    )

    # Enhanced TF-IDF vectorizer
    tfidf = joblib.load(
        os.path.join(
            models_dir,
            "tfidf_vectorizer_combined.pkl"
        )
    )

    # Enhanced Sentence-BERT model
    bert_model = joblib.load(
        os.path.join(
            models_dir,
            "bert_model_combined.pkl"
        )
    )

    print("Combined ranking components loaded!")

    return best_model, le, tfidf, bert_model


# ============================================================
# 2. PREPROCESS TEXT
# ============================================================

def preprocess_text(text):
    """
    Preprocessing used for the combined dataset.

    - lowercase
    - email removal
    - URL removal
    - phone number removal
    - non-letter removal
    - extra-space removal
    """

    import re

    text = str(text).lower()

    # Remove emails
    text = re.sub(
        r'\S+@\S+',
        ' ',
        text
    )

    # Remove URLs
    text = re.sub(
        r'http\S+|www\S+',
        ' ',
        text
    )

    # Remove phone numbers
    text = re.sub(
        r'\+?\d[\d\s\-().]{7,}\d',
        ' ',
        text
    )

    # Keep letters and spaces
    text = re.sub(
        r'[^a-zA-Z\s]',
        ' ',
        text
    )

    # Remove extra spaces
    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


# ============================================================
# 3. PREDICT LABELS AND CONFIDENCE
# ============================================================

def predict_candidates(model, le, X):
    """
    Predict labels and model confidence.
    """

    y_pred = model.predict(X)

    proba = model.predict_proba(X)

    labels = le.inverse_transform(y_pred)

    confidence = proba.max(
        axis=1
    )

    return labels, confidence


# ============================================================
# 4. COMPUTE FINAL SCORE
# ============================================================

def compute_final_score(
    confidence,
    bert_sim,
    cosine_sim,
    label
):
    """
    Final ranking score:

    55% BERT semantic similarity
    30% TF-IDF cosine similarity
    15% model confidence

    Confidence is only rewarded for:
    Good Fit / Potential Fit.

    No Fit confidence receives no bonus.
    """

    label_bonus = {
        "Good Fit": 1.0,
        "Potential Fit": 0.6,
        "No Fit": 0.0
    }

    adjusted_confidence = (
        confidence *
        label_bonus.get(
            label,
            0.0
        )
    )

    return (
        0.55 * bert_sim +
        0.30 * cosine_sim +
        0.15 * adjusted_confidence
    )


# ============================================================
# 5. FIX PREDICTED LABEL
# ============================================================

def fix_predicted_label(
    label,
    bert_sim,
    cosine_sim
):
    """
    Adjust the model prediction using similarity signals.

    Good Fit:
        BERT >= 0.65 AND TF-IDF >= 0.35

    Potential Fit:
        BERT >= 0.55 AND TF-IDF >= 0.10

    Otherwise:
        No Fit
    """

    if (
        bert_sim >= 0.65
        and cosine_sim >= 0.35
    ):
        return "Good Fit"

    if (
        bert_sim >= 0.55
        and cosine_sim >= 0.10
    ):
        return "Potential Fit"

    return "No Fit"


# ============================================================
# 6. ASSIGN TIER
# ============================================================

def assign_tier(
    label,
    final_score=None
):
    """
    Tier is used only as a display hint.

    1 = Good Fit
    2 = Potential Fit
    3 = No Fit
    """

    if label == "Good Fit":
        return 1

    if label == "Potential Fit":
        return 2

    if (
        final_score is not None
        and final_score >= 0.45
    ):
        return 2

    return 3


# ============================================================
# 7. RANK LIVE UPLOADED RESUMES
# ============================================================

def rank_live_candidates(
    resumes,
    jd_text,
    model,
    le,
    tfidf,
    bert_model,
    preprocess_fn,
    explainer=None,
    explain_fn=None,
    skill_gap_fn=None
):
    """
    Rank uploaded resumes against a job description.

    The combined model expects EXACTLY 10,769 features:

    10,000 TF-IDF
    + 384 Resume BERT
    + 384 JD BERT
    + 1 BERT cosine similarity
    --------------------------------
    = 10,769 features

    TF-IDF cosine similarity is calculated separately
    for the final ranking score, but is NOT passed
    into the XGBoost model.
    """

    # --------------------------------------------------------
    # Clean JD
    # --------------------------------------------------------

    jd_clean = preprocess_fn(
        jd_text
    )

    results = []

    # ========================================================
    # PROCESS EACH RESUME
    # ========================================================

    for resume in resumes:

        name = resume["name"]

        resume_clean = preprocess_fn(
            resume["text"]
        )

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        resume_vec = tfidf.transform(
            [resume_clean]
        )

        jd_vec = tfidf.transform(
            [jd_clean]
        )

        # TF-IDF cosine similarity
        # Used for ranking score only
        cosine_sim = float(
            cosine_similarity(
                resume_vec,
                jd_vec
            )[0][0]
        )

        # IMPORTANT:
        # DO NOT add cosine_sim to X_tfidf.
        #
        # The combined XGBoost model was trained with:
        # 5000 resume TF-IDF + 5000 JD TF-IDF
        # = 10000 TF-IDF features
        #
        # Adding cosine here would make 10001 TF-IDF
        # features and cause a shape mismatch.

        X_tfidf = sp.hstack(
            [
                resume_vec,
                jd_vec
            ]
        )

        # ----------------------------------------------------
        # BERT / Sentence-BERT
        # ----------------------------------------------------

        resume_emb = bert_model.encode(
            [resume_clean]
        )

        jd_emb = bert_model.encode(
            [jd_clean]
        )

        # BERT cosine similarity
        bert_sim = float(
            cosine_similarity(
                resume_emb,
                jd_emb
            )[0][0]
        )

        # ----------------------------------------------------
        # Combine BERT features
        # ----------------------------------------------------

        bert_features = sp.csr_matrix(
            np.hstack(
                [
                    resume_emb,
                    jd_emb,
                    np.array(
                        [[bert_sim]]
                    )
                ]
            )
        )

        # ----------------------------------------------------
        # FINAL MODEL INPUT
        # ----------------------------------------------------

        # 10,000 TF-IDF
        # + 384 Resume BERT
        # + 384 JD BERT
        # + 1 BERT similarity
        # = 10,769 features

        X_combined = sp.hstack(
            [
                X_tfidf,
                bert_features
            ]
        ).tocsr()

        # ----------------------------------------------------
        # XGBOOST PREDICTION
        # ----------------------------------------------------

        y_pred = model.predict(
            X_combined
        )

        probabilities = model.predict_proba(
            X_combined
        )

        raw_label = le.inverse_transform(
            y_pred
        )[0]

        confidence = float(
            probabilities.max()
        )

        label = fix_predicted_label(
            raw_label,
            bert_sim,
            cosine_sim
        )

        # ----------------------------------------------------
        # FINAL RANKING SCORE
        # ----------------------------------------------------

        final_score = compute_final_score(
            confidence,
            bert_sim,
            cosine_sim,
            label
        )

        # ----------------------------------------------------
        # EXPLAINABILITY
        # ----------------------------------------------------

        resume_raw = resume["text"]

        if explain_fn is not None:
            try:
                explanation = explain_fn(
                    explainer,
                    model,
                    le,
                    tfidf,
                    resume_vec,
                    jd_vec,
                    X_combined.toarray()[0],
                    raw_label,
                    label,
                    confidence,
                    bert_sim,
                    cosine_sim,
                    resume_text_raw=resume_raw,
                    jd_text_raw=jd_text
                )

            except Exception:
                explanation = {
                    "score_explanation": {},
                    "keyword_explanation": [],
                    "semantic_note": {},
                    "model_diagnostic": {},
                    "skill_gap": None
                }

        else:
            explanation = None

        # ----------------------------------------------------
        # SKILL GAP
        # ----------------------------------------------------

        if skill_gap_fn is not None:
            try:
                skill_gap = skill_gap_fn(
                    jd_text,
                    resume_raw,
                    bert_model
                )

            except Exception:
                skill_gap = None

        else:
            skill_gap = None

        # Add skill gap to explanation
        if explanation is not None:
            explanation["skill_gap"] = skill_gap

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        result_row = {
            "candidate": name,

            "predicted_label": label,

            "confidence":
                round(
                    confidence * 100,
                    1
                ),

            "bert_similarity":
                round(
                    bert_sim,
                    4
                ),

            "cosine_similarity":
                round(
                    cosine_sim,
                    4
                ),

            "final_score":
                round(
                    final_score,
                    4
                ),

            "tier":
                assign_tier(
                    label,
                    final_score
                ),

            "explanation": explanation,

            "skill_gap": skill_gap
        }

        results.append(
            result_row
        )

    # ========================================================
    # SORT RESULTS
    # ========================================================

    df_results = pd.DataFrame(
        results
    )

    if not df_results.empty:

        df_results = (
            df_results
            .sort_values(
                "final_score",
                ascending=False
            )
            .reset_index(drop=True)
        )

        df_results["rank"] = (
            df_results.index + 1
        )

    return df_results


# ============================================================
# 8. DISPLAY HELPERS
# ============================================================

def get_top_candidates(
    df_ranking,
    n=10
):
    """
    Return top N candidates.
    """

    cols = [
        "rank",
        "candidate",
        "predicted_label",
        "confidence",
        "cosine_similarity",
        "bert_similarity",
        "final_score"
    ]

    available = [
        c
        for c in cols
        if c in df_ranking.columns
    ]

    return (
        df_ranking[
            available
        ]
        .head(n)
    )


def get_label_emoji(label):
    """
    Return emoji for each label.
    """

    emojis = {
        "Good Fit": "🟢",
        "Potential Fit": "🟡",
        "No Fit": "🔴"
    }

    return emojis.get(
        label,
        "⚪"
    )


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":

    model, le, tfidf, bert_model = (
        load_ranking_components()
    )

    print(
        "\nEnhanced ranking components "
        "loaded successfully."
    )

