"""
ranking.py
==========
Candidate ranking using trained model.
Used by Streamlit frontend and notebooks.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────
# 1. LOAD MODEL AND DATA
# ──────────────────────────────────────────────────

def load_ranking_components(models_dir="../models",
                             data_dir="../data"):
    """Load model, label encoder and test features."""
    best_model = joblib.load(f"{models_dir}/best_model.pkl")
    le         = joblib.load(f"{models_dir}/label_encoder.pkl")
    tfidf      = joblib.load(f"{models_dir}/tfidf_vectorizer.pkl")
    X_test     = sp.load_npz(f"{data_dir}/X_test.npz")
    df_test    = pd.read_csv(
        f"{data_dir}/resume_jd_test_cleaned.csv")

    print("✅ Ranking components loaded!")
    return best_model, le, tfidf, X_test, df_test


# ──────────────────────────────────────────────────
# 2. PREDICT LABELS AND SCORES
# ──────────────────────────────────────────────────

def predict_candidates(model, le, X):
    """
    Predict labels and confidence scores for candidates.
    Returns predicted labels and confidence scores.
    """
    y_pred = model.predict(X)
    proba  = model.predict_proba(X)

    labels     = le.inverse_transform(y_pred)
    confidence = proba.max(axis=1)

    return labels, confidence


# ──────────────────────────────────────────────────
# 3. COMPUTE FINAL SCORE  (FIXED)
# ──────────────────────────────────────────────────

def compute_final_score(confidence, bert_sim, cosine_sim, label):
    """
    Weighted final score combining:
    - 55% BERT semantic similarity  (most important — captures true relevance)
    - 30% TF-IDF cosine similarity  (keyword overlap)
    - 15% model confidence          (only rewarded for Good/Potential Fit)

    Key fix: confidence for a 'No Fit' prediction is NOT rewarded.
    Previously, high confidence in 'No Fit' was inflating scores
    for irrelevant candidates (e.g. DevOps ranked above Data Scientist).
    """
    label_bonus = {
        'Good Fit':      1.0,
        'Potential Fit': 0.6,
        'No Fit':        0.0   # don't reward confidence in No Fit
    }
    adjusted_confidence = confidence * label_bonus.get(label, 0.0)

    return (0.55 * bert_sim +
            0.30 * cosine_sim +
            0.15 * adjusted_confidence)


# ──────────────────────────────────────────────────
# 3b. FIX PREDICTED LABEL (NEW)
# ──────────────────────────────────────────────────

def fix_predicted_label(label, bert_sim, cosine_sim):
    """
    Override XGBoost label when similarity scores
    clearly contradict the prediction.

    Thresholds (tunable):
      bert_sim >= 0.65  AND  cosine_sim >= 0.35  → Good Fit
      bert_sim >= 0.55  AND  cosine_sim >= 0.10  → Potential Fit
      Anything else                               → No Fit

    This also DOWNGRADES incorrect XGBoost Good Fit / Potential Fit
    predictions — e.g. a Junior Dev labelled Good Fit for a Senior
    Data Scientist JD will be corrected because cosine_sim will be low.

    Why: XGBoost was trained on limited data and mis-classifies
    candidates when the JD is outside its training distribution.
    BERT + TF-IDF cosine similarity are more reliable signals
    for unseen JDs, so we use them as the single source of truth.
    """
    if bert_sim >= 0.65 and cosine_sim >= 0.35:
        return 'Good Fit'
    if bert_sim >= 0.55 and cosine_sim >= 0.10:
        return 'Potential Fit'
    return 'No Fit'   # ← always override, not just upgrade


# ──────────────────────────────────────────────────
# 4. ASSIGN TIER  (FIXED)
# ──────────────────────────────────────────────────

def assign_tier(label, final_score=None):
    """
    Tier is now a display hint only — NOT used as primary sort key.

    Fix: If model says 'No Fit' but similarity score is high (>=0.45),
    candidate is bumped to tier 2 (Potential Fit zone).
    This handles cases where XGBoost mis-classifies a relevant candidate.
    """
    if label == 'Good Fit':
        return 1
    if label == 'Potential Fit':
        return 2
    # No Fit — override if similarity is actually high
    if final_score is not None and final_score >= 0.45:
        return 2
    return 3


# ──────────────────────────────────────────────────
# 5. RANK CANDIDATES FROM DATAFRAME
# ──────────────────────────────────────────────────

def rank_candidates_from_df(model, le, X_test, df_test):
    """
    Rank all candidates in test set.
    Returns sorted DataFrame with rank column.
    """
    labels, confidence = predict_candidates(model, le, X_test)

    df_ranking = df_test.copy()
    df_ranking['predicted_label'] = labels
    df_ranking['confidence']      = confidence

    df_ranking['final_score'] = df_ranking.apply(
        lambda row: compute_final_score(
            row['confidence'],
            row['bert_similarity'],
            row['cosine_similarity'],
            row['predicted_label']
        ), axis=1
    )

    df_ranking['tier'] = df_ranking.apply(
        lambda row: assign_tier(
            row['predicted_label'],
            row['final_score']
        ), axis=1
    )

    # Sort purely by final_score — tier is display only
    df_ranking = df_ranking.sort_values(
        'final_score',
        ascending=False
    ).reset_index(drop=True)

    df_ranking['rank'] = df_ranking.index + 1

    return df_ranking


# ──────────────────────────────────────────────────
# 6. RANK LIVE UPLOADED RESUMES (for Streamlit)
# ──────────────────────────────────────────────────

def rank_live_candidates(resumes, jd_text,
                          model, le, tfidf, bert_model,
                          preprocess_fn):
    """
    Rank a list of live uploaded resumes against a JD.

    Parameters:
        resumes      : list of dicts with 'name' and 'text'
        jd_text      : raw JD string
        model        : trained best model
        le           : label encoder
        tfidf        : fitted TF-IDF vectorizer
        bert_model   : loaded Sentence-BERT model
        preprocess_fn: function to clean raw text

    Returns: sorted DataFrame with ranking results
    """
    jd_clean = preprocess_fn(jd_text)
    results  = []

    for resume in resumes:
        name         = resume['name']
        resume_clean = preprocess_fn(resume['text'])

        # TF-IDF features
        r_vec      = tfidf.transform([resume_clean])
        j_vec      = tfidf.transform([jd_clean])
        cosine_sim = float(cosine_similarity(r_vec, j_vec)[0][0])
        sim_sparse = sp.csr_matrix([[cosine_sim]])
        X_tfidf    = sp.hstack([r_vec, j_vec, sim_sparse])

        # BERT features
        r_emb     = bert_model.encode([resume_clean])
        j_emb     = bert_model.encode([jd_clean])
        bert_sim  = float(cosine_similarity(r_emb, j_emb)[0][0])

        # Combine TF-IDF + BERT
        bert_vec = sp.csr_matrix(
            np.hstack([r_emb, j_emb,
                       np.array([[bert_sim]])])
        )
        X_combined = sp.hstack([X_tfidf, bert_vec])

        # Predict
        y_pred     = model.predict(X_combined)
        proba      = model.predict_proba(X_combined)
        label      = le.inverse_transform(y_pred)[0]
        confidence = float(proba.max())

        # Override label if similarity scores contradict XGBoost
        label = fix_predicted_label(label, bert_sim, cosine_sim)

        # Final score — label passed so No Fit confidence is not rewarded
        final_score = compute_final_score(
            confidence, bert_sim, cosine_sim, label)

        results.append({
            'candidate':         name,
            'predicted_label':   label,
            'confidence':        round(confidence * 100, 1),
            'bert_similarity':   round(bert_sim, 4),
            'cosine_similarity': round(cosine_sim, 4),
            'final_score':       round(final_score, 4),
            'tier':              assign_tier(label, final_score)
        })

    # Sort purely by final_score — tier is display hint only
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(
        'final_score',
        ascending=False
    ).reset_index(drop=True)

    df_results['rank'] = df_results.index + 1
    return df_results


# ──────────────────────────────────────────────────
# 7. DISPLAY HELPERS
# ──────────────────────────────────────────────────

def get_top_candidates(df_ranking, n=10):
    """Return top N candidates."""
    cols = ['rank', 'predicted_label', 'confidence',
            'cosine_similarity', 'bert_similarity', 'final_score']
    available = [c for c in cols if c in df_ranking.columns]
    return df_ranking[available].head(n)


def get_label_emoji(label):
    """Return emoji for each label."""
    emojis = {
        'Good Fit':      '🟢',
        'Potential Fit': '🟡',
        'No Fit':        '🔴'
    }
    return emojis.get(label, '⚪')


# ──────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    best_model, le, tfidf, X_test, df_test = \
        load_ranking_components()

    df_ranking = rank_candidates_from_df(
        best_model, le, X_test, df_test)

    print("\n=== TOP 10 CANDIDATES ===")
    print(get_top_candidates(df_ranking))

    print("\n=== BOTTOM 10 CANDIDATES ===")
    print(get_top_candidates(
        df_ranking.sort_values('rank', ascending=False)))