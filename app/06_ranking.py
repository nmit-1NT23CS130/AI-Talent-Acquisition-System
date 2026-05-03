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
# 3. COMPUTE FINAL SCORE
# ──────────────────────────────────────────────────

def compute_final_score(confidence, bert_sim, cosine_sim):
    """
    Weighted final score combining:
    - 50% model confidence
    - 30% BERT semantic similarity
    - 20% TF-IDF cosine similarity
    """
    return (0.5 * confidence +
            0.3 * bert_sim +
            0.2 * cosine_sim)


# ──────────────────────────────────────────────────
# 4. ASSIGN TIER
# ──────────────────────────────────────────────────

def assign_tier(label):
    """
    Assign numeric tier for sorting:
    Good Fit = 1, Potential Fit = 2, No Fit = 3
    """
    tiers = {'Good Fit': 1, 'Potential Fit': 2, 'No Fit': 3}
    return tiers.get(label, 3)


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

    df_ranking['final_score'] = compute_final_score(
        df_ranking['confidence'],
        df_ranking['bert_similarity'],
        df_ranking['cosine_similarity']
    )

    df_ranking['tier'] = df_ranking['predicted_label'].apply(
        assign_tier)

    df_ranking = df_ranking.sort_values(
        ['tier', 'final_score'],
        ascending=[True, False]
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

        # Final score
        final_score = compute_final_score(
            confidence, bert_sim, cosine_sim)

        results.append({
            'candidate':       name,
            'predicted_label': label,
            'confidence':      round(confidence * 100, 1),
            'bert_similarity': round(bert_sim, 4),
            'cosine_similarity': round(cosine_sim, 4),
            'final_score':     round(final_score, 4),
            'tier':            assign_tier(label)
        })

    # Sort by tier then final score
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(
        ['tier', 'final_score'],
        ascending=[True, False]
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