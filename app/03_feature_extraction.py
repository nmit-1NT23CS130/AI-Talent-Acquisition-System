"""
feature_extraction.py
=====================
TF-IDF vectorization and cosine similarity features.
Used by Streamlit frontend and notebooks.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────────
# 1. BUILD TF-IDF VECTORIZER
# ──────────────────────────────────────────────────

def build_tfidf(df_train):
    """
    Fit TF-IDF on train data (resume + JD combined).
    Returns fitted vectorizer.
    """
    print("Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    all_train_text = pd.concat([
        df_train['resume_clean'],
        df_train['jd_clean']
    ])
    tfidf.fit(all_train_text)
    print("✅ TF-IDF fitted!")
    return tfidf


# ──────────────────────────────────────────────────
# 2. TRANSFORM TEXT TO TF-IDF VECTORS
# ──────────────────────────────────────────────────

def transform_tfidf(tfidf, df_train, df_test):
    """
    Transform train and test resumes + JDs into TF-IDF vectors.
    Returns 4 sparse matrices.
    """
    train_resume = tfidf.transform(df_train['resume_clean'])
    train_jd     = tfidf.transform(df_train['jd_clean'])
    test_resume  = tfidf.transform(df_test['resume_clean'])
    test_jd      = tfidf.transform(df_test['jd_clean'])

    print("✅ TF-IDF transform done!")
    print(f"   Train resume shape: {train_resume.shape}")
    return train_resume, train_jd, test_resume, test_jd


# ──────────────────────────────────────────────────
# 3. COMPUTE COSINE SIMILARITY
# ──────────────────────────────────────────────────

def compute_cosine_similarity(resume_vecs, jd_vecs):
    """
    Compute pairwise cosine similarity for each resume-JD pair.
    Returns list of float scores.
    """
    scores = [
        cosine_similarity(resume_vecs[i], jd_vecs[i])[0][0]
        for i in range(resume_vecs.shape[0])
    ]
    return scores


def compute_single_similarity(resume_text, jd_text, tfidf):
    """
    Compute cosine similarity for ONE resume and ONE JD.
    Used by Streamlit for live uploads.
    """
    r_vec = tfidf.transform([resume_text])
    j_vec = tfidf.transform([jd_text])
    return float(cosine_similarity(r_vec, j_vec)[0][0])


# ──────────────────────────────────────────────────
# 4. BUILD FEATURE MATRIX
# ──────────────────────────────────────────────────

def build_feature_matrix(train_resume, train_jd,
                          test_resume,  test_jd,
                          train_sim,    test_sim):
    """
    Combine TF-IDF resume + JD vectors + cosine similarity
    into one feature matrix per split.
    """
    train_sim_sparse = sp.csr_matrix(
        np.array(train_sim).reshape(-1, 1))
    test_sim_sparse  = sp.csr_matrix(
        np.array(test_sim).reshape(-1, 1))

    X_train = sp.hstack([train_resume, train_jd, train_sim_sparse])
    X_test  = sp.hstack([test_resume,  test_jd,  test_sim_sparse])

    print("✅ Feature matrix built!")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    return X_train, X_test


def build_single_feature_vector(resume_clean, jd_clean, tfidf):
    """
    Build feature vector for ONE resume-JD pair.
    Used by Streamlit for live prediction.
    """
    r_vec     = tfidf.transform([resume_clean])
    j_vec     = tfidf.transform([jd_clean])
    sim_score = cosine_similarity(r_vec, j_vec)[0][0]
    sim_sparse = sp.csr_matrix([[sim_score]])
    return sp.hstack([r_vec, j_vec, sim_sparse]), sim_score


# ──────────────────────────────────────────────────
# 5. ENCODE LABELS
# ──────────────────────────────────────────────────

def encode_labels(df_train, df_test):
    """
    Encode string labels to integers.
    Returns y_train, y_test, fitted LabelEncoder.
    """
    le = LabelEncoder()
    y_train = le.fit_transform(df_train['label'])
    y_test  = le.transform(df_test['label'])
    print(f"✅ Labels encoded: {le.classes_}")
    return y_train, y_test, le


# ──────────────────────────────────────────────────
# 6. SAVE FEATURES
# ──────────────────────────────────────────────────

def save_features(X_train, X_test, y_train, y_test,
                  tfidf, le, data_dir="../data",
                  models_dir="../models"):
    """Save all feature matrices, labels, vectorizer, encoder."""
    os.makedirs(data_dir,   exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    sp.save_npz(f"{data_dir}/X_train.npz", X_train)
    sp.save_npz(f"{data_dir}/X_test.npz",  X_test)
    np.save(f"{data_dir}/y_train.npy", y_train)
    np.save(f"{data_dir}/y_test.npy",  y_test)

    joblib.dump(tfidf, f"{models_dir}/tfidf_vectorizer.pkl")
    joblib.dump(le,    f"{models_dir}/label_encoder.pkl")
    print("✅ Features saved!")


# ──────────────────────────────────────────────────
# 7. LOAD FEATURES
# ──────────────────────────────────────────────────

def load_features(data_dir="../data", models_dir="../models"):
    """Load saved feature matrices and encoders."""
    X_train = sp.load_npz(f"{data_dir}/X_train.npz")
    X_test  = sp.load_npz(f"{data_dir}/X_test.npz")
    y_train = np.load(f"{data_dir}/y_train.npy")
    y_test  = np.load(f"{data_dir}/y_test.npy")
    tfidf   = joblib.load(f"{models_dir}/tfidf_vectorizer.pkl")
    le      = joblib.load(f"{models_dir}/label_encoder.pkl")
    print("✅ Features loaded!")
    return X_train, X_test, y_train, y_test, tfidf, le


# ──────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    df_train = pd.read_csv("../data/resume_jd_train_cleaned.csv")
    df_test  = pd.read_csv("../data/resume_jd_test_cleaned.csv")

    tfidf = build_tfidf(df_train)
    tr_r, tr_j, te_r, te_j = transform_tfidf(tfidf, df_train, df_test)

    train_sim = compute_cosine_similarity(tr_r, tr_j)
    test_sim  = compute_cosine_similarity(te_r, te_j)

    df_train['cosine_similarity'] = train_sim
    df_test['cosine_similarity']  = test_sim

    X_train, X_test = build_feature_matrix(
        tr_r, tr_j, te_r, te_j, train_sim, test_sim)

    y_train, y_test, le = encode_labels(df_train, df_test)

    save_features(X_train, X_test, y_train, y_test,
                  tfidf, le)

    df_train.to_csv("../data/resume_jd_train_cleaned.csv", index=False)
    df_test.to_csv("../data/resume_jd_test_cleaned.csv",  index=False)