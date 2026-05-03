"""
semantic_matching.py
====================
Sentence-BERT embeddings and semantic similarity.
Used by Streamlit frontend and notebooks.
"""

import numpy as np
import joblib
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────
# 1. LOAD BERT MODEL
# ──────────────────────────────────────────────────

def load_bert_model(model_name='all-MiniLM-L6-v2'):
    """
    Load Sentence-BERT model.
    Downloads ~90MB on first run.
    """
    print(f"Loading BERT model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("✅ BERT model loaded!")
    return model


# ──────────────────────────────────────────────────
# 2. GENERATE EMBEDDINGS
# ──────────────────────────────────────────────────

def generate_embeddings(model, texts, batch_size=32):
    """
    Generate BERT embeddings for a list of texts.
    Shows progress bar during encoding.
    Returns numpy array of shape (n, 384).
    """
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=batch_size
    )
    return embeddings


def generate_all_embeddings(model, df_train, df_test):
    """
    Generate embeddings for all train and test
    resumes and JDs.
    Returns 4 numpy arrays.
    """
    print("Generating train resume embeddings... ⏳")
    train_resume = generate_embeddings(
        model, df_train['resume_clean'].tolist())

    print("Generating train JD embeddings... ⏳")
    train_jd = generate_embeddings(
        model, df_train['jd_clean'].tolist())

    print("Generating test resume embeddings... ⏳")
    test_resume = generate_embeddings(
        model, df_test['resume_clean'].tolist())

    print("Generating test JD embeddings... ⏳")
    test_jd = generate_embeddings(
        model, df_test['jd_clean'].tolist())

    print("✅ All embeddings generated!")
    print(f"   Train resume shape: {train_resume.shape}")
    return train_resume, train_jd, test_resume, test_jd


# ──────────────────────────────────────────────────
# 3. COMPUTE BERT SIMILARITY
# ──────────────────────────────────────────────────

def compute_bert_similarity(resume_embeddings, jd_embeddings):
    """
    Compute cosine similarity for each resume-JD pair.
    Returns list of float scores.
    """
    scores = [
        float(cosine_similarity(
            resume_embeddings[i].reshape(1, -1),
            jd_embeddings[i].reshape(1, -1)
        )[0][0])
        for i in range(len(resume_embeddings))
    ]
    return scores


def compute_single_bert_similarity(resume_text, jd_text, model):
    """
    Compute BERT similarity for ONE resume and ONE JD.
    Used by Streamlit for live uploads.
    """
    r_emb = model.encode([resume_text])
    j_emb = model.encode([jd_text])
    return float(cosine_similarity(r_emb, j_emb)[0][0])


# ──────────────────────────────────────────────────
# 4. SAVE / LOAD EMBEDDINGS
# ──────────────────────────────────────────────────

def save_embeddings(train_resume, train_jd,
                    test_resume,  test_jd,
                    model, data_dir="../data",
                    models_dir="../models"):
    """Save BERT embeddings and model to disk."""
    os.makedirs(data_dir,   exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    np.save(f"{data_dir}/train_resume_embeddings.npy", train_resume)
    np.save(f"{data_dir}/train_jd_embeddings.npy",     train_jd)
    np.save(f"{data_dir}/test_resume_embeddings.npy",  test_resume)
    np.save(f"{data_dir}/test_jd_embeddings.npy",      test_jd)

    joblib.dump(model, f"{models_dir}/bert_model.pkl")
    print("✅ Embeddings and model saved!")


def load_embeddings(data_dir="../data"):
    """Load saved BERT embeddings from disk."""
    train_resume = np.load(
        f"{data_dir}/train_resume_embeddings.npy")
    train_jd     = np.load(
        f"{data_dir}/train_jd_embeddings.npy")
    test_resume  = np.load(
        f"{data_dir}/test_resume_embeddings.npy")
    test_jd      = np.load(
        f"{data_dir}/test_jd_embeddings.npy")
    print("✅ Embeddings loaded!")
    return train_resume, train_jd, test_resume, test_jd


# ──────────────────────────────────────────────────
# 5. COMPARE TF-IDF VS BERT
# ──────────────────────────────────────────────────

def compare_tfidf_bert(df_train):
    """
    Print average TF-IDF vs BERT similarity per label.
    Requires cosine_similarity and bert_similarity columns.
    """
    print("=== TF-IDF vs BERT COMPARISON ===\n")
    for label in ['Good Fit', 'Potential Fit', 'No Fit']:
        subset   = df_train[df_train['label'] == label]
        tfidf_avg = subset['cosine_similarity'].mean()
        bert_avg  = subset['bert_similarity'].mean()
        print(f"{label}:")
        print(f"  TF-IDF avg: {tfidf_avg:.4f}")
        print(f"  BERT avg:   {bert_avg:.4f}\n")


# ──────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    df_train = pd.read_csv("../data/resume_jd_train_cleaned.csv")
    df_test  = pd.read_csv("../data/resume_jd_test_cleaned.csv")

    model = load_bert_model()

    tr_r, tr_j, te_r, te_j = generate_all_embeddings(
        model, df_train, df_test)

    train_scores = compute_bert_similarity(tr_r, tr_j)
    test_scores  = compute_bert_similarity(te_r, te_j)

    df_train['bert_similarity'] = train_scores
    df_test['bert_similarity']  = test_scores

    compare_tfidf_bert(df_train)

    save_embeddings(tr_r, tr_j, te_r, te_j, model)

    df_train.to_csv(
        "../data/resume_jd_train_cleaned.csv", index=False)
    df_test.to_csv(
        "../data/resume_jd_test_cleaned.csv",  index=False)