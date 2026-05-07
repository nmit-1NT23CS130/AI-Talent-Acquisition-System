import os
import numpy as np
import scipy.sparse as sp
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# BASE DIRECTORY
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

data_dir = os.path.join(BASE_DIR, "data")

# ---------------------------------------------------
# LOAD TF-IDF FEATURES
# ---------------------------------------------------
X_train_tfidf = sp.load_npz(
    os.path.join(data_dir, "X_train.npz")
)

X_test_tfidf = sp.load_npz(
    os.path.join(data_dir, "X_test.npz")
)

# ---------------------------------------------------
# LOAD BERT EMBEDDINGS
# ---------------------------------------------------
train_resume_bert = np.load(
    os.path.join(data_dir, "train_resume_embeddings.npy")
)

train_jd_bert = np.load(
    os.path.join(data_dir, "train_jd_embeddings.npy")
)

test_resume_bert = np.load(
    os.path.join(data_dir, "test_resume_embeddings.npy")
)

test_jd_bert = np.load(
    os.path.join(data_dir, "test_jd_embeddings.npy")
)

# ---------------------------------------------------
# COMPUTE BERT SIMILARITY
# ---------------------------------------------------
print("Computing BERT similarity... ⏳")

train_bert_sim = np.array([
    cosine_similarity(
        train_resume_bert[i].reshape(1, -1),
        train_jd_bert[i].reshape(1, -1)
    )[0][0]

    for i in range(len(train_resume_bert))
])

test_bert_sim = np.array([
    cosine_similarity(
        test_resume_bert[i].reshape(1, -1),
        test_jd_bert[i].reshape(1, -1)
    )[0][0]

    for i in range(len(test_resume_bert))
])

# ---------------------------------------------------
# COMBINE BERT FEATURES
# ---------------------------------------------------
train_bert_sparse = sp.csr_matrix(np.hstack([
    train_resume_bert,
    train_jd_bert,
    train_bert_sim.reshape(-1, 1)
]))

test_bert_sparse = sp.csr_matrix(np.hstack([
    test_resume_bert,
    test_jd_bert,
    test_bert_sim.reshape(-1, 1)
]))

# ---------------------------------------------------
# FINAL COMBINED FEATURES
# ---------------------------------------------------
X_train_final = sp.hstack([
    X_train_tfidf,
    train_bert_sparse
])

X_test_final = sp.hstack([
    X_test_tfidf,
    test_bert_sparse
])

print(f"X_train_final: {X_train_final.shape}")
print(f"X_test_final: {X_test_final.shape}")

# ---------------------------------------------------
# SAVE UPDATED FEATURES
# ---------------------------------------------------
sp.save_npz(
    os.path.join(data_dir, "X_train_final.npz"),
    X_train_final
)

sp.save_npz(
    os.path.join(data_dir, "X_test_final.npz"),
    X_test_final
)

print("✅ Updated features saved!")