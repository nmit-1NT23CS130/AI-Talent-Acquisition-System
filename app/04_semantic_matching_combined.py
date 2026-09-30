import pandas as pd
import numpy as np
import joblib
import os

from sentence_transformers import SentenceTransformer


# =========================
# PATHS
# =========================

TRAIN_FILE = "data/combined/combined_train.csv"
TEST_FILE = "data/combined/combined_test.csv"

TRAIN_RESUME_EMB = "data/combined/resume_embeddings_combined_train.npy"
TRAIN_JD_EMB = "data/combined/jd_embeddings_combined_train.npy"

TEST_RESUME_EMB = "data/combined/resume_embeddings_combined_test.npy"
TEST_JD_EMB = "data/combined/jd_embeddings_combined_test.npy"

MODEL_FILE = "data/combined/bert_model_combined.pkl"


# =========================
# LOAD DATA
# =========================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Train:", train_df.shape)
print("Test :", test_df.shape)


# =========================
# LOAD SENTENCE TRANSFORMER
# =========================

print("\nLoading Sentence-BERT model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# TEXT
# =========================

train_resumes = train_df["resume_text"].fillna("").tolist()
train_jds = train_df["job_description_text"].fillna("").tolist()

test_resumes = test_df["resume_text"].fillna("").tolist()
test_jds = test_df["job_description_text"].fillna("").tolist()


# =========================
# TRAIN RESUME EMBEDDINGS
# =========================

print("\nCreating TRAIN resume embeddings...")

train_resume_embeddings = model.encode(
    train_resumes,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(
    "Train resume embeddings:",
    train_resume_embeddings.shape
)


# =========================
# TRAIN JD EMBEDDINGS
# =========================

print("\nCreating TRAIN JD embeddings...")

train_jd_embeddings = model.encode(
    train_jds,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(
    "Train JD embeddings:",
    train_jd_embeddings.shape
)


# =========================
# TEST RESUME EMBEDDINGS
# =========================

print("\nCreating TEST resume embeddings...")

test_resume_embeddings = model.encode(
    test_resumes,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(
    "Test resume embeddings:",
    test_resume_embeddings.shape
)


# =========================
# TEST JD EMBEDDINGS
# =========================

print("\nCreating TEST JD embeddings...")

test_jd_embeddings = model.encode(
    test_jds,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(
    "Test JD embeddings:",
    test_jd_embeddings.shape
)


# =========================
# SAVE EMBEDDINGS
# =========================

np.save(
    TRAIN_RESUME_EMB,
    train_resume_embeddings
)

np.save(
    TRAIN_JD_EMB,
    train_jd_embeddings
)

np.save(
    TEST_RESUME_EMB,
    test_resume_embeddings
)

np.save(
    TEST_JD_EMB,
    test_jd_embeddings
)

joblib.dump(
    model,
    MODEL_FILE
)


# =========================
# FINAL OUTPUT
# =========================

print("\n=========================")
print("SEMANTIC MATCHING DONE")
print("=========================")

print(
    "Train resume:",
    train_resume_embeddings.shape
)

print(
    "Train JD:",
    train_jd_embeddings.shape
)

print(
    "Test resume:",
    test_resume_embeddings.shape
)

print(
    "Test JD:",
    test_jd_embeddings.shape
)

print("\nSaved:")
print(TRAIN_RESUME_EMB)
print(TRAIN_JD_EMB)
print(TEST_RESUME_EMB)
print(TEST_JD_EMB)
print(MODEL_FILE)