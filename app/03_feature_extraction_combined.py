import pandas as pd
import numpy as np
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, save_npz


# =========================
# PATHS
# =========================

TRAIN_FILE = "data/combined/combined_train.csv"
TEST_FILE = "data/combined/combined_test.csv"

X_TRAIN_FILE = "data/combined/X_combined_train.npz"
X_TEST_FILE = "data/combined/X_combined_test.npz"

Y_TRAIN_FILE = "data/combined/y_combined_train.npy"
Y_TEST_FILE = "data/combined/y_combined_test.npy"

VECTORIZER_FILE = "data/combined/tfidf_vectorizer_combined.pkl"
LABEL_ENCODER_FILE = "data/combined/label_encoder_combined.pkl"


# =========================
# LOAD DATA
# =========================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Train:", train_df.shape)
print("Test :", test_df.shape)


# =========================
# TEXT COLUMNS
# =========================

train_resume = train_df["resume_text"].fillna("")
train_jd = train_df["job_description_text"].fillna("")

test_resume = test_df["resume_text"].fillna("")
test_jd = test_df["job_description_text"].fillna("")


# =========================
# TF-IDF
# =========================

print("\nCreating TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)


# IMPORTANT:
# Fit ONLY on training text
all_train_text = pd.concat(
    [train_resume, train_jd],
    ignore_index=True
)

vectorizer.fit(all_train_text)


# =========================
# TRANSFORM RESUMES
# =========================

print("Transforming resumes...")

X_train_resume = vectorizer.transform(train_resume)
X_test_resume = vectorizer.transform(test_resume)


# =========================
# TRANSFORM JOB DESCRIPTIONS
# =========================

print("Transforming job descriptions...")

X_train_jd = vectorizer.transform(train_jd)
X_test_jd = vectorizer.transform(test_jd)


# =========================
# COMBINE TF-IDF FEATURES
# =========================

X_train = hstack([
    X_train_resume,
    X_train_jd
]).tocsr()

X_test = hstack([
    X_test_resume,
    X_test_jd
]).tocsr()


print("\nTF-IDF feature shapes:")
print("Train:", X_train.shape)
print("Test :", X_test.shape)


# =========================
# LABEL ENCODING
# =========================

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(train_df["label"])
y_test = label_encoder.transform(test_df["label"])

print("\nLabels:")
print(label_encoder.classes_)


# =========================
# SAVE
# =========================

save_npz(X_TRAIN_FILE, X_train)
save_npz(X_TEST_FILE, X_test)

np.save(Y_TRAIN_FILE, y_train)
np.save(Y_TEST_FILE, y_test)

joblib.dump(vectorizer, VECTORIZER_FILE)
joblib.dump(label_encoder, LABEL_ENCODER_FILE)


print("\n=========================")
print("FEATURE EXTRACTION DONE")
print("=========================")

print("Train TF-IDF:", X_train.shape)
print("Test TF-IDF :", X_test.shape)

print("\nSaved:")
print(X_TRAIN_FILE)
print(X_TEST_FILE)
print(Y_TRAIN_FILE)
print(Y_TEST_FILE)
print(VECTORIZER_FILE)
print(LABEL_ENCODER_FILE)