import numpy as np
import pandas as pd
import joblib
import os

from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PATHS
# =========================================================

X_TRAIN_FILE = "data/combined/X_combined_train.npz"
X_TEST_FILE = "data/combined/X_combined_test.npz"

Y_TRAIN_FILE = "data/combined/y_combined_train.npy"
Y_TEST_FILE = "data/combined/y_combined_test.npy"

TRAIN_RESUME_EMB = "data/combined/resume_embeddings_combined_train.npy"
TRAIN_JD_EMB = "data/combined/jd_embeddings_combined_train.npy"

TEST_RESUME_EMB = "data/combined/resume_embeddings_combined_test.npy"
TEST_JD_EMB = "data/combined/jd_embeddings_combined_test.npy"

LABEL_ENCODER_FILE = "data/combined/label_encoder_combined.pkl"

MODEL_FILE = "models/best_model_combined.pkl"


# =========================================================
# LOAD TF-IDF FEATURES
# =========================================================

print("Loading TF-IDF features...")

X_train_tfidf = load_npz(X_TRAIN_FILE)
X_test_tfidf = load_npz(X_TEST_FILE)

y_train = np.load(Y_TRAIN_FILE)
y_test = np.load(Y_TEST_FILE)

print("Train TF-IDF:", X_train_tfidf.shape)
print("Test TF-IDF :", X_test_tfidf.shape)


# =========================================================
# LOAD BERT EMBEDDINGS
# =========================================================

print("\nLoading BERT embeddings...")

train_resume_emb = np.load(TRAIN_RESUME_EMB)
train_jd_emb = np.load(TRAIN_JD_EMB)

test_resume_emb = np.load(TEST_RESUME_EMB)
test_jd_emb = np.load(TEST_JD_EMB)

print("Train resume BERT:", train_resume_emb.shape)
print("Train JD BERT    :", train_jd_emb.shape)
print("Test resume BERT :", test_resume_emb.shape)
print("Test JD BERT     :", test_jd_emb.shape)


# =========================================================
# BERT COSINE SIMILARITY
# =========================================================

print("\nCalculating BERT cosine similarity...")

train_bert_similarity = np.array([
    cosine_similarity(
        train_resume_emb[i:i+1],
        train_jd_emb[i:i+1]
    )[0][0]
    for i in range(len(train_resume_emb))
]).reshape(-1, 1)

test_bert_similarity = np.array([
    cosine_similarity(
        test_resume_emb[i:i+1],
        test_jd_emb[i:i+1]
    )[0][0]
    for i in range(len(test_resume_emb))
]).reshape(-1, 1)

print(
    "Train BERT similarity:",
    train_bert_similarity.shape
)

print(
    "Test BERT similarity:",
    test_bert_similarity.shape
)


# =========================================================
# CONVERT BERT FEATURES TO SPARSE MATRICES
# =========================================================

train_bert_features = csr_matrix(
    np.hstack([
        train_resume_emb,
        train_jd_emb,
        train_bert_similarity
    ])
)

test_bert_features = csr_matrix(
    np.hstack([
        test_resume_emb,
        test_jd_emb,
        test_bert_similarity
    ])
)


# =========================================================
# COMBINE TF-IDF + BERT
# =========================================================

print("\nCombining TF-IDF + BERT features...")

X_train = hstack([
    X_train_tfidf,
    train_bert_features
]).tocsr()

X_test = hstack([
    X_test_tfidf,
    test_bert_features
]).tocsr()

print("\nFinal feature shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# =========================================================
# EXPECTED FEATURE COUNT
# =========================================================

expected_features = 10769

if X_train.shape[1] != expected_features:
    raise ValueError(
        f"Unexpected feature count. "
        f"Expected {expected_features}, "
        f"got {X_train.shape[1]}"
    )

if X_test.shape[1] != expected_features:
    raise ValueError(
        f"Unexpected test feature count. "
        f"Expected {expected_features}, "
        f"got {X_test.shape[1]}"
    )

print("\nFeature count check passed: 10,769")


# =========================================================
# LOAD LABEL ENCODER
# =========================================================

label_encoder = joblib.load(LABEL_ENCODER_FILE)

print("\nClasses:")
print(label_encoder.classes_)


# =========================================================
# MODEL 1 — LOGISTIC REGRESSION
# =========================================================

print("\n==============================")
print("TRAINING LOGISTIC REGRESSION")
print("==============================")

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(
    y_test,
    lr_pred,
    average="weighted"
)

print("\nLogistic Regression")
print("Accuracy:", round(lr_accuracy, 4))
print("Weighted F1:", round(lr_f1, 4))

print(
    classification_report(
        y_test,
        lr_pred,
        target_names=label_encoder.classes_
    )
)


# =========================================================
# MODEL 2 — RANDOM FOREST
# =========================================================

print("\n==============================")
print("TRAINING RANDOM FOREST")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(
    y_test,
    rf_pred,
    average="weighted"
)

print("\nRandom Forest")
print("Accuracy:", round(rf_accuracy, 4))
print("Weighted F1:", round(rf_f1, 4))

print(
    classification_report(
        y_test,
        rf_pred,
        target_names=label_encoder.classes_
    )
)


# =========================================================
# MODEL 3 — XGBOOST
# =========================================================

print("\n==============================")
print("TRAINING XGBOOST")
print("==============================")

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)

xgb_f1 = f1_score(
    y_test,
    xgb_pred,
    average="weighted"
)

xgb_macro_f1 = f1_score(
    y_test,
    xgb_pred,
    average="macro"
)

print("\nXGBoost")
print("Accuracy:", round(xgb_accuracy, 4))
print("Weighted F1:", round(xgb_f1, 4))
print("Macro F1:", round(xgb_macro_f1, 4))

print(
    classification_report(
        y_test,
        xgb_pred,
        target_names=label_encoder.classes_
    )
)


# =========================================================
# SAVE BEST MODEL
# =========================================================

models = {
    "Logistic Regression": (
        lr_model,
        lr_accuracy,
        lr_f1
    ),
    "Random Forest": (
        rf_model,
        rf_accuracy,
        rf_f1
    ),
    "XGBoost": (
        xgb_model,
        xgb_accuracy,
        xgb_f1
    )
}


best_model_name = max(
    models,
    key=lambda name: models[name][2]
)

best_model = models[best_model_name][0]

joblib.dump(
    best_model,
    MODEL_FILE
)


# =========================================================
# FINAL COMPARISON
# =========================================================

print("\n======================================")
print("FINAL MODEL COMPARISON")
print("======================================")

print(
    f"Logistic Regression : "
    f"Accuracy = {lr_accuracy:.4f}, "
    f"Weighted F1 = {lr_f1:.4f}"
)

print(
    f"Random Forest       : "
    f"Accuracy = {rf_accuracy:.4f}, "
    f"Weighted F1 = {rf_f1:.4f}"
)

print(
    f"XGBoost             : "
    f"Accuracy = {xgb_accuracy:.4f}, "
    f"Weighted F1 = {xgb_f1:.4f}"
)

print("\nBest model based on weighted F1:")
print(best_model_name)

print("\nSaved best model:")
print(MODEL_FILE)