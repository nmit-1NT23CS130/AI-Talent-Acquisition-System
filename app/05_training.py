"""
model_training.py
=================
Train, evaluate and save ML models.
Used by Streamlit frontend and notebooks.
"""

import os
import numpy as np
import scipy.sparse as sp
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────
# BASE DIRECTORIES
# ──────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────
# 1. LOAD FEATURES
# ──────────────────────────────────────────────────

def load_all_features():
    """
    Load TF-IDF + BERT combined feature matrices.
    Returns X_train, X_test, y_train, y_test, le.
    """

    # ------------------------------------------------
    # TF-IDF FEATURES
    # ------------------------------------------------
    X_train_tfidf = sp.load_npz(
        os.path.join(DATA_DIR, "X_train.npz")
    )

    X_test_tfidf = sp.load_npz(
        os.path.join(DATA_DIR, "X_test.npz")
    )

    # ------------------------------------------------
    # BERT EMBEDDINGS
    # ------------------------------------------------
    train_resume_bert = np.load(
        os.path.join(DATA_DIR, "train_resume_embeddings.npy")
    )

    train_jd_bert = np.load(
        os.path.join(DATA_DIR, "train_jd_embeddings.npy")
    )

    test_resume_bert = np.load(
        os.path.join(DATA_DIR, "test_resume_embeddings.npy")
    )

    test_jd_bert = np.load(
        os.path.join(DATA_DIR, "test_jd_embeddings.npy")
    )

    # ------------------------------------------------
    # BERT COSINE SIMILARITY
    # ------------------------------------------------
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

    # ------------------------------------------------
    # COMBINE BERT FEATURES
    # ------------------------------------------------
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

    # ------------------------------------------------
    # FINAL FEATURE MATRICES
    # ------------------------------------------------
    X_train = sp.hstack([
        X_train_tfidf,
        train_bert_sparse
    ])

    X_test = sp.hstack([
        X_test_tfidf,
        test_bert_sparse
    ])

    # ------------------------------------------------
    # LABELS
    # ------------------------------------------------
    y_train = np.load(
        os.path.join(DATA_DIR, "y_train.npy")
    )

    y_test = np.load(
        os.path.join(DATA_DIR, "y_test.npy")
    )

    le = joblib.load(
        os.path.join(MODELS_DIR, "label_encoder.pkl")
    )

    print("✅ All features loaded!")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")

    return X_train, X_test, y_train, y_test, le


# ──────────────────────────────────────────────────
# 2. DEFINE MODELS
# ──────────────────────────────────────────────────

def get_models():

    return {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1
        )
    }


# ──────────────────────────────────────────────────
# 3. TRAIN MODELS
# ──────────────────────────────────────────────────

def train_all_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    models = get_models()

    results = {}

    for name, model in models.items():

        print(f"\n🔹 Training {name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted"
        )

        results[name] = {

            "model": model,
            "accuracy": acc,
            "f1": f1,
            "y_pred": y_pred
        }

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")

    return results


# ──────────────────────────────────────────────────
# 4. SELECT BEST MODEL
# ──────────────────────────────────────────────────

def select_best_model(results):

    best_name = max(
        results,
        key=lambda k: results[k]["f1"]
    )

    best_model = results[best_name]["model"]

    print(f"\n✅ Best Model: {best_name}")

    return best_name, best_model


# ──────────────────────────────────────────────────
# 5. PRINT COMPARISON
# ──────────────────────────────────────────────────

def print_comparison(results):

    print("\n=== MODEL COMPARISON ===")

    print(f"{'Model':<25} {'Accuracy':>10} {'F1 Score':>10}")

    print("-" * 47)

    for name, res in results.items():

        print(
            f"{name:<25} "
            f"{res['accuracy']:>10.4f} "
            f"{res['f1']:>10.4f}"
        )


# ──────────────────────────────────────────────────
# 6. CLASSIFICATION REPORT
# ──────────────────────────────────────────────────

def get_classification_report(
    results,
    best_name,
    y_test,
    le
):

    y_pred = results[best_name]["y_pred"]

    return classification_report(
        y_test,
        y_pred,
        target_names=le.classes_
    )


def get_confusion_matrix(
    results,
    best_name,
    y_test
):

    y_pred = results[best_name]["y_pred"]

    return confusion_matrix(
        y_test,
        y_pred
    )


# ──────────────────────────────────────────────────
# 7. SAVE / LOAD MODEL
# ──────────────────────────────────────────────────

def save_best_model(best_model):

    model_path = os.path.join(
        MODELS_DIR,
        "best_model.pkl"
    )

    joblib.dump(best_model, model_path)

    print(f"✅ Best model saved → {model_path}")


def load_best_model():

    model_path = os.path.join(
        MODELS_DIR,
        "best_model.pkl"
    )

    model = joblib.load(model_path)

    print("✅ Best model loaded!")

    return model


# ──────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────

if __name__ == "__main__":

    X_train, X_test, y_train, y_test, le = load_all_features()

    results = train_all_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print_comparison(results)

    best_name, best_model = select_best_model(results)

    print("\n=== CLASSIFICATION REPORT ===")

    print(
        get_classification_report(
            results,
            best_name,
            y_test,
            le
        )
    )

    save_best_model(best_model)