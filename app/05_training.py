"""
model_training.py
=================
Train, evaluate and save ML models.
Used by Streamlit frontend and notebooks.
"""

import numpy as np
import scipy.sparse as sp
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report,
                              confusion_matrix)


# ──────────────────────────────────────────────────
# 1. LOAD FEATURES
# ──────────────────────────────────────────────────

def load_all_features(data_dir="../data", models_dir="../models"):
    """
    Load TF-IDF + BERT combined feature matrices.
    Returns X_train, X_test, y_train, y_test, le.
    """
    # TF-IDF features
    X_train_tfidf = sp.load_npz(f"{data_dir}/X_train.npz")
    X_test_tfidf  = sp.load_npz(f"{data_dir}/X_test.npz")

    # BERT embeddings
    train_resume_bert = np.load(
        f"{data_dir}/train_resume_embeddings.npy")
    train_jd_bert     = np.load(
        f"{data_dir}/train_jd_embeddings.npy")
    test_resume_bert  = np.load(
        f"{data_dir}/test_resume_embeddings.npy")
    test_jd_bert      = np.load(
        f"{data_dir}/test_jd_embeddings.npy")

    # BERT cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity

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

    # Combine BERT features
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

    # Final combined feature matrix
    X_train = sp.hstack([X_train_tfidf, train_bert_sparse])
    X_test  = sp.hstack([X_test_tfidf,  test_bert_sparse])

    # Labels
    y_train = np.load(f"{data_dir}/y_train.npy")
    y_test  = np.load(f"{data_dir}/y_test.npy")
    le      = joblib.load(f"{models_dir}/label_encoder.pkl")

    print("✅ All features loaded!")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    return X_train, X_test, y_train, y_test, le


# ──────────────────────────────────────────────────
# 2. DEFINE MODELS
# ──────────────────────────────────────────────────

def get_models():
    """Return dictionary of ML models to train."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver='lbfgs'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        )
    }


# ──────────────────────────────────────────────────
# 3. TRAIN AND EVALUATE
# ──────────────────────────────────────────────────

def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train all 3 models and return results dictionary.
    Each entry has: model, accuracy, f1, y_pred.
    """
    models  = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='weighted')

        results[name] = {
            'model':    model,
            'accuracy': acc,
            'f1':       f1,
            'y_pred':   y_pred
        }
        print(f"   Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

    return results


# ──────────────────────────────────────────────────
# 4. SELECT BEST MODEL
# ──────────────────────────────────────────────────

def select_best_model(results):
    """
    Select model with highest F1 score.
    F1 preferred over accuracy for imbalanced data.
    Returns (best_name, best_model).
    """
    best_name  = max(results, key=lambda k: results[k]['f1'])
    best_model = results[best_name]['model']
    print(f"\n✅ Best model: {best_name}")
    print(f"   Accuracy: {results[best_name]['accuracy']:.4f}")
    print(f"   F1 Score: {results[best_name]['f1']:.4f}")
    return best_name, best_model


# ──────────────────────────────────────────────────
# 5. PRINT COMPARISON TABLE
# ──────────────────────────────────────────────────

def print_comparison(results):
    """Print model comparison table."""
    print("\n=== MODEL COMPARISON ===")
    print(f"{'Model':<25} {'Accuracy':>10} {'F1 Score':>10}")
    print("-" * 47)
    for name, res in results.items():
        print(f"{name:<25} {res['accuracy']:>10.4f} "
              f"{res['f1']:>10.4f}")


def get_classification_report(results, best_name, y_test, le):
    """Return classification report for best model."""
    y_pred = results[best_name]['y_pred']
    return classification_report(
        y_test, y_pred,
        target_names=le.classes_
    )


def get_confusion_matrix(results, best_name, y_test):
    """Return confusion matrix for best model."""
    y_pred = results[best_name]['y_pred']
    return confusion_matrix(y_test, y_pred)


# ──────────────────────────────────────────────────
# 6. SAVE BEST MODEL
# ──────────────────────────────────────────────────

def save_best_model(best_model, models_dir="../models"):
    """Save best model to disk."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, f"{models_dir}/best_model.pkl")
    print(f"✅ Best model saved → {models_dir}/best_model.pkl")


def load_best_model(models_dir="../models"):
    """Load saved best model from disk."""
    model = joblib.load(f"{models_dir}/best_model.pkl")
    print("✅ Best model loaded!")
    return model


# ──────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le = load_all_features()

    results = train_all_models(X_train, X_test, y_train, y_test)

    print_comparison(results)

    best_name, best_model = select_best_model(results)

    print("\n=== CLASSIFICATION REPORT ===")
    print(get_classification_report(results, best_name, y_test, le))

    save_best_model(best_model)