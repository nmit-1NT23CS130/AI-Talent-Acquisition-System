import pandas as pd
import numpy as np
import joblib

from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PATHS
# =========================================================

COMBINED_TEST = "data/combined/combined_test.csv"

X_TEST_FILE = "data/combined/X_combined_test.npz"
Y_TEST_FILE = "data/combined/y_combined_test.npy"

TEST_RESUME_EMB = (
    "data/combined/resume_embeddings_combined_test.npy"
)

TEST_JD_EMB = (
    "data/combined/jd_embeddings_combined_test.npy"
)

MODEL_FILE = "models/best_model_combined.pkl"

LABEL_ENCODER_FILE = (
    "data/combined/label_encoder_combined.pkl"
)

# HF
HF_TRAIN = "data/resume_grouped_train_cleaned.csv"
HF_TEST = "data/resume_grouped_test_cleaned.csv"

# Synthetic
SYN_TRAIN = "data/enhanced/enhanced_train.csv"
SYN_TEST = "data/enhanced/enhanced_test.csv"


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    return " ".join(text.split())


def make_key(df):

    resume = (
        df["resume_text"]
        .fillna("")
        .apply(clean_text)
    )

    jd = (
        df["job_description_text"]
        .fillna("")
        .apply(clean_text)
    )

    return resume + " ||| " + jd


# =========================================================
# LOAD COMBINED TEST
# =========================================================

print("Loading combined test data...")

combined_test = pd.read_csv(COMBINED_TEST)

print(
    "Combined test:",
    combined_test.shape
)

combined_test["source_key"] = make_key(
    combined_test
)


# =========================================================
# LOAD HF DATA
# =========================================================

print("\nLoading HF grouped data...")

hf_train = pd.read_csv(HF_TRAIN)
hf_test = pd.read_csv(HF_TEST)

hf_all = pd.concat(
    [hf_train, hf_test],
    ignore_index=True
)

hf_keys = set(
    make_key(hf_all)
)

print("HF rows:", len(hf_all))
print("Unique HF keys:", len(hf_keys))


# =========================================================
# LOAD SYNTHETIC DATA
# =========================================================

print("\nLoading synthetic data...")

syn_train = pd.read_csv(SYN_TRAIN)
syn_test = pd.read_csv(SYN_TEST)

syn_all = pd.concat(
    [syn_train, syn_test],
    ignore_index=True
)

syn_keys = set(
    make_key(syn_all)
)

print(
    "Synthetic rows:",
    len(syn_all)
)

print(
    "Unique synthetic keys:",
    len(syn_keys)
)


# =========================================================
# IDENTIFY SOURCE
# =========================================================

def identify_source(key):

    in_hf = key in hf_keys
    in_syn = key in syn_keys

    if in_hf and in_syn:
        return "Both"

    if in_hf:
        return "HF"

    if in_syn:
        return "Synthetic"

    return "Unknown"


print("\nIdentifying sources...")

combined_test["source"] = (
    combined_test["source_key"]
    .apply(identify_source)
)


print("\n======================================")
print("INITIAL SOURCE MATCH")
print("======================================")

print(
    combined_test["source"]
    .value_counts()
)


# =========================================================
# IMPORTANT FALLBACK
# =========================================================
#
# The combined dataset was created ONLY from:
#
#   1. HF dataset
#   2. Synthetic dataset
#
# Therefore, if a row is not found in HF,
# it belongs to the synthetic portion.
#
# We only use this after checking that there
# are no "Both" rows.
# =========================================================

both_count = (
    combined_test["source"] == "Both"
).sum()

if both_count > 0:

    print(
        "\nWARNING:",
        both_count,
        "rows appear in BOTH source datasets."
    )

else:

    unknown_count = (
        combined_test["source"] == "Unknown"
    ).sum()

    if unknown_count > 0:

        print(
            "\nAssigning unmatched rows to Synthetic "
            "because the combined dataset was created "
            "from exactly HF + Synthetic."
        )

        combined_test.loc[
            combined_test["source"] == "Unknown",
            "source"
        ] = "Synthetic"


print("\n======================================")
print("FINAL SOURCE DISTRIBUTION")
print("======================================")

print(
    combined_test["source"]
    .value_counts()
)


# =========================================================
# LOAD FEATURES
# =========================================================

print("\nLoading test features...")

X_test_tfidf = load_npz(
    X_TEST_FILE
)

y_test = np.load(
    Y_TEST_FILE
)

resume_emb = np.load(
    TEST_RESUME_EMB
)

jd_emb = np.load(
    TEST_JD_EMB
)

print(
    "TF-IDF:",
    X_test_tfidf.shape
)

print(
    "Resume BERT:",
    resume_emb.shape
)

print(
    "JD BERT:",
    jd_emb.shape
)


# =========================================================
# BERT COSINE SIMILARITY
# =========================================================

print(
    "\nCalculating BERT cosine similarity..."
)

bert_similarity = np.array([

    cosine_similarity(
        resume_emb[i:i+1],
        jd_emb[i:i+1]
    )[0][0]

    for i in range(
        len(resume_emb)
    )

]).reshape(-1, 1)


print(
    "BERT similarity:",
    bert_similarity.shape
)


# =========================================================
# CREATE BERT FEATURES
# =========================================================

bert_features = csr_matrix(
    np.hstack([
        resume_emb,
        jd_emb,
        bert_similarity
    ])
)


# =========================================================
# COMBINE
# =========================================================

X_test = hstack([
    X_test_tfidf,
    bert_features
]).tocsr()


print("\n======================================")
print("FINAL TEST FEATURES")
print("======================================")

print(
    "X_test:",
    X_test.shape
)


# =========================================================
# CHECK FEATURE COUNT
# =========================================================

if X_test.shape[1] != 10769:

    raise ValueError(
        f"Expected 10769 features, "
        f"got {X_test.shape[1]}"
    )

print(
    "Feature count check passed: 10,769"
)


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading combined model...")

model = joblib.load(
    MODEL_FILE
)

label_encoder = joblib.load(
    LABEL_ENCODER_FILE
)


# =========================================================
# PREDICT
# =========================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test
)


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_source(source_name):

    mask = (
        combined_test["source"].values
        == source_name
    )

    count = mask.sum()

    if count == 0:

        print(
            f"\nNo rows found for {source_name}"
        )

        return

    y_true = y_test[mask]
    y_pred = predictions[mask]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    print("\n======================================")
    print(source_name.upper())
    print("======================================")

    print(
        "Samples:",
        count
    )

    print(
        "Accuracy:",
        round(accuracy, 4)
    )

    print(
        "Macro F1:",
        round(macro_f1, 4)
    )

    print(
        "Weighted F1:",
        round(weighted_f1, 4)
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_
        )
    )


# =========================================================
# HF
# =========================================================

evaluate_source(
    "HF"
)


# =========================================================
# SYNTHETIC
# =========================================================

evaluate_source(
    "Synthetic"
)


# =========================================================
# OVERALL
# =========================================================

print("\n======================================")
print("OVERALL COMBINED TEST SET")
print("======================================")

print(
    "Samples:",
    len(y_test)
)

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            predictions
        ),
        4
    )
)

print(
    "Macro F1:",
    round(
        f1_score(
            y_test,
            predictions,
            average="macro"
        ),
        4
    )
)

print(
    "Weighted F1:",
    round(
        f1_score(
            y_test,
            predictions,
            average="weighted"
        ),
        4
    )
)