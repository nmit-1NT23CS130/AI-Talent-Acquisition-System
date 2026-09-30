import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os

# =========================
# PATHS
# =========================

INPUT_FILE = "data/combined/combined_resume_jd_dataset.csv"

TRAIN_FILE = "data/combined/combined_train.csv"
TEST_FILE = "data/combined/combined_test.csv"


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(INPUT_FILE)

print("Original dataset:", df.shape)

# Keep only required columns
df = df[
    ["resume_text", "job_description_text", "label"]
].dropna()

# Remove exact duplicate resume-JD-label combinations
df = df.drop_duplicates(
    subset=["resume_text", "job_description_text", "label"]
).reset_index(drop=True)

print("After duplicate removal:", df.shape)


# =========================
# BASIC CLEANING
# =========================

def clean_text(text):
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    return " ".join(text.split())


df["resume_text"] = df["resume_text"].apply(clean_text)
df["job_description_text"] = df["job_description_text"].apply(clean_text)


# =========================
# GROUPED SPLIT
# =========================

groups = df["resume_text"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, groups=groups)
)

train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)


# =========================
# SAVE
# =========================

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


# =========================
# VALIDATION
# =========================

train_resumes = set(train_df["resume_text"])
test_resumes = set(test_df["resume_text"])

train_jds = set(train_df["job_description_text"])
test_jds = set(test_df["job_description_text"])

resume_overlap = train_resumes & test_resumes
jd_overlap = train_jds & test_jds

print("\n=========================")
print("COMBINED SPLIT")
print("=========================")

print("Train:", train_df.shape)
print("Test :", test_df.shape)

print("\nUnique resumes")
print("Train:", len(train_resumes))
print("Test :", len(test_resumes))
print("Overlap:", len(resume_overlap))

print("\nUnique JDs")
print("Train:", len(train_jds))
print("Test :", len(test_jds))
print("Overlap:", len(jd_overlap))

print("\nTrain labels:")
print(train_df["label"].value_counts())

print("\nTest labels:")
print(test_df["label"].value_counts())

print("\nSaved:")
print(TRAIN_FILE)
print(TEST_FILE)