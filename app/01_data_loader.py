"""
data_loader.py
==============
Functions to load and explore the resume-JD dataset.
Used by Streamlit frontend and notebooks.
"""

import pandas as pd
from datasets import load_dataset
import os


# ──────────────────────────────────────────────
# 1. LOAD DATASET FROM HUGGINGFACE
# ──────────────────────────────────────────────

def load_raw_dataset():
    """
    Downloads and returns the dataset from HuggingFace.
    Returns train and test as pandas DataFrames.
    """
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("cnamuangtoun/resume-job-description-fit")

    df_train = pd.DataFrame(ds['train'])
    df_test  = pd.DataFrame(ds['test'])

    print("✅ Dataset loaded!")
    print(f"   Train shape: {df_train.shape}")
    print(f"   Test shape:  {df_test.shape}")
    print(f"   Columns:     {df_train.columns.tolist()}")

    return df_train, df_test


# ──────────────────────────────────────────────
# 2. SAVE DATASET TO CSV
# ──────────────────────────────────────────────

def save_dataset(df_train, df_test, data_dir="../data"):
    """
    Saves train and test DataFrames as CSV files.
    Creates the data directory if it doesn't exist.
    """
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "resume_jd_train.csv")
    test_path  = os.path.join(data_dir, "resume_jd_test.csv")

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path,  index=False)

    print("✅ Dataset saved!")
    print(f"   Train → {train_path}")
    print(f"   Test  → {test_path}")


# ──────────────────────────────────────────────
# 3. LOAD DATASET FROM CSV
# ──────────────────────────────────────────────

def load_from_csv(data_dir="../data"):
    """
    Loads already saved CSV files.
    Use this instead of HuggingFace after first download.
    Returns train and test DataFrames.
    """
    train_path = os.path.join(data_dir, "resume_jd_train.csv")
    test_path  = os.path.join(data_dir, "resume_jd_test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"CSV not found at {train_path}. "
            "Run load_raw_dataset() and save_dataset() first!"
        )

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    print("✅ Dataset loaded from CSV!")
    print(f"   Train shape: {df_train.shape}")
    print(f"   Test shape:  {df_test.shape}")

    return df_train, df_test


# ──────────────────────────────────────────────
# 4. GET DATASET STATISTICS
# ──────────────────────────────────────────────

def get_dataset_stats(df_train, df_test):
    """
    Returns a dictionary of dataset statistics.
    Used by Streamlit to display dataset info.
    """
    stats = {
        "total_pairs":      len(df_train) + len(df_test),
        "train_pairs":      len(df_train),
        "test_pairs":       len(df_test),
        "unique_resumes":   df_train['resume_text'].nunique(),
        "unique_jds":       df_train['job_description_text'].nunique(),
        "label_counts":     df_train['label'].value_counts().to_dict(),
        "columns":          df_train.columns.tolist(),
    }
    return stats


# ──────────────────────────────────────────────
# 5. GET SAMPLE ROW
# ──────────────────────────────────────────────

def get_sample(df, index=0, preview_chars=300):
    """
    Returns a sample row from the dataset.
    Useful for displaying examples in Streamlit.
    """
    row = df.iloc[index]
    sample = {
        "resume_preview": row['resume_text'][:preview_chars] + "...",
        "jd_preview":     row['job_description_text'][:preview_chars] + "...",
        "label":          row['label'],
    }
    return sample


# ──────────────────────────────────────────────
# 6. MAIN — Run directly to download & save
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # Step 1: Download from HuggingFace
    df_train, df_test = load_raw_dataset()

    # Step 2: Save to CSV
    save_dataset(df_train, df_test)

    # Step 3: Print stats
    stats = get_dataset_stats(df_train, df_test)

    print("\n=== DATASET STATISTICS ===")
    print(f"Total pairs:     {stats['total_pairs']}")
    print(f"Train pairs:     {stats['train_pairs']}")
    print(f"Test pairs:      {stats['test_pairs']}")
    print(f"Unique Resumes:  {stats['unique_resumes']}")
    print(f"Unique JDs:      {stats['unique_jds']}")
    print(f"\nLabel Distribution:")
    for label, count in stats['label_counts'].items():
        print(f"  {label:<15} → {count}")

    # Step 4: Show sample
    sample = get_sample(df_train, index=0)
    print(f"\n=== SAMPLE ROW ===")
    print(f"Label:  {sample['label']}")
    print(f"Resume: {sample['resume_preview']}")
    print(f"JD:     {sample['jd_preview']}")