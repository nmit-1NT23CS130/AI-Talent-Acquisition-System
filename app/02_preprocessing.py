"""
preprocessing.py
================
Text cleaning and preprocessing functions.
Used by Streamlit frontend and notebooks.
"""

import pandas as pd
import re
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── globals ──────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))


# ──────────────────────────────────────────────────
# 1. LOAD CLEANED CSVs
# ──────────────────────────────────────────────────

def load_cleaned_data(data_dir="../data"):
    """Load already-cleaned train/test CSVs."""
    df_train = pd.read_csv(f"{data_dir}/resume_jd_train_cleaned.csv")
    df_test  = pd.read_csv(f"{data_dir}/resume_jd_test_cleaned.csv")
    print(f"✅ Cleaned data loaded — train: {df_train.shape}, test: {df_test.shape}")
    return df_train, df_test


# ──────────────────────────────────────────────────
# 2. CHECK DATA QUALITY
# ──────────────────────────────────────────────────

def check_data_quality(df):
    """Return missing value and duplicate counts."""
    quality = {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates":     int(df.duplicated().sum()),
    }
    return quality


def remove_duplicates(df):
    """Drop duplicate rows and return cleaned DataFrame."""
    before = len(df)
    df = df.drop_duplicates()
    after  = len(df)
    print(f"✅ Duplicates removed: {before - after} rows dropped")
    return df


# ──────────────────────────────────────────────────
# 3. TEXT CLEANING
# ──────────────────────────────────────────────────

def clean_text(text):
    """
    Full NLP cleaning pipeline:
    lowercase → remove emails/URLs/phones →
    remove special chars → tokenize →
    remove stopwords → lemmatize
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)                          # emails
    text = re.sub(r'http\S+|www\S+', '', text)                   # URLs
    text = re.sub(r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b',
                  '', text)                                       # phones
    text = re.sub(r'[^a-z0-9\s]', ' ', text)                    # special chars
    text = re.sub(r'\s+', ' ', text).strip()                     # extra spaces

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 2
    ]
    return ' '.join(tokens)


def remove_gender_signals(text):
    """
    Mask gendered pronouns and titles with [MASKED].
    Applied BEFORE clean_text for fairness.
    """
    if not isinstance(text, str):
        return ""
    pattern = r'\b(he|she|him|her|his|hers|mr|ms|mrs|miss|male|female)\b'
    return re.sub(pattern, '[MASKED]', text, flags=re.IGNORECASE)


def preprocess_text(text):
    """
    Combined pipeline: gender masking → cleaning.
    Use this for any single resume or JD string.
    """
    return clean_text(remove_gender_signals(text))


# ──────────────────────────────────────────────────
# 4. APPLY CLEANING TO FULL DATAFRAME
# ──────────────────────────────────────────────────

def preprocess_dataframe(df_train, df_test):
    """
    Apply full cleaning to train and test DataFrames.
    Adds resume_clean and jd_clean columns.
    """
    print("Cleaning resumes... ⏳")
    df_train['resume_clean'] = (df_train['resume_text']
                                .apply(remove_gender_signals)
                                .apply(clean_text))

    print("Cleaning JDs... ⏳")
    df_train['jd_clean'] = df_train['job_description_text'].apply(clean_text)

    print("Cleaning test set... ⏳")
    df_test['resume_clean'] = (df_test['resume_text']
                               .apply(remove_gender_signals)
                               .apply(clean_text))
    df_test['jd_clean'] = df_test['job_description_text'].apply(clean_text)

    print("✅ Cleaning complete!")
    return df_train, df_test


# ──────────────────────────────────────────────────
# 5. SAVE CLEANED DATA
# ──────────────────────────────────────────────────

def save_cleaned_data(df_train, df_test, data_dir="../data"):
    """Save cleaned DataFrames to CSV."""
    df_train.to_csv(f"{data_dir}/resume_jd_train_cleaned.csv", index=False)
    df_test.to_csv(f"{data_dir}/resume_jd_test_cleaned.csv",  index=False)
    print("✅ Cleaned data saved!")
    print(f"   Columns: {df_train.columns.tolist()}")


# ──────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    df_train = pd.read_csv("../data/resume_jd_train.csv")
    df_test  = pd.read_csv("../data/resume_jd_test.csv")

    df_train = remove_duplicates(df_train)

    quality = check_data_quality(df_train)
    print("Missing values:", quality["missing_values"])
    print("Duplicates:", quality["duplicates"])

    df_train, df_test = preprocess_dataframe(df_train, df_test)
    save_cleaned_data(df_train, df_test)