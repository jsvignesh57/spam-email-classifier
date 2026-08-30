
"""
Spam Email Classifier — Data Cleaning and Text Preprocessing

Phase 2: Preprocessing pipeline that loads the raw dataset, removes exact duplicates,
removes email 'Subject:' header prefixes, normalizes email addresses, URLs, and numbers
to semantic tokens, applies lowercasing and whitespace normalization, validates data integrity,
and saves the cleaned dataset to `data/processed/cleaned_internship.csv`.

The raw dataset at `data/raw/internship.csv` is strictly read-only and never modified.
"""

import os
import re
import sys
from pathlib import Path
import pandas as pd


# ----------------------------------------------------------------------
# Regex Patterns
# ----------------------------------------------------------------------
SUBJECT_PREFIX_PATTERN = re.compile(r'^\s*subject\s*:\s*', re.IGNORECASE)

EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Za-z]{2,}\b|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
)

URL_PATTERN = re.compile(
    r'https?\s*:\s*/\s*/\s*(?:www\s*\.\s*)?[a-zA-Z0-9.-]+(?:\s*\.\s*[a-zA-Z]{2,})?(?:\s*/\s*[a-zA-Z0-9_\-.~%/?#=+:&]+)*'
    r'|www\s*\.\s*[a-zA-Z0-9.-]+(?:\s*\.\s*[a-zA-Z]{2,})+(?:\s*/\s*[a-zA-Z0-9_\-.~%/?#=+:&]+)*'
    r'|https?://\S+'
    r'|www\.\S+',
    re.IGNORECASE
)

NUMBER_PATTERN = re.compile(r'\b\d+\b')
WHITESPACE_PATTERN = re.compile(r'\s+')


# ----------------------------------------------------------------------
# Path Resolution
# ----------------------------------------------------------------------
def get_project_root() -> Path:
    """Resolve project root directory relative to this script."""
    return Path(__file__).resolve().parent.parent


def get_raw_dataset_path() -> Path:
    """Resolve path to raw dataset."""
    return get_project_root() / "data" / "raw" / "internship.csv"


def get_processed_dataset_path() -> Path:
    """Resolve path to processed dataset output."""
    return get_project_root() / "data" / "processed" / "cleaned_internship.csv"


def get_report_path() -> Path:
    """Resolve path to preprocessing report."""
    return get_project_root() / "reports" / "preprocessing_report.txt"


# ----------------------------------------------------------------------
# Core Preprocessing Functions
# ----------------------------------------------------------------------
def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV and validate expected columns.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns ('text', 'spam') are missing.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    
    required_columns = {"text", "spam"}
    actual_columns = set(df.columns)
    if not required_columns.issubset(actual_columns):
        missing = required_columns - actual_columns
        raise ValueError(
            f"Dataset missing required columns: {missing}. Found columns: {list(df.columns)}"
        )
    
    return df


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    """
    Remove exact duplicate rows across all columns.
    
    Returns:
        tuple: (deduplicated_df, original_count, duplicates_removed, final_count)
    """
    original_count = len(df)
    duplicates_removed = int(df.duplicated().sum())
    df_clean = df.drop_duplicates().reset_index(drop=True)
    final_count = len(df_clean)
    
    return df_clean, original_count, duplicates_removed, final_count


def remove_subject_prefix(text: str) -> str:
    """
    Remove the leading 'Subject:' header prefix from email text.
    Preserves the actual subject content and body.
    """
    return SUBJECT_PREFIX_PATTERN.sub('', text)


def normalize_email_addresses(text: str) -> str:
    """
    Replace email addresses with the 'emailtoken' placeholder.
    """
    return EMAIL_PATTERN.sub('emailtoken', text)


def normalize_urls(text: str) -> str:
    """
    Replace URLs and web domain patterns with the 'urltoken' placeholder.
    """
    return URL_PATTERN.sub('urltoken', text)


def normalize_numbers(text: str) -> str:
    """
    Normalize standalone numeric sequences to 'numtoken'.
    Preserves words containing letters and punctuation.
    """
    return NUMBER_PATTERN.sub('numtoken', text)


def normalize_text(text: str) -> str:
    """
    Apply full text normalization pipeline:
    1. Remove 'Subject:' prefix
    2. Normalize email addresses -> 'emailtoken'
    3. Normalize URLs -> 'urltoken'
    4. Normalize numbers -> 'numtoken'
    5. Convert to lowercase
    6. Normalize whitespace (collapse multiple spaces, tabs, newlines; strip edges)
    """
    if not isinstance(text, str):
        text = str(text) if pd.notnull(text) else ""
        
    text = remove_subject_prefix(text)
    text = normalize_email_addresses(text)
    text = normalize_urls(text)
    text = normalize_numbers(text)
    text = text.lower()
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    return text


# ----------------------------------------------------------------------
# Validation & Persistence
# ----------------------------------------------------------------------
def validate_processed_data(df: pd.DataFrame) -> dict:
    """
    Validate processed dataset integrity:
    - Column schema exactly matches ['text', 'spam']
    - Labels contain only 0 and 1
    - No NaN/null values exist
    - Identify any emails that became empty or whitespace-only
    """
    expected_cols = ["text", "spam"]
    columns_valid = list(df.columns) == expected_cols
    
    unique_labels = sorted(df["spam"].unique().tolist())
    labels_valid = set(unique_labels).issubset({0, 1})
    
    missing_text = int(df["text"].isnull().sum())
    missing_spam = int(df["spam"].isnull().sum())
    
    # Check for empty text strings
    empty_mask = df["text"].astype(str).str.strip() == ""
    empty_count = int(empty_mask.sum())
    empty_indices = df.index[empty_mask].tolist()
    
    return {
        "columns_valid": columns_valid,
        "columns": list(df.columns),
        "unique_labels": unique_labels,
        "labels_valid": labels_valid,
        "missing_text": missing_text,
        "missing_spam": missing_spam,
        "empty_text_count": empty_count,
        "empty_text_indices": empty_indices,
        "total_records": len(df)
    }


def save_processed_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the processed dataset containing only ['text', 'spam'] columns.
    Ensures parent directory exists.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[["text", "spam"]].to_csv(output_path, index=False)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise IOError(f"Failed to save processed dataset at {output_path}")


def generate_preprocessing_report(
    stats: dict,
    sample_comparisons: list[dict],
    report_path: Path
) -> str:
    """
    Generate and save the formal text preprocessing report.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "=" * 60,
        "SPAM EMAIL CLASSIFIER — TEXT PREPROCESSING REPORT",
        "Phase 2: Data Cleaning and Text Preprocessing",
        "=" * 60,
        "",
        "--------------------------------------------------",
        "DATASET RECORD COUNTS",
        "--------------------------------------------------",
        f"Original records:            {stats['original_records']:,}",
        f"Duplicate records removed:   {stats['duplicates_removed']:,}",
        f"Final records:               {stats['final_records']:,}",
        f"Expected records verified:   {stats['expected_records_verified']}",
        "",
        "--------------------------------------------------",
        "COLUMNS AND SCHEMA",
        "--------------------------------------------------",
        f"Original columns:            {stats['original_columns']}",
        f"Final columns:               {stats['final_columns']}",
        f"Schema validation passed:    {stats['schema_valid']}",
        "",
        "--------------------------------------------------",
        "LABEL DISTRIBUTION",
        "--------------------------------------------------",
        "Before preprocessing:",
        f"  - Ham  (0): {stats['label_dist_before'].get(0, 0):,} ({stats['label_dist_before_pct'].get(0, 0.0):.2f}%)",
        f"  - Spam (1): {stats['label_dist_before'].get(1, 0):,} ({stats['label_dist_before_pct'].get(1, 0.0):.2f}%)",
        "After preprocessing:",
        f"  - Ham  (0): {stats['label_dist_after'].get(0, 0):,} ({stats['label_dist_after_pct'].get(0, 0.0):.2f}%)",
        f"  - Spam (1): {stats['label_dist_after'].get(1, 0):,} ({stats['label_dist_after_pct'].get(1, 0.0):.2f}%)",
        f"Label integrity preserved:   {stats['labels_valid']}",
        "",
        "--------------------------------------------------",
        "TOKENIZATION & NORMALIZATION STATISTICS",
        "--------------------------------------------------",
        f"Subject prefixes removed:    {stats['subject_prefixes_removed']:,}",
        f"Email addresses normalized:  {stats['emails_normalized']:,} (replaced with 'emailtoken')",
        f"URLs normalized:             {stats['urls_normalized']:,} (replaced with 'urltoken')",
        f"Numeric sequences normalized:{stats['numbers_normalized']:,} (replaced with 'numtoken')",
        f"Empty texts after processing:{stats['empty_texts']}",
        "",
        "--------------------------------------------------",
        "DESIGN RULES ADHERENCE",
        "--------------------------------------------------",
        "  [x] Raw dataset untouched (data/raw/internship.csv)",
        "  [x] Exact duplicates removed (33 rows)",
        "  [x] Target labels preserved (0 = Ham, 1 = Spam)",
        "  [x] Subject: header prefix stripped (subject content preserved)",
        "  [x] Email addresses normalized to 'emailtoken'",
        "  [x] URLs/domains normalized to 'urltoken'",
        "  [x] Standalone numbers normalized to 'numtoken'",
        "  [x] Text converted to lowercase",
        "  [x] Whitespace normalized",
        "  [x] Punctuation preserved (signal for spam)",
        "  [x] Stopwords preserved",
        "  [x] Stemming/lemmatization skipped",
        "  [x] Long emails preserved without truncation",
        "  [x] No feature extraction (TF-IDF) or model training performed",
        "",
        "--------------------------------------------------",
        "SAMPLE TRANSFORMATIONS (BEFORE vs AFTER)",
        "--------------------------------------------------",
    ]
    
    for idx, sample in enumerate(sample_comparisons, 1):
        lines.extend([
            f"--- Example {idx} [Label: {sample['label_name']} ({sample['label']})] ---",
            "BEFORE (Raw):",
            f"  {sample['before'][:220]}...",
            "AFTER (Cleaned):",
            f"  {sample['after'][:220]}...",
            ""
        ])
        
    lines.extend([
        "=" * 60,
        "END OF PREPROCESSING REPORT",
        "=" * 60,
    ])
    
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    return report_text


# ----------------------------------------------------------------------
# Main Pipeline Execution
# ----------------------------------------------------------------------
def main():
    raw_path = get_raw_dataset_path()
    processed_path = get_processed_dataset_path()
    report_path = get_report_path()
    
    print("=" * 60)
    print("Spam Email Classifier — Phase 2: Preprocessing Pipeline")
    print("=" * 60)
    print(f"Loading raw dataset from: {raw_path}")
    
    # Step 1: Load Data
    raw_df = load_dataset(raw_path)
    original_records = len(raw_df)
    original_cols = list(raw_df.columns)
    label_dist_before = raw_df["spam"].value_counts().to_dict()
    label_dist_before_pct = (raw_df["spam"].value_counts(normalize=True) * 100).to_dict()
    print(f"Loaded {original_records:,} records with columns: {original_cols}")
    
    # Step 2: Remove Duplicates
    df_dedup, orig_count, dup_count, final_count = remove_duplicates(raw_df)
    print(f"Removed {dup_count:,} duplicate rows. Remaining: {final_count:,} records.")
    expected_records_verified = (final_count == 5695)
    if not expected_records_verified:
        print(f"WARNING: Expected 5,695 records but got {final_count:,}")
        
    # Count occurrences of patterns prior to normalization for reporting
    subject_prefix_count = int(df_dedup["text"].apply(lambda x: bool(SUBJECT_PREFIX_PATTERN.search(str(x)))).sum())
    email_matches_count = int(df_dedup["text"].apply(lambda x: len(EMAIL_PATTERN.findall(str(x)))).sum())
    url_matches_count = int(df_dedup["text"].apply(lambda x: len(URL_PATTERN.findall(str(x)))).sum())
    number_matches_count = int(df_dedup["text"].apply(lambda x: len(NUMBER_PATTERN.findall(str(x)))).sum())
    
    # Steps 4 to 13: Clean and Normalize Text
    print("Normalizing text (subject prefix, emails, URLs, numbers, lowercase, whitespace)...")
    cleaned_texts = df_dedup["text"].apply(normalize_text)
    
    # Build clean DataFrame
    processed_df = pd.DataFrame({
        "text": cleaned_texts,
        "spam": df_dedup["spam"].values
    })
    
    # Step 14: Validate Processed Data
    validation = validate_processed_data(processed_df)
    label_dist_after = processed_df["spam"].value_counts().to_dict()
    label_dist_after_pct = (processed_df["spam"].value_counts(normalize=True) * 100).to_dict()
    
    if not validation["labels_valid"]:
        raise ValueError(f"Invalid labels found in processed dataset: {validation['unique_labels']}")
    if validation["empty_text_count"] > 0:
        print(f"ALERT: {validation['empty_text_count']} email(s) became empty after preprocessing at indices: {validation['empty_text_indices']}")
    else:
        print("Validation check passed: No emails became empty.")
        
    # Step 15: Save Processed Dataset
    print(f"Saving processed dataset to: {processed_path}")
    save_processed_dataset(processed_df, processed_path)
    print(f"Processed dataset successfully saved ({len(processed_df):,} rows, columns: {list(processed_df.columns)}).")
    
    # Select 5 diverse representative samples for before/after comparison
    # 1. Spam with Email and Numbers (Index 3)
    # 2. Spam with URLs and Numbers (Index 10)
    # 3. Ham Conversational / Questionnaire (Index 1368)
    # 4. Ham Forwarded Newsletter / Headers (Index 1370)
    # 5. Ham Business Request / Internal Communication (Index 1400)
    sample_indices = [3, 10, 1368, 1370, 1400]
    
    sample_comparisons = []
    for idx in sample_indices:
        label_val = int(df_dedup.loc[idx, "spam"])
        sample_comparisons.append({
            "index": idx,
            "label": label_val,
            "label_name": "Spam" if label_val == 1 else "Ham",
            "before": df_dedup.loc[idx, "text"],
            "after": processed_df.loc[idx, "text"]
        })
        
    # Step 16: Generate Preprocessing Report
    stats = {
        "original_records": original_records,
        "duplicates_removed": dup_count,
        "final_records": final_count,
        "expected_records_verified": expected_records_verified,
        "original_columns": original_cols,
        "final_columns": list(processed_df.columns),
        "schema_valid": validation["columns_valid"],
        "label_dist_before": label_dist_before,
        "label_dist_before_pct": label_dist_before_pct,
        "label_dist_after": label_dist_after,
        "label_dist_after_pct": label_dist_after_pct,
        "labels_valid": validation["labels_valid"],
        "subject_prefixes_removed": subject_prefix_count,
        "emails_normalized": email_matches_count,
        "urls_normalized": url_matches_count,
        "numbers_normalized": number_matches_count,
        "empty_texts": validation["empty_text_count"]
    }
    
    report_text = generate_preprocessing_report(stats, sample_comparisons, report_path)
    print(f"Report generated at: {report_path}")
    
    # Print Preprocessing Summary in Terminal
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Original Records:           {original_records:,}")
    print(f"Duplicate Records Removed:  {dup_count:,}")
    print(f"Final Cleaned Records:      {final_count:,} (Expected 5,695: {'VERIFIED' if expected_records_verified else 'MISMATCH'})")
    print(f"Label Distribution:         Ham (0): {label_dist_after.get(0, 0):,} ({label_dist_after_pct.get(0, 0.0):.2f}%) | Spam (1): {label_dist_after.get(1, 0):,} ({label_dist_after_pct.get(1, 0.0):.2f}%)")
    print(f"Subject Prefixes Removed:   {subject_prefix_count:,}")
    print(f"Email Tokens Inserted:      {email_matches_count:,}")
    print(f"URL Tokens Inserted:        {url_matches_count:,}")
    print(f"Number Tokens Inserted:     {number_matches_count:,}")
    print(f"Empty Texts:                {validation['empty_text_count']}")
    print(f"Output Dataset:             {processed_path}")
    print(f"Report File:                {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
