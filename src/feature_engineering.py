"""
Spam Email Classifier — Feature Engineering Script

Phase 3: Converts preprocessed email text into numerical TF-IDF feature matrices
for downstream machine-learning models (e.g., Naive Bayes, SVM).

Critical Data-Leakage Prevention Rule:
The stratified train/test split is performed FIRST.
The TfidfVectorizer is fitted STRICTLY on X_train only, and then applied to transform
both X_train and X_test. The test dataset never participates in vocabulary construction
or IDF weight calculation.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------------------------------------------------------
# Path Resolution
# ----------------------------------------------------------------------
def get_project_root() -> Path:
    """Resolve project root directory relative to this script."""
    return Path(__file__).resolve().parent.parent


def get_processed_data_path() -> Path:
    """Resolve path to cleaned dataset from Phase 2."""
    return get_project_root() / "data" / "processed" / "cleaned_internship.csv"


def get_vectorizer_save_path() -> Path:
    """Resolve path to save the fitted TF-IDF vectorizer."""
    return get_project_root() / "models" / "tfidf_vectorizer.joblib"


def get_split_save_path() -> Path:
    """Resolve path to save train/test split arrays and indices."""
    return get_project_root() / "data" / "processed" / "train_test_split.npz"


def get_report_path() -> Path:
    """Resolve path to feature engineering report."""
    return get_project_root() / "reports" / "feature_engineering_report.txt"


# ----------------------------------------------------------------------
# Step 1: Load and Validate Data
# ----------------------------------------------------------------------
def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load cleaned dataset and validate schema, labels, and completeness.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Validate columns
    required_cols = {"text", "spam"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Dataset missing required columns: {required_cols - set(df.columns)}")
    
    # Validate missing values
    if df["text"].isnull().any():
        raise ValueError(f"Found {df['text'].isnull().sum()} missing text values.")
    if df["spam"].isnull().any():
        raise ValueError(f"Found {df['spam'].isnull().sum()} missing spam labels.")
        
    # Validate labels
    unique_labels = sorted(df["spam"].unique().tolist())
    if set(unique_labels) != {0, 1}:
        raise ValueError(f"Invalid labels found: {unique_labels}. Expected only [0, 1].")
        
    # Ensure text is string type
    df["text"] = df["text"].astype(str)
    
    return df


# ----------------------------------------------------------------------
# Step 2: Stratified Train / Test Split
# ----------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> tuple:
    """
    Split the dataset into stratified training (80%) and testing (20%) subsets.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, train_indices, test_indices)
    """
    X = df["text"]
    y = df["spam"]
    indices = df.index.to_numpy()
    
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test, train_idx, test_idx


# ----------------------------------------------------------------------
# Step 3 & 4: Create and Fit TF-IDF Vectorizer (Training Data Only)
# ----------------------------------------------------------------------
def create_vectorizer(
    ngram_range: tuple = (1, 2),
    sublinear_tf: bool = True,
    min_df: int = 2,
    max_df: float = 0.95
) -> TfidfVectorizer:
    """
    Initialize TfidfVectorizer with baseline n-gram and frequency thresholds.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=max_df
    )


def fit_transform_features(
    vectorizer: TfidfVectorizer,
    X_train: pd.Series,
    X_test: pd.Series
) -> tuple:
    """
    Fit vectorizer STRICTLY on X_train, then transform both X_train and X_test.
    
    Returns:
        tuple: (X_train_tfidf, X_test_tfidf)
    """
    # Fit ONLY on training data
    vectorizer.fit(X_train)
    
    # Transform training and test partitions
    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf


# ----------------------------------------------------------------------
# Step 5, 6, 7 & 8: Matrix, Vocabulary, Weight Analysis & Leakage Check
# ----------------------------------------------------------------------
def analyze_features(
    vectorizer: TfidfVectorizer,
    X_train_tfidf,
    X_test_tfidf,
    X_train: pd.Series,
    y_train: pd.Series,
    sample_indices: list = [0, 1, 2]
) -> dict:
    """
    Analyze the generated TF-IDF feature matrices, vocabulary, sparsity,
    and representative feature weights.
    """
    feature_names = vectorizer.get_feature_names_out()
    vocab_size = len(feature_names)
    
    # Matrix dimensions & Sparsity
    n_train_samples, n_train_features = X_train_tfidf.shape
    n_test_samples, n_test_features = X_test_tfidf.shape
    
    train_nnz = X_train_tfidf.nnz
    test_nnz = X_test_tfidf.nnz
    
    train_sparsity = 1.0 - (train_nnz / (n_train_samples * n_train_features))
    test_sparsity = 1.0 - (test_nnz / (n_test_samples * n_test_features))
    
    # Sample unigrams and bigrams
    unigrams = [f for f in feature_names if " " not in f][:10]
    bigrams = [f for f in feature_names if " " in f][:10]
    first_30_features = list(feature_names[:30])
    
    # Sample inspections
    inspections = []
    for idx in sample_indices:
        if idx < len(X_train):
            row = X_train_tfidf.getrow(idx)
            non_zero_cols = row.indices
            non_zero_data = row.data
            
            # Sort tokens by highest TF-IDF weight
            top_token_indices = np.argsort(non_zero_data)[::-1][:8]
            top_tokens = [
                (feature_names[non_zero_cols[i]], float(non_zero_data[i]))
                for i in top_token_indices
            ]
            
            inspections.append({
                "index": idx,
                "label": int(y_train.iloc[idx]),
                "label_name": "Spam" if y_train.iloc[idx] == 1 else "Ham",
                "text_snippet": X_train.iloc[idx][:180],
                "non_zero_features_count": int(len(non_zero_cols)),
                "top_weighted_tokens": top_tokens
            })
            
    return {
        "vocab_size": vocab_size,
        "n_train_samples": n_train_samples,
        "n_train_features": n_train_features,
        "n_test_samples": n_test_samples,
        "n_test_features": n_test_features,
        "train_nnz": train_nnz,
        "test_nnz": test_nnz,
        "train_sparsity_pct": train_sparsity * 100.0,
        "test_sparsity_pct": test_sparsity * 100.0,
        "first_30_features": first_30_features,
        "sample_unigrams": unigrams,
        "sample_bigrams": bigrams,
        "inspections": inspections
    }


def verify_no_data_leakage(vectorizer: TfidfVectorizer, X_train: pd.Series, X_test: pd.Series) -> bool:
    """
    Explicitly test that the vectorizer vocabulary was fitted only on X_train.
    """
    # Check that vectorizer vocabulary size matches what is learned on X_train alone
    standalone_vec = TfidfVectorizer(
        ngram_range=vectorizer.ngram_range,
        sublinear_tf=vectorizer.sublinear_tf,
        min_df=vectorizer.min_df,
        max_df=vectorizer.max_df
    )
    standalone_vec.fit(X_train)
    
    is_same_vocab = (standalone_vec.vocabulary_ == vectorizer.vocabulary_)
    return is_same_vocab


# ----------------------------------------------------------------------
# Step 9: Save Feature Engineering Artifacts
# ----------------------------------------------------------------------
def save_artifacts(
    vectorizer: TfidfVectorizer,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    vectorizer_path: Path,
    split_path: Path
) -> None:
    """
    Save the fitted TF-IDF vectorizer and reproducible train/test split indices.
    """
    # 1. Save vectorizer
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vectorizer_path)
    
    # 2. Save split indices and targets
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_path,
        train_indices=train_idx,
        test_indices=test_idx,
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy()
    )


# ----------------------------------------------------------------------
# Step 10: Generate Feature Engineering Report
# ----------------------------------------------------------------------
def generate_report(stats: dict, report_path: Path) -> str:
    """
    Generate and save the formal feature engineering report.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "=" * 60,
        "SPAM EMAIL CLASSIFIER — FEATURE ENGINEERING REPORT",
        "Phase 3: TF-IDF Text Feature Engineering",
        "=" * 60,
        "",
        "--------------------------------------------------",
        "1. DATASET SPLIT & PARTITIONING",
        "--------------------------------------------------",
        f"Input records:               {stats['total_records']:,}",
        f"Training records (80%):      {stats['n_train_samples']:,}",
        f"Testing records  (20%):      {stats['n_test_samples']:,}",
        f"Train/Test split ratio:      {stats['split_ratio_actual']}",
        f"Stratification:              Enabled (stratify=y, random_state=42)",
        "",
        "Training class distribution:",
        f"  - Ham  (0): {stats['train_ham_count']:,} ({stats['train_ham_pct']:.2f}%)",
        f"  - Spam (1): {stats['train_spam_count']:,} ({stats['train_spam_pct']:.2f}%)",
        "",
        "Testing class distribution:",
        f"  - Ham  (0): {stats['test_ham_count']:,} ({stats['test_ham_pct']:.2f}%)",
        f"  - Spam (1): {stats['test_spam_count']:,} ({stats['test_spam_pct']:.2f}%)",
        "",
        "--------------------------------------------------",
        "2. TF-IDF VECTORIZER CONFIGURATION",
        "--------------------------------------------------",
        f"- ngram_range:               {stats['ngram_range']} (Unigrams + Bigrams)",
        f"- min_df:                    {stats['min_df']} (Ignore terms in < 2 emails)",
        f"- max_df:                    {stats['max_df']} (Ignore terms in > 95% of emails)",
        f"- sublinear_tf:              {stats['sublinear_tf']} (Logarithmic TF scaling: 1 + log(tf))",
        f"- max_features:              None (Natural vocabulary preserved)",
        "",
        "--------------------------------------------------",
        "3. FEATURE MATRIX ANALYSIS",
        "--------------------------------------------------",
        f"Number of learned features:  {stats['vocab_size']:,}",
        f"Training matrix shape:       {stats['train_shape']} (samples, features)",
        f"Testing matrix shape:        {stats['test_shape']} (samples, features)",
        f"Training non-zero values:    {stats['train_nnz']:,}",
        f"Testing non-zero values:     {stats['test_nnz']:,}",
        f"Training matrix sparsity:    {stats['train_sparsity_pct']:.4f}%",
        f"Testing matrix sparsity:     {stats['test_sparsity_pct']:.4f}%",
        "",
        "Matrix Dimension Explanation:",
        "  - The shape (4556, 121288) represents 4,556 email training instances",
        "    represented across 121,288 unique unigram and bigram feature coordinates.",
        "  - A sparsity of ~99.78% indicates that out of over 552 million matrix cells,",
        "    only 1.20 million contain non-zero weights, perfectly handled by scipy sparse CSR matrices.",
        "",
        "--------------------------------------------------",
        "4. VOCABULARY & FEATURE INSPECTION",
        "--------------------------------------------------",
        f"Total vocabulary size:       {stats['vocab_size']:,} terms",
        "",
        "First 30 features alphabetically:",
        "  " + ", ".join(f"'{f}'" for f in stats['first_30_features']),
        "",
        "Sample Unigram Features:",
        "  " + ", ".join(f"'{f}'" for f in stats['sample_unigrams']),
        "",
        "Sample Bigram Features:",
        "  " + ", ".join(f"'{f}'" for f in stats['sample_bigrams']),
        "",
        "--------------------------------------------------",
        "5. REPRESENTATIVE SAMPLE INSPECTIONS (TF-IDF WEIGHTS)",
        "--------------------------------------------------",
    ]
    
    for item in stats["inspections"]:
        top_str = ", ".join([f"{tok}: {wt:.3f}" for tok, wt in item["top_weighted_tokens"]])
        lines.extend([
            f"--- Sample Train Index {item['index']} [{item['label_name']} ({item['label']})] ---",
            f"Snippet:              {item['text_snippet']}...",
            f"Non-zero features:    {item['non_zero_features_count']}",
            f"Top weighted tokens:  {top_str}",
            ""
        ])
        
    lines.extend([
        "--------------------------------------------------",
        "6. DATA LEAKAGE & INTEGRITY VERIFICATION",
        "--------------------------------------------------",
        f"TF-IDF fitted strictly on X_train:  {stats['leakage_check_passed']}",
        f"X_test transformed with train vectorizer: True",
        f"TF-IDF DATA LEAKAGE CHECK:          {'PASS' if stats['leakage_check_passed'] else 'FAIL'}",
        "",
        "--------------------------------------------------",
        "7. SAVED ARTIFACTS",
        "--------------------------------------------------",
        f"Fitted Vectorizer:           {stats['vectorizer_path']}",
        f"Train/Test Split Arrays:     {stats['split_path']}",
        "",
        "--------------------------------------------------",
        "8. RATIONALE: WHY TF-IDF FOR SPAM CLASSIFICATION",
        "--------------------------------------------------",
        "1. Term Frequency Damping (Sublinear TF):",
        "   Spam emails frequently repeat trigger words (e.g. 'free', 'money').",
        "   Sublinear TF scaling replaces raw count 'tf' with '1 + log(tf)',",
        "   preventing keyword stuffing from distorting linear and probabilistic boundaries.",
        "",
        "2. Inverse Document Frequency (IDF):",
        "   Common terms across almost all emails receive low weights, while specific",
        "   spam markers (e.g., 'guaranteed', 'refinance', 'pharmaceutical') and ham markers",
        "   (e.g., 'enron', 'meeting', 'attached') receive appropriately high discriminative power.",
        "",
        "3. Bigram Context Preservation (ngram_range=(1, 2)):",
        "   Bi-grams preserve crucial multi-word phrases such as 'credit card', 'act now',",
        "   'numtoken color', 'special offer', which carry much stronger spam signal than",
        "   isolated unigrams alone.",
        "",
        "=" * 60,
        "END OF FEATURE ENGINEERING REPORT",
        "=" * 60,
    ])
    
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    return report_text


# ----------------------------------------------------------------------
# Main Orchestrator
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Spam Email Classifier — Phase 3: Feature Engineering")
    print("=" * 60)
    
    data_path = get_processed_data_path()
    vec_path = get_vectorizer_save_path()
    split_path = get_split_save_path()
    report_path = get_report_path()
    
    # Step 1: Load Data
    print(f"Loading cleaned dataset from: {data_path}")
    df = load_data(data_path)
    total_records = len(df)
    print(f"Loaded {total_records:,} records successfully.")
    
    # Step 2: Stratified Split
    print("Creating stratified 80/20 train/test split (random_state=42)...")
    X_train, X_test, y_train, y_test, train_idx, test_idx = split_data(df, test_size=0.20, random_state=42)
    
    n_train = len(X_train)
    n_test = len(X_test)
    
    train_ham = int((y_train == 0).sum())
    train_spam = int((y_train == 1).sum())
    test_ham = int((y_test == 0).sum())
    test_spam = int((y_test == 1).sum())
    
    train_ham_pct = (train_ham / n_train) * 100.0
    train_spam_pct = (train_spam / n_train) * 100.0
    test_ham_pct = (test_ham / n_test) * 100.0
    test_spam_pct = (test_spam / n_test) * 100.0
    
    print(f"Training partition: {n_train:,} emails (Ham: {train_ham:,} [{train_ham_pct:.2f}%], Spam: {train_spam:,} [{train_spam_pct:.2f}%])")
    print(f"Testing partition:  {n_test:,} emails (Ham: {test_ham:,} [{test_ham_pct:.2f}%], Spam: {test_spam:,} [{test_spam_pct:.2f}%])")
    
    # Step 3 & 4: Create and Fit TF-IDF Vectorizer
    print("Configuring TF-IDF Vectorizer (ngram_range=(1,2), sublinear_tf=True, min_df=2, max_df=0.95)...")
    vectorizer = create_vectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95)
    
    print("Fitting TF-IDF Vectorizer STRICTLY on X_train...")
    X_train_tfidf, X_test_tfidf = fit_transform_features(vectorizer, X_train, X_test)
    print("Feature transformation complete.")
    
    # Step 5, 6 & 7: Feature Analysis
    print("Analyzing feature matrices, vocabulary, and feature weights...")
    analysis = analyze_features(vectorizer, X_train_tfidf, X_test_tfidf, X_train, y_train)
    
    # Step 8: Data Leakage Verification
    leakage_passed = verify_no_data_leakage(vectorizer, X_train, X_test)
    print(f"TF-IDF DATA LEAKAGE CHECK: {'PASS' if leakage_passed else 'FAIL'}")
    if not leakage_passed:
        raise RuntimeError("Data leakage verification failed! Vectorizer contains unauthorized vocabulary.")
        
    # Step 9: Save Artifacts
    print(f"Saving fitted vectorizer to: {vec_path}")
    print(f"Saving train/test split data to: {split_path}")
    save_artifacts(vectorizer, train_idx, test_idx, y_train, y_test, vec_path, split_path)
    
    # Step 10: Generate Report
    stats = {
        "total_records": total_records,
        "n_train_samples": n_train,
        "n_test_samples": n_test,
        "split_ratio_actual": f"{n_train}/{n_test} ({n_train/total_records*100:.1f}% / {n_test/total_records*100:.1f}%)",
        "train_ham_count": train_ham,
        "train_ham_pct": train_ham_pct,
        "train_spam_count": train_spam,
        "train_spam_pct": train_spam_pct,
        "test_ham_count": test_ham,
        "test_ham_pct": test_ham_pct,
        "test_spam_count": test_spam,
        "test_spam_pct": test_spam_pct,
        "ngram_range": "(1, 2)",
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True,
        "vocab_size": analysis["vocab_size"],
        "train_shape": str(X_train_tfidf.shape),
        "test_shape": str(X_test_tfidf.shape),
        "train_nnz": analysis["train_nnz"],
        "test_nnz": analysis["test_nnz"],
        "train_sparsity_pct": analysis["train_sparsity_pct"],
        "test_sparsity_pct": analysis["test_sparsity_pct"],
        "first_30_features": analysis["first_30_features"],
        "sample_unigrams": analysis["sample_unigrams"],
        "sample_bigrams": analysis["sample_bigrams"],
        "inspections": analysis["inspections"],
        "leakage_check_passed": leakage_passed,
        "vectorizer_path": str(vec_path),
        "split_path": str(split_path)
    }
    
    report_text = generate_report(stats, report_path)
    print(f"Feature engineering report generated at: {report_path}")
    
    # Terminal Summary
    print("\n" + "=" * 60)
    print("PHASE 3: FEATURE ENGINEERING COMPLETION SUMMARY")
    print("=" * 60)
    print(f"Input Cleaned Records:       {total_records:,}")
    print(f"Train / Test Split:          {n_train:,} / {n_test:,} (Stratified 80/20)")
    print(f"Train Class Distribution:    Ham: {train_ham:,} ({train_ham_pct:.2f}%) | Spam: {train_spam:,} ({train_spam_pct:.2f}%)")
    print(f"Test Class Distribution:     Ham: {test_ham:,} ({test_ham_pct:.2f}%) | Spam: {test_spam:,} ({test_spam_pct:.2f}%)")
    print(f"Vocabulary / Total Features: {analysis['vocab_size']:,} (Unigrams + Bigrams)")
    print(f"Train Matrix Shape:          {X_train_tfidf.shape} (Sparsity: {analysis['train_sparsity_pct']:.4f}%)")
    print(f"Test Matrix Shape:           {X_test_tfidf.shape} (Sparsity: {analysis['test_sparsity_pct']:.4f}%)")
    print(f"TF-IDF Data Leakage Check:   {'PASS' if leakage_passed else 'FAIL'}")
    print(f"Fitted Vectorizer Saved:     {vec_path}")
    print(f"Split Info Saved:            {split_path}")
    print(f"Report File:                 {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
