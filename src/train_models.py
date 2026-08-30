"""
Spam Email Classifier — Model Training Script

Phase 4: Model Training
Trains baseline candidate machine-learning classifiers on the Phase 3 TF-IDF features:
  1. Multinomial Naive Bayes (MultinomialNB)
  2. Linear Support Vector Machine (LinearSVC)

Strict Data-Leakage & Workflow Guardrails:
  - Reuses the EXACT train/test split saved during Phase 3 (train_test_split.npz).
  - Reuses the EXACT fitted TF-IDF vectorizer from Phase 3 (tfidf_vectorizer.joblib).
  - Transforms X_train using the pre-fitted vectorizer (no fitting on test data or full dataset).
  - Trains exclusively on X_train_tfidf and y_train.
  - Test set features are NOT used for training or evaluation in this phase.
  - Evaluation metrics (accuracy, F1, ROC-AUC, etc.) are strictly deferred to Phase 5.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------------------------------------------------------
# Path Resolution
# ----------------------------------------------------------------------
def get_project_root() -> Path:
    """Resolve project root directory relative to this script."""
    return Path(__file__).resolve().parent.parent


def get_cleaned_data_path() -> Path:
    """Resolve path to cleaned dataset from Phase 2."""
    return get_project_root() / "data" / "processed" / "cleaned_internship.csv"


def get_split_path() -> Path:
    """Resolve path to train/test split saved in Phase 3."""
    return get_project_root() / "data" / "processed" / "train_test_split.npz"


def get_vectorizer_path() -> Path:
    """Resolve path to TF-IDF vectorizer saved in Phase 3."""
    return get_project_root() / "models" / "tfidf_vectorizer.joblib"


def get_naive_bayes_path() -> Path:
    """Resolve path to save the trained Multinomial Naive Bayes model."""
    return get_project_root() / "models" / "naive_bayes_model.joblib"


def get_linear_svm_path() -> Path:
    """Resolve path to save the trained Linear SVM model."""
    return get_project_root() / "models" / "linear_svm_model.joblib"


def get_report_path() -> Path:
    """Resolve path to save the Phase 4 training report."""
    return get_project_root() / "reports" / "model_training_report.txt"


# ----------------------------------------------------------------------
# Data & Artifact Loading
# ----------------------------------------------------------------------
def load_cleaned_data(file_path: Path) -> pd.DataFrame:
    """
    Load cleaned dataset from Phase 2 and validate schema.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {file_path}")

    df = pd.read_csv(file_path)

    required_cols = {"text", "spam"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Dataset missing required columns: {required_cols - set(df.columns)}")

    if len(df) != 5695:
        raise ValueError(f"Expected 5,695 records in cleaned dataset, found: {len(df)}")

    df["text"] = df["text"].astype(str)
    return df


def load_split(split_path: Path) -> dict:
    """
    Load stored train/test split indices and targets from Phase 3.
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Train/test split file not found at: {split_path}")

    npz_data = np.load(split_path)
    train_indices = npz_data["train_indices"]
    test_indices = npz_data["test_indices"]
    y_train = npz_data["y_train"]
    y_test = npz_data["y_test"]

    # Sanity checks on split arrays
    if len(train_indices) != 4556:
        raise ValueError(f"Expected 4,556 train indices, found: {len(train_indices)}")
    if len(test_indices) != 1139:
        raise ValueError(f"Expected 1,139 test indices, found: {len(test_indices)}")

    overlap = set(train_indices).intersection(set(test_indices))
    if len(overlap) > 0:
        raise ValueError(f"Data leakage detected! Train and test indices overlap: {len(overlap)} samples")

    total_accounted = len(set(train_indices).union(set(test_indices)))
    if total_accounted != 5695:
        raise ValueError(f"Total accounted indices {total_accounted} != expected 5,695")

    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "y_train": y_train,
        "y_test": y_test
    }


def load_vectorizer(vectorizer_path: Path) -> TfidfVectorizer:
    """
    Load the saved Phase 3 TF-IDF vectorizer.
    """
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer artifact not found at: {vectorizer_path}")

    vectorizer = joblib.load(vectorizer_path)
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError(f"Loaded object is not TfidfVectorizer, got: {type(vectorizer)}")

    n_features = len(vectorizer.get_feature_names_out())
    if n_features != 121288:
        raise ValueError(f"Expected 121,288 TF-IDF vocabulary features, found: {n_features}")

    return vectorizer


# ----------------------------------------------------------------------
# Feature Preparation & Sanity Checks
# ----------------------------------------------------------------------
def prepare_training_features(
    df: pd.DataFrame,
    split_data: dict,
    vectorizer: TfidfVectorizer
) -> tuple:
    """
    Reconstruct exact X_train, transform with the pre-fitted vectorizer,
    and validate dimensions and label constraints.
    """
    train_indices = split_data["train_indices"]
    test_indices = split_data["test_indices"]
    y_train = split_data["y_train"]

    # Validate target labels
    unique_labels = set(np.unique(y_train))
    if unique_labels != {0, 1}:
        raise ValueError(f"y_train contains unexpected labels: {unique_labels}. Expected {{0, 1}}.")

    # Extract X_train text series
    X_train_text = df["text"].iloc[train_indices].reset_index(drop=True)
    X_test_text = df["text"].iloc[test_indices].reset_index(drop=True)

    # Verification: check alignment with target labels
    if len(X_train_text) != 4556:
        raise ValueError(f"X_train length mismatch: {len(X_train_text)} != 4556")
    if len(X_test_text) != 1139:
        raise ValueError(f"X_test length mismatch: {len(X_test_text)} != 1139")

    # Transform training features strictly using pre-fitted vectorizer
    # NOTE: We do NOT call fit() or fit_transform(). Only transform().
    X_train_tfidf = vectorizer.transform(X_train_text)

    # Validate feature matrix dimensions
    n_rows, n_cols = X_train_tfidf.shape
    if n_rows != 4556:
        raise ValueError(f"X_train_tfidf rows mismatch: {n_rows} != 4556")
    if n_cols != 121288:
        raise ValueError(f"X_train_tfidf features mismatch: {n_cols} != 121,288")

    return X_train_tfidf, y_train


# ----------------------------------------------------------------------
# Model Training
# ----------------------------------------------------------------------
def train_naive_bayes(
    X_train_tfidf,
    y_train: np.ndarray,
    alpha: float = 1.0
) -> tuple:
    """
    Train a baseline Multinomial Naive Bayes classifier on training features.
    
    Returns:
        tuple: (fitted_model, training_metadata_dict)
    """
    print("Training Model 1: Multinomial Naive Bayes (alpha=1.0)...")
    start_time = time.perf_counter()

    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_tfidf, y_train)

    elapsed_time_sec = time.perf_counter() - start_time
    print(f"  -> Training completed in {elapsed_time_sec * 1000:.2f} ms")

    metadata = {
        "model_name": "Multinomial Naive Bayes",
        "parameters": {"alpha": alpha},
        "training_samples": X_train_tfidf.shape[0],
        "n_features": X_train_tfidf.shape[1],
        "training_time_sec": elapsed_time_sec,
        "status": "SUCCESS"
    }

    return model, metadata


def train_linear_svm(
    X_train_tfidf,
    y_train: np.ndarray,
    C: float = 1.0,
    random_state: int = 42
) -> tuple:
    """
    Train a baseline Linear Support Vector Machine classifier on training features.
    
    Returns:
        tuple: (fitted_model, training_metadata_dict)
    """
    print("Training Model 2: Linear Support Vector Machine (C=1.0, random_state=42)...")
    start_time = time.perf_counter()

    model = LinearSVC(C=C, random_state=random_state)
    model.fit(X_train_tfidf, y_train)

    elapsed_time_sec = time.perf_counter() - start_time
    print(f"  -> Training completed in {elapsed_time_sec * 1000:.2f} ms")

    metadata = {
        "model_name": "Linear Support Vector Machine",
        "parameters": {"C": C, "random_state": random_state},
        "training_samples": X_train_tfidf.shape[0],
        "n_features": X_train_tfidf.shape[1],
        "training_time_sec": elapsed_time_sec,
        "status": "SUCCESS"
    }

    return model, metadata


# ----------------------------------------------------------------------
# Model Serialization & Reload Verification
# ----------------------------------------------------------------------
def save_models(
    nb_model: MultinomialNB,
    svm_model: LinearSVC,
    nb_path: Path,
    svm_path: Path
) -> None:
    """
    Save trained model artifacts using joblib.
    """
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    svm_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving Multinomial Naive Bayes model to: {nb_path}")
    joblib.dump(nb_model, nb_path)

    print(f"Saving Linear SVM model to: {svm_path}")
    joblib.dump(svm_model, svm_path)


def verify_saved_models(
    nb_path: Path,
    svm_path: Path,
    vectorizer_path: Path
) -> dict:
    """
    Perform artifact integrity checks by reloading each saved model and vectorizer.
    Verifies object type and structural integrity without calculating evaluation metrics.
    """
    print("\nVerifying saved model artifacts...")

    # Verify Naive Bayes
    if not nb_path.exists():
        raise FileNotFoundError(f"Naive Bayes artifact missing at: {nb_path}")
    nb_loaded = joblib.load(nb_path)
    nb_pass = isinstance(nb_loaded, MultinomialNB) and hasattr(nb_loaded, "classes_")
    print(f"  Multinomial Naive Bayes reload: {'PASS' if nb_pass else 'FAIL'}")

    # Verify Linear SVM
    if not svm_path.exists():
        raise FileNotFoundError(f"Linear SVM artifact missing at: {svm_path}")
    svm_loaded = joblib.load(svm_path)
    svm_pass = isinstance(svm_loaded, LinearSVC) and hasattr(svm_loaded, "classes_")
    print(f"  Linear SVM reload: {'PASS' if svm_pass else 'FAIL'}")

    # Verify TF-IDF Vectorizer
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer missing at: {vectorizer_path}")
    vec_loaded = joblib.load(vectorizer_path)
    vec_pass = isinstance(vec_loaded, TfidfVectorizer) and hasattr(vec_loaded, "vocabulary_")
    print(f"  TF-IDF vectorizer reload: {'PASS' if vec_pass else 'FAIL'}")

    return {
        "nb_reload": "PASS" if nb_pass else "FAIL",
        "svm_reload": "PASS" if svm_pass else "FAIL",
        "vec_reload": "PASS" if vec_pass else "FAIL"
    }


# ----------------------------------------------------------------------
# Report Generation
# ----------------------------------------------------------------------
def generate_training_report(
    stats: dict,
    report_path: Path
) -> str:
    """
    Generate and save the formal Phase 4 Model Training Report.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_content = f"""==================================================
SPAM EMAIL CLASSIFIER
PHASE 4 — MODEL TRAINING REPORT
==================================================

DATASET
--------
Input:
data/processed/cleaned_internship.csv

Records:
{stats['total_records']:,}

Training records:
{stats['train_records']:,}

Features:
{stats['n_features']:,}

TRAINING SPLIT
--------------
Train indices:
{stats['train_records']:,}

Test indices:
{stats['test_records']:,}

Split:
80/20 stratified

Random state:
42

MODEL 1
--------
Model:
Multinomial Naive Bayes

Parameters:
alpha = 1.0

Training status:
{stats['nb_status']}

Saved artifact:
models/naive_bayes_model.joblib

MODEL 2
--------
Model:
Linear Support Vector Machine

Parameters:
C = 1.0
random_state = 42

Training status:
{stats['svm_status']}

Saved artifact:
models/linear_svm_model.joblib

ARTIFACT VERIFICATION
---------------------
Naive Bayes reload:
{stats['nb_reload']}

Linear SVM reload:
{stats['svm_reload']}

TF-IDF vectorizer:
{stats['vec_reload']}

DATA LEAKAGE
------------
Confirm:

- TF-IDF was fitted only during Phase 3 on training data.
- Phase 4 reused the fitted vectorizer.
- No test samples were used during model fitting.
- No new train/test split was created.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nTraining report generated at: {report_path}")
    return report_content


# ----------------------------------------------------------------------
# Main Orchestration
# ----------------------------------------------------------------------
def main() -> None:
    """
    Phase 4 Pipeline Orchestration.
    """
    print("=" * 60)
    print("SPAM EMAIL CLASSIFIER — PHASE 4: MODEL TRAINING")
    print("=" * 60)

    # 1. Path resolution
    cleaned_data_path = get_cleaned_data_path()
    split_path = get_split_path()
    vectorizer_path = get_vectorizer_path()
    nb_model_path = get_naive_bayes_path()
    svm_model_path = get_linear_svm_path()
    report_path = get_report_path()

    print("\n--- STEP 1: LOAD CLEANED DATA & REUSE PHASE 3 ARTIFACTS ---")
    df = load_cleaned_data(cleaned_data_path)
    print(f"Loaded cleaned dataset: {len(df):,} records")

    split_data = load_split(split_path)
    print(f"Loaded train/test split indices: {len(split_data['train_indices']):,} train, {len(split_data['test_indices']):,} test")

    vectorizer = load_vectorizer(vectorizer_path)
    vocab_size = len(vectorizer.get_feature_names_out())
    print(f"Loaded TF-IDF vectorizer: {vocab_size:,} vocabulary features")

    print("\n--- STEP 2: PREPARE TRAINING FEATURES & SANITY CHECKS ---")
    X_train_tfidf, y_train = prepare_training_features(df, split_data, vectorizer)
    print(f"Training feature matrix shape: {X_train_tfidf.shape[0]:,} samples x {X_train_tfidf.shape[1]:,} features")
    print(f"Training label distribution: Ham (0) = {np.sum(y_train == 0):,}, Spam (1) = {np.sum(y_train == 1):,}")

    print("\n--- STEP 3: TRAIN CANDIDATE MODELS ---")
    nb_model, nb_meta = train_naive_bayes(X_train_tfidf, y_train, alpha=1.0)
    svm_model, svm_meta = train_linear_svm(X_train_tfidf, y_train, C=1.0, random_state=42)

    print("\n--- STEP 4: SAVE TRAINED MODELS ---")
    save_models(nb_model, svm_model, nb_model_path, svm_model_path)

    print("\n--- STEP 5: VERIFY SAVED ARTIFACTS (RELOAD TEST) ---")
    verification_results = verify_saved_models(nb_model_path, svm_model_path, vectorizer_path)

    print("\n--- STEP 6: GENERATE PHASE 4 TRAINING REPORT ---")
    stats = {
        "total_records": len(df),
        "train_records": len(split_data["train_indices"]),
        "test_records": len(split_data["test_indices"]),
        "n_features": vocab_size,
        "nb_status": nb_meta["status"],
        "svm_status": svm_meta["status"],
        "nb_reload": verification_results["nb_reload"],
        "svm_reload": verification_results["svm_reload"],
        "vec_reload": verification_results["vec_reload"]
    }

    report_text = generate_training_report(stats, report_path)
    print("\n" + report_text)
    print("=" * 60)
    print("PHASE 4 — MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
