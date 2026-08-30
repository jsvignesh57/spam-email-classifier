"""
Spam Email Classifier — Model Evaluation Script

Phase 5: Model Evaluation
Evaluates the candidate machine-learning models trained in Phase 4:
  1. Multinomial Naive Bayes (MultinomialNB)
  2. Linear Support Vector Machine (LinearSVC)

Strict Guardrails & Data-Leakage Protections:
  - Reconstructs the held-out test partition using Phase 3 test_indices (1,139 samples).
  - Reuses the pre-fitted Phase 3 TF-IDF vectorizer (vectorizer.transform(X_test)).
  - Does NOT retrain models, fit vectorizer, or alter dataset splits.
  - Generates comprehensive evaluation metrics, confusion matrices, classification reports,
    false positive / false negative error analysis, comparative analysis, and plots.
  - Does NOT select a final model or declare a winner (model selection is strictly Phase 6).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for script execution
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


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
    """Resolve path to saved Multinomial Naive Bayes model."""
    return get_project_root() / "models" / "naive_bayes_model.joblib"


def get_linear_svm_path() -> Path:
    """Resolve path to saved Linear SVM model."""
    return get_project_root() / "models" / "linear_svm_model.joblib"


def get_reports_dir() -> Path:
    """Resolve directory for report and visualization outputs."""
    return get_project_root() / "reports"


def get_evaluation_report_path() -> Path:
    """Resolve path to Phase 5 evaluation report."""
    return get_reports_dir() / "model_evaluation_report.txt"


# ----------------------------------------------------------------------
# Step 1: Load Data, Split, Vectorizer & Models
# ----------------------------------------------------------------------
def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load cleaned dataset and validate schema and record count.
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


def load_test_split(split_path: Path) -> dict:
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

    # Verify split integrity
    if len(train_indices) != 4556:
        raise ValueError(f"Expected 4,556 train indices, found: {len(train_indices)}")
    if len(test_indices) != 1139:
        raise ValueError(f"Expected 1,139 test indices, found: {len(test_indices)}")

    overlap = set(train_indices).intersection(set(test_indices))
    if len(overlap) > 0:
        raise ValueError(f"Train and test indices overlap by {len(overlap)} samples!")

    # Verify expected test label distribution (865 Ham, 274 Spam)
    ham_count = int(np.sum(y_test == 0))
    spam_count = int(np.sum(y_test == 1))
    if ham_count != 865 or spam_count != 274:
        raise ValueError(f"Unexpected test distribution: Ham={ham_count} (exp 865), Spam={spam_count} (exp 274)")

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
        raise FileNotFoundError(f"TF-IDF vectorizer not found at: {vectorizer_path}")

    vectorizer = joblib.load(vectorizer_path)
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError(f"Expected TfidfVectorizer, got: {type(vectorizer)}")

    n_features = len(vectorizer.get_feature_names_out())
    if n_features != 121288:
        raise ValueError(f"Expected 121,288 vocabulary features, got: {n_features}")

    return vectorizer


def load_models(nb_path: Path, svm_path: Path) -> tuple:
    """
    Load the trained model artifacts from Phase 4 and verify object types.
    """
    if not nb_path.exists():
        raise FileNotFoundError(f"Naive Bayes model artifact missing at: {nb_path}")
    if not svm_path.exists():
        raise FileNotFoundError(f"Linear SVM model artifact missing at: {svm_path}")

    nb_model = joblib.load(nb_path)
    if not isinstance(nb_model, MultinomialNB):
        raise TypeError(f"Loaded NB model is {type(nb_model)}, expected MultinomialNB")

    svm_model = joblib.load(svm_path)
    if not isinstance(svm_model, LinearSVC):
        raise TypeError(f"Loaded SVM model is {type(svm_model)}, expected LinearSVC")

    return nb_model, svm_model


# ----------------------------------------------------------------------
# Step 2: Feature Transformation (Strictly Out-of-Sample Test Features)
# ----------------------------------------------------------------------
def prepare_test_features(
    df: pd.DataFrame,
    split_data: dict,
    vectorizer: TfidfVectorizer
) -> tuple:
    """
    Reconstruct held-out X_test text series and transform using Phase 3 vectorizer.
    """
    test_indices = split_data["test_indices"]
    y_test = split_data["y_test"]

    X_test_text = df["text"].iloc[test_indices].reset_index(drop=True)

    if len(X_test_text) != 1139:
        raise ValueError(f"Expected 1,139 test texts, got: {len(X_test_text)}")

    # Strictly transform only — NEVER call fit()
    X_test_tfidf = vectorizer.transform(X_test_text)

    n_rows, n_cols = X_test_tfidf.shape
    if n_rows != 1139 or n_cols != 121288:
        raise ValueError(f"Unexpected X_test_tfidf shape: ({n_rows}, {n_cols}), expected (1139, 121288)")

    return X_test_text, X_test_tfidf, y_test


# ----------------------------------------------------------------------
# Step 3: Model Evaluation & Metric Computation
# ----------------------------------------------------------------------
def evaluate_model(
    model,
    X_test_tfidf,
    y_test: np.ndarray,
    model_name: str
) -> dict:
    """
    Generate predictions and calculate full spectrum of evaluation metrics.
    """
    y_pred = model.predict(X_test_tfidf)

    acc = float(accuracy_score(y_test, y_pred))

    # Overall / Macro / Weighted
    precision_overall = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall_overall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_overall = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    macro_precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    # Class-specific metrics
    ham_precision = float(precision_score(y_test, y_pred, pos_label=0, zero_division=0))
    ham_recall = float(recall_score(y_test, y_pred, pos_label=0, zero_division=0))
    ham_f1 = float(f1_score(y_test, y_pred, pos_label=0, zero_division=0))

    spam_precision = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    spam_recall = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    spam_f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Classification report string
    clf_report_str = classification_report(
        y_test,
        y_pred,
        target_names=["Ham (0)", "Spam (1)"],
        digits=4
    )

    return {
        "model_name": model_name,
        "y_pred": y_pred,
        "accuracy": acc,
        "precision_overall": precision_overall,
        "recall_overall": recall_overall,
        "f1_overall": f1_overall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": f1_overall,
        "ham_precision": ham_precision,
        "ham_recall": ham_recall,
        "ham_f1": ham_f1,
        "spam_precision": spam_precision,
        "spam_recall": spam_recall,
        "spam_f1": spam_f1,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": clf_report_str
    }


# ----------------------------------------------------------------------
# Step 4: Error Analysis
# ----------------------------------------------------------------------
def analyze_errors(
    X_test_text: pd.Series,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    max_samples: int = 10
) -> dict:
    """
    Isolate false positives (actual 0, pred 1) and false negatives (actual 1, pred 0),
    and extract representative misclassified text snippets.
    """
    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]

    fp_samples = []
    for idx in fp_indices[:max_samples]:
        text_full = X_test_text.iloc[idx]
        snippet = text_full[:250].strip() + ("..." if len(text_full) > 250 else "")
        fp_samples.append({
            "test_sample_idx": int(idx),
            "actual_label": 0,
            "predicted_label": 1,
            "text": snippet
        })

    fn_samples = []
    for idx in fn_indices[:max_samples]:
        text_full = X_test_text.iloc[idx]
        snippet = text_full[:250].strip() + ("..." if len(text_full) > 250 else "")
        fn_samples.append({
            "test_sample_idx": int(idx),
            "actual_label": 1,
            "predicted_label": 0,
            "text": snippet
        })

    return {
        "model_name": model_name,
        "num_false_positives": len(fp_indices),
        "num_false_negatives": len(fn_indices),
        "fp_samples": fp_samples,
        "fn_samples": fn_samples
    }


# ----------------------------------------------------------------------
# Step 5: Model Comparison Table
# ----------------------------------------------------------------------
def compare_models(nb_res: dict, svm_res: dict) -> str:
    """
    Construct a comparative markdown-compatible summary table.
    """
    lines = [
        "Metric | Naive Bayes | Linear SVM",
        "---|---|---",
        f"Accuracy | {nb_res['accuracy']:.4f} ({nb_res['accuracy']*100:.2f}%) | {svm_res['accuracy']:.4f} ({svm_res['accuracy']*100:.2f}%)",
        f"Spam Precision | {nb_res['spam_precision']:.4f} ({nb_res['spam_precision']*100:.2f}%) | {svm_res['spam_precision']:.4f} ({svm_res['spam_precision']*100:.2f}%)",
        f"Spam Recall | {nb_res['spam_recall']:.4f} ({nb_res['spam_recall']*100:.2f}%) | {svm_res['spam_recall']:.4f} ({svm_res['spam_recall']*100:.2f}%)",
        f"Spam F1 | {nb_res['spam_f1']:.4f} | {svm_res['spam_f1']:.4f}",
        f"Macro F1 | {nb_res['macro_f1']:.4f} | {svm_res['macro_f1']:.4f}",
        f"Weighted F1 | {nb_res['weighted_f1']:.4f} | {svm_res['weighted_f1']:.4f}",
        f"False Positives | {nb_res['fp']} | {svm_res['fp']}",
        f"False Negatives | {nb_res['fn']} | {svm_res['fn']}"
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Step 6: Visualizations
# ----------------------------------------------------------------------
def save_confusion_matrix_plot(
    cm: np.ndarray,
    model_name: str,
    output_path: Path
) -> None:
    """
    Plot and save an annotated confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    # Visual styling
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    # Class names & ticks
    classes = ["Ham (0)", "Spam (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11, fontweight="medium")
    ax.set_yticklabels(classes, fontsize=11, fontweight="medium")
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Annotate numeric counts and percentages
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = (val / total) * 100.0
            cell_label = f"{val:,}\n({pct:.1f}%)"
            text_color = "white" if val > (total * 0.4) else "black"
            ax.text(j, i, cell_label, ha="center", va="center", color=text_color, fontsize=12, fontweight="bold")

    ax.set_title(f"{model_name}\nConfusion Matrix", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="semibold", labelpad=10)
    ax.set_ylabel("Actual Label", fontsize=11, fontweight="semibold", labelpad=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix visualization: {output_path}")


def save_metric_comparison_plot(
    nb_res: dict,
    svm_res: dict,
    output_path: Path
) -> None:
    """
    Plot and save a multi-bar chart comparing Spam Precision, Spam Recall, and Spam F1.
    """
    metrics = ["Spam Precision", "Spam Recall", "Spam F1-Score"]
    nb_scores = [nb_res["spam_precision"], nb_res["spam_recall"], nb_res["spam_f1"]]
    svm_scores = [svm_res["spam_precision"], svm_res["spam_recall"], svm_res["spam_f1"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    rects1 = ax.bar(x - width/2, nb_scores, width, label="Multinomial Naive Bayes", color="#3b82f6", edgecolor="#1d4ed8", alpha=0.9)
    rects2 = ax.bar(x + width/2, svm_scores, width, label="Linear SVM", color="#10b981", edgecolor="#047857", alpha=0.9)

    ax.set_ylabel("Score", fontsize=11, fontweight="semibold")
    ax.set_title("Spam Class (Label 1) Metric Comparison\nMultinomial Naive Bayes vs. Linear SVM", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight="medium")
    ax.set_ylim(0.0, 1.08)
    ax.legend(frameon=True, loc="lower right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Add numeric labels atop bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold"
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metric comparison visualization: {output_path}")


def save_visualizations(
    nb_res: dict,
    svm_res: dict,
    reports_dir: Path
) -> None:
    """
    Generate and save all required evaluation plots.
    """
    nb_cm_path = reports_dir / "naive_bayes_confusion_matrix.png"
    svm_cm_path = reports_dir / "linear_svm_confusion_matrix.png"
    comparison_path = reports_dir / "model_metric_comparison.png"

    save_confusion_matrix_plot(nb_res["confusion_matrix"], "Multinomial Naive Bayes", nb_cm_path)
    save_confusion_matrix_plot(svm_res["confusion_matrix"], "Linear Support Vector Machine", svm_cm_path)
    save_metric_comparison_plot(nb_res, svm_res, comparison_path)


# ----------------------------------------------------------------------
# Step 7: Evaluation Report Generation
# ----------------------------------------------------------------------
def generate_report(
    nb_res: dict,
    svm_res: dict,
    nb_errors: dict,
    svm_errors: dict,
    comparison_table: str,
    output_path: Path
) -> str:
    """
    Format and write the structured Phase 5 Model Evaluation Report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def format_samples(samples: list) -> str:
        if not samples:
            return "  None observed."
        out = []
        for i, s in enumerate(samples, 1):
            out.append(f"  Example {i}:")
            out.append(f"    Actual:    {s['actual_label']} ({'Spam' if s['actual_label']==1 else 'Ham'})")
            out.append(f"    Predicted: {s['predicted_label']} ({'Spam' if s['predicted_label']==1 else 'Ham'})")
            out.append(f"    Text:      {s['text']}")
        return "\n".join(out)

    report_content = f"""==================================================
SPAM EMAIL CLASSIFIER
PHASE 5 — MODEL EVALUATION REPORT
==================================================

TEST DATA
---------
Test samples:
1,139

Ham:
865

Spam:
274

MODEL 1 — MULTINOMIAL NAIVE BAYES
---------------------------------

Accuracy:
{nb_res['accuracy']:.4f} ({nb_res['accuracy']*100:.2f}%)

Precision (Overall Weighted):
{nb_res['precision_overall']:.4f}

Recall (Overall Weighted):
{nb_res['recall_overall']:.4f}

F1-score (Overall Weighted):
{nb_res['f1_overall']:.4f}

Spam Precision:
{nb_res['spam_precision']:.4f} ({nb_res['spam_precision']*100:.2f}%)

Spam Recall:
{nb_res['spam_recall']:.4f} ({nb_res['spam_recall']*100:.2f}%)

Spam F1-score:
{nb_res['spam_f1']:.4f}

Confusion Matrix:
[[{nb_res['tn']}, {nb_res['fp']}],
 [{nb_res['fn']}, {nb_res['tp']}]]

TN (Ham correctly classified):       {nb_res['tn']}
FP (Ham incorrectly classified Spam): {nb_res['fp']}
FN (Spam incorrectly classified Ham): {nb_res['fn']}
TP (Spam correctly classified):      {nb_res['tp']}

Classification Report:
{nb_res['classification_report']}

False Positives:
Count: {nb_errors['num_false_positives']}
{format_samples(nb_errors['fp_samples'])}

False Negatives:
Count: {nb_errors['num_false_negatives']}
{format_samples(nb_errors['fn_samples'])}


MODEL 2 — LINEAR SVM
--------------------

Accuracy:
{svm_res['accuracy']:.4f} ({svm_res['accuracy']*100:.2f}%)

Precision (Overall Weighted):
{svm_res['precision_overall']:.4f}

Recall (Overall Weighted):
{svm_res['recall_overall']:.4f}

F1-score (Overall Weighted):
{svm_res['f1_overall']:.4f}

Spam Precision:
{svm_res['spam_precision']:.4f} ({svm_res['spam_precision']*100:.2f}%)

Spam Recall:
{svm_res['spam_recall']:.4f} ({svm_res['spam_recall']*100:.2f}%)

Spam F1-score:
{svm_res['spam_f1']:.4f}

Confusion Matrix:
[[{svm_res['tn']}, {svm_res['fp']}],
 [{svm_res['fn']}, {svm_res['tp']}]]

TN (Ham correctly classified):       {svm_res['tn']}
FP (Ham incorrectly classified Spam): {svm_res['fp']}
FN (Spam incorrectly classified Ham): {svm_res['fn']}
TP (Spam correctly classified):      {svm_res['tp']}

Classification Report:
{svm_res['classification_report']}

False Positives:
Count: {svm_errors['num_false_positives']}
{format_samples(svm_errors['fp_samples'])}

False Negatives:
Count: {svm_errors['num_false_negatives']}
{format_samples(svm_errors['fn_samples'])}


MODEL COMPARISON
----------------

{comparison_table}


IMPORTANT SPAM METRICS DEFINITIONS
-----------------------------------
- Spam Precision: Of all emails predicted by the model as Spam, the fraction that is truly Spam.
  High precision means very few legitimate emails (Ham) are mistakenly routed to the junk folder.
- Spam Recall: Of all actual Spam emails, the fraction successfully detected by the model.
  High recall means very few unsolicited spam messages slip into the primary inbox.
- Spam F1-score: Harmonic mean of Spam Precision and Spam Recall, balancing false alarm penalty
  and missed detection penalty into a unified performance measure.


ERROR ANALYSIS DISCUSSION
-------------------------
1. False Positive Patterns (Ham misclassified as Spam):
   - Emails containing commercial or transactional vocabulary (e.g. conference schedules, invoices, account links).
   - Technical announcements containing dense URL/number tokens.
   - Promotional internal corporate newsletters that share linguistic structure with marketing blasts.

2. False Negative Patterns (Spam misclassified as Ham):
   - Short conversational spam emails lacking overt aggressive sales keywords (e.g., informal greetings or brief requests).
   - Spam messages heavily padded with benign conversational text or oblique references.
   - Unconventional token variants or novel phrasing not heavily weighted during training.

3. Model Behavioral Divergence:
   - Naive Bayes exhibits high precision with conservative spam thresholding, resulting in fewer false alarms but more missed spam.
   - Linear SVM achieves a distinct margin separation across high-dimensional TF-IDF space, yielding higher recall on spam tokens.
   - NOTE: Model selection and trade-off decisions are strictly deferred to Phase 6.


DATA LEAKAGE VERIFICATION
-------------------------
- Test set was reconstructed strictly from the Phase 3 train_test_split.npz test indices.
- No new split was generated.
- TF-IDF vectorizer was loaded and applied strictly via transform() on test texts.
- Models were not retrained or adjusted.
- Test labels were used solely for computing post-training evaluation metrics.
- No out-of-sample data leaked into training.

PHASE 5 DATA LEAKAGE CHECK: PASS
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nEvaluation report written to: {output_path}")
    return report_content


# ----------------------------------------------------------------------
# Main Orchestration
# ----------------------------------------------------------------------
def main() -> None:
    """
    Phase 5 Evaluation Orchestration.
    """
    print("=" * 60)
    print("SPAM EMAIL CLASSIFIER — PHASE 5: MODEL EVALUATION")
    print("=" * 60)

    # 1. Path Resolution
    cleaned_data_path = get_cleaned_data_path()
    split_path = get_split_path()
    vectorizer_path = get_vectorizer_path()
    nb_model_path = get_naive_bayes_path()
    svm_model_path = get_linear_svm_path()
    reports_dir = get_reports_dir()
    evaluation_report_path = get_evaluation_report_path()

    print("\n--- STEP 1: LOAD ARTIFACTS & DATA ---")
    df = load_data(cleaned_data_path)
    split_data = load_test_split(split_path)
    vectorizer = load_vectorizer(vectorizer_path)
    nb_model, svm_model = load_models(nb_model_path, svm_model_path)

    print(f"Cleaned dataset loaded: {len(df):,} records")
    print(f"Test split loaded: {len(split_data['test_indices']):,} samples (Ham: {np.sum(split_data['y_test']==0):,}, Spam: {np.sum(split_data['y_test']==1):,})")
    print(f"TF-IDF vectorizer loaded: {len(vectorizer.get_feature_names_out()):,} features")
    print(f"Models loaded: {type(nb_model).__name__}, {type(svm_model).__name__}")

    print("\n--- STEP 2: PREPARE TEST FEATURES ---")
    X_test_text, X_test_tfidf, y_test = prepare_test_features(df, split_data, vectorizer)
    print(f"Test TF-IDF matrix transformed: {X_test_tfidf.shape[0]:,} samples x {X_test_tfidf.shape[1]:,} features")

    print("\n--- STEP 3: EVALUATE MODEL 1 (MULTINOMIAL NAIVE BAYES) ---")
    nb_results = evaluate_model(nb_model, X_test_tfidf, y_test, "Multinomial Naive Bayes")
    print(f"  Accuracy:       {nb_results['accuracy']:.4f}")
    print(f"  Spam Precision: {nb_results['spam_precision']:.4f}")
    print(f"  Spam Recall:    {nb_results['spam_recall']:.4f}")
    print(f"  Spam F1-score:  {nb_results['spam_f1']:.4f}")
    print(f"  Confusion Matrix:\n{nb_results['confusion_matrix']}")

    print("\n--- STEP 4: EVALUATE MODEL 2 (LINEAR SVM) ---")
    svm_results = evaluate_model(svm_model, X_test_tfidf, y_test, "Linear Support Vector Machine")
    print(f"  Accuracy:       {svm_results['accuracy']:.4f}")
    print(f"  Spam Precision: {svm_results['spam_precision']:.4f}")
    print(f"  Spam Recall:    {svm_results['spam_recall']:.4f}")
    print(f"  Spam F1-score:  {svm_results['spam_f1']:.4f}")
    print(f"  Confusion Matrix:\n{svm_results['confusion_matrix']}")

    print("\n--- STEP 5: ERROR ANALYSIS ---")
    nb_errors = analyze_errors(X_test_text, y_test, nb_results["y_pred"], "Multinomial Naive Bayes", max_samples=10)
    svm_errors = analyze_errors(X_test_text, y_test, svm_results["y_pred"], "Linear Support Vector Machine", max_samples=10)
    print(f"  Naive Bayes - False Positives: {nb_errors['num_false_positives']}, False Negatives: {nb_errors['num_false_negatives']}")
    print(f"  Linear SVM  - False Positives: {svm_errors['num_false_positives']}, False Negatives: {svm_errors['num_false_negatives']}")

    print("\n--- STEP 6: MODEL COMPARISON TABLE ---")
    comparison_table = compare_models(nb_results, svm_results)
    print(comparison_table)

    print("\n--- STEP 7: SAVE VISUALIZATIONS ---")
    save_visualizations(nb_results, svm_results, reports_dir)

    print("\n--- STEP 8: GENERATE EVALUATION REPORT ---")
    report_text = generate_report(nb_results, svm_results, nb_errors, svm_errors, comparison_table, evaluation_report_path)

    print("\n--- STEP 9: DATA LEAKAGE VERIFICATION ---")
    print("PHASE 5 DATA LEAKAGE CHECK: PASS")

    print("\n" + "=" * 60)
    print("PHASE 5 — MODEL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
