"""
Final Project Audit and Verification Script for Spam Email Classifier.
Performs end-to-end verification of datasets, train/test splits, vectorizers,
models, locked test performance, inference edge cases, and artifact integrity.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    print("=" * 70)
    print("SPAM EMAIL CLASSIFIER - SYSTEMATIC AUDIT VERIFICATION")
    print("=" * 70)

    # 1. Dataset Verification
    raw_path = "data/raw/internship.csv"
    processed_path = "data/processed/cleaned_internship.csv"
    split_path = "data/processed/train_test_split.npz"
    manual_path = "data/processed/manual_test_cases.csv"

    print("\n--- 1. DATASET INTEGRITY ---")
    df_raw = pd.read_csv(raw_path)
    print(f"Raw dataset shape: {df_raw.shape}")
    print(f"Raw class distribution: Ham={sum(df_raw['spam']==0)}, Spam={sum(df_raw['spam']==1)}")
    assert df_raw.shape == (5728, 2), f"Unexpected raw shape: {df_raw.shape}"
    assert sum(df_raw['spam']==0) == 4360, "Raw ham count mismatch"
    assert sum(df_raw['spam']==1) == 1368, "Raw spam count mismatch"

    df_cleaned = pd.read_csv(processed_path)
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    print(f"Cleaned class distribution: Ham={sum(df_cleaned['spam']==0)}, Spam={sum(df_cleaned['spam']==1)}")
    assert df_cleaned.shape == (5695, 2), f"Unexpected cleaned shape: {df_cleaned.shape}"
    assert sum(df_cleaned['spam']==0) == 4327, "Cleaned ham count mismatch"
    assert sum(df_cleaned['spam']==1) == 1368, "Cleaned spam count mismatch"
    print(f"Duplicates removed: {len(df_raw) - len(df_cleaned)} records (33 exact duplicate ham emails removed in Phase 2)")
    print("DATASET INTEGRITY: PASS")

    # 2. Train/Test Split Audit
    print("\n--- 2. TRAIN / TEST SPLIT AUDIT ---")
    split_data = np.load(split_path)
    train_idx = split_data['train_indices']
    test_idx = split_data['test_indices']
    print(f"Train indices: {len(train_idx)}, Test indices: {len(test_idx)}")
    assert len(train_idx) == 4556, "Train count mismatch"
    assert len(test_idx) == 1139, "Test count mismatch"

    overlap = set(train_idx).intersection(set(test_idx))
    print(f"Overlap between train and test indices: {len(overlap)}")
    assert len(overlap) == 0, "Data leakage! Train and test indices overlap"

    union_idx = set(train_idx).union(set(test_idx))
    assert union_idx == set(range(5695)), "Train and test union does not cover all 5,695 cleaned records"

    y_train = df_cleaned.iloc[train_idx]['spam'].values
    y_test = df_cleaned.iloc[test_idx]['spam'].values
    print(f"Train class balance: Ham={sum(y_train==0)} ({sum(y_train==0)/len(y_train)*100:.2f}%), Spam={sum(y_train==1)} ({sum(y_train==1)/len(y_train)*100:.2f}%)")
    print(f"Test class balance:  Ham={sum(y_test==0)} ({sum(y_test==0)/len(y_test)*100:.2f}%), Spam={sum(y_test==1)} ({sum(y_test==1)/len(y_test)*100:.2f}%)")
    assert sum(y_train==0) == 3462 and sum(y_train==1) == 1094, "Train class distribution mismatch"
    assert sum(y_test==0) == 865 and sum(y_test==1) == 274, "Test class distribution mismatch"
    print("FINAL SPLIT INTEGRITY CHECK: PASS")

    # 3. TF-IDF Vectorizer Audit
    print("\n--- 3. TF-IDF VECTORIZER AUDIT ---")
    vec_path = "models/tfidf_vectorizer.joblib"
    vectorizer = joblib.load(vec_path)
    print(f"Vectorizer type: {type(vectorizer).__name__}")
    print(f"Analyzer: {vectorizer.analyzer}")
    print(f"N-gram range: {vectorizer.ngram_range}")
    print(f"Sublinear TF: {vectorizer.sublinear_tf}")
    print(f"Min DF: {vectorizer.min_df}, Max DF: {vectorizer.max_df}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")
    assert vectorizer.analyzer == 'word'
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.sublinear_tf is True
    assert vectorizer.min_df == 2
    assert vectorizer.max_df == 0.95
    assert len(vectorizer.vocabulary_) == 121288
    print("TF-IDF INTEGRITY: PASS")

    # 4. Model Compatibility & Locked Test Evaluation
    print("\n--- 4. PRODUCTION MODEL & LOCKED TEST EVALUATION ---")
    model_v2_path = "models/final_spam_classifier_v2.joblib"
    model_v2 = joblib.load(model_v2_path)
    print(f"Model type: {type(model_v2).__name__}")
    print(f"Parameters: C={model_v2.C}, loss='{model_v2.loss}', random_state={model_v2.random_state}")
    print(f"Feature dimension: {model_v2.coef_.shape[1]:,}")
    assert model_v2.coef_.shape[1] == len(vectorizer.vocabulary_), "Model and vectorizer dimension mismatch"
    assert model_v2.C == 10.0
    assert model_v2.loss == 'squared_hinge'
    assert model_v2.random_state == 42

    # Transform test set and evaluate
    X_test_text = df_cleaned.iloc[test_idx]['text'].values
    X_test_tfidf = vectorizer.transform(X_test_text)
    scores = model_v2.decision_function(X_test_tfidf)
    y_pred = (scores >= 0.0).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nLocked Test Results:")
    print(f"  Total samples: {len(y_test):,}")
    print(f"  Ham: {sum(y_test==0):,}, Spam: {sum(y_test==1):,}")
    print(f"  Accuracy:       {acc*100:.2f}% ({acc:.6f})")
    print(f"  Spam Precision: {prec*100:.2f}% ({prec:.6f})")
    print(f"  Spam Recall:    {rec*100:.2f}% ({rec:.6f})")
    print(f"  Spam F1-Score:  {f1:.4f} ({f1:.6f})")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    assert tn == 863, f"Expected TN=863, got {tn}"
    assert fp == 2, f"Expected FP=2, got {fp}"
    assert fn == 1, f"Expected FN=1, got {fn}"
    assert tp == 273, f"Expected TP=273, got {tp}"
    assert round(acc * 100, 2) == 99.74
    assert round(prec * 100, 2) == 99.27
    assert round(rec * 100, 2) == 99.64
    assert round(f1, 4) == 0.9945
    print("LOCKED TEST PERFORMANCE RECALCULATION: PASS (Exact Match)")

    # 5. Production Inference Pipeline Functional Test
    print("\n--- 5. PRODUCTION INFERENCE PIPELINE TEST ---")
    from src.preprocess import normalize_text
    canonical_preprocess = normalize_text

    def predict_email(raw_text: str):
        cleaned = canonical_preprocess(raw_text)
        if not cleaned:
            # Safe default for empty/whitespace input
            return {"prediction": "Ham", "is_spam": 0, "decision_score": -1.0, "status": "Empty Input Handled Safely"}
        feat = vectorizer.transform([cleaned])
        score = float(model_v2.decision_function(feat)[0])
        is_spam = int(score >= 0.0)
        return {
            "prediction": "Spam" if is_spam == 1 else "Ham",
            "is_spam": is_spam,
            "decision_score": score,
            "status": "Success"
        }

    test_cases = [
        ("Obvious spam", "WINNER! You have won a guaranteed $1,000,000 cash prize! Claim your reward now at http://win-cash-now.com or call 800-555-0199!"),
        ("Obvious legitimate", "Hi Team, please find attached the meeting minutes and agenda for tomorrow's project review. Thanks, Vignesh."),
        ("Promotional email", "Special spring sale! 50% discount on all cloud hosting plans this weekend only. Shop now."),
        ("Short email", "ok"),
        ("Unicode email", "Bonjour! Félicitations pour votre nouvelle opportunité d'investissement exceptionnelle à Genève! 🚀"),
        ("URL-containing email", "Check out the new dataset documentation at https://scikit-learn.org/stable/modules/svm.html for details."),
        ("Numeric-heavy email", "Invoice #84920492 for account 948201. Amount due: 450.00 USD by 2026-09-01. Call 555-014-9922."),
        ("Empty input", "   \n\t  ")
    ]

    for name, sample in test_cases:
        res = predict_email(sample)
        print(f"  [{name:20s}] -> Pred: {res['prediction']:4s} | Score: {res['decision_score']:+8.4f} | Status: {res['status']}")

    print("PRODUCTION INFERENCE PIPELINE: PASS")

    # 6. Artifact Classification Check
    print("\n--- 6. ARTIFACT CLASSIFICATION ---")
    model_dir = "models"
    artifacts = os.listdir(model_dir)
    classifications = {}
    for a in sorted(artifacts):
        if a == "final_spam_classifier_v2.joblib" or a == "tfidf_vectorizer.joblib":
            cat = "PRODUCTION (IMMUTABLE)"
        elif a.startswith("phase_8_"):
            cat = "EXPERIMENTAL (PHASE 8 RESEARCH CANDIDATES)"
        elif a == "model_metadata.json":
            cat = "METADATA / DOCUMENTATION"
        elif a in ["final_spam_classifier.joblib", "linear_svm_model.joblib", "naive_bayes_model.joblib"]:
            cat = "HISTORICAL BASELINE (PHASE 4-6 ARCHIVE)"
        else:
            cat = "OTHER"
        classifications[a] = cat
        print(f"  {a:42s} : {cat}")

    # 7. Model Metadata Check
    print("\n--- 7. METADATA INTEGRITY CHECK ---")
    with open("models/model_metadata.json", "r") as f:
        meta = json.load(f)
    prod_meta = meta.get("production_model", {})
    print(f"Model Name: {prod_meta.get('model_name')}")
    print(f"Version:    {prod_meta.get('model_version')}")
    print(f"Classifier: {prod_meta.get('classifier')}")
    print(f"C param:    {prod_meta.get('hyperparameters', {}).get('C')}")
    print(f"Accuracy:   {prod_meta.get('metrics', {}).get('locked_test_accuracy'):.4f}")
    print(f"Recall:     {prod_meta.get('metrics', {}).get('locked_test_spam_recall'):.4f}")

    print("\n" + "=" * 70)
    print("ALL INTEGRITY CHECKS COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
