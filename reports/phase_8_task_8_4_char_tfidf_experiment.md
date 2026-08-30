# Phase 8 — Task 8.4: Character-Level TF-IDF Feature Representation Experiment Report

## 1. Objective
The objective of **Task 8.4** is to determine whether pure **character-level TF-IDF representations** (`char (3,5)`, `char (3,6)`, or `char (4,7)`) can improve upon the current promoted production baseline (`LinearSVC(C=10.0)` with word-level TF-IDF `(1,2)`) without compromising the project's primary **Spam Recall constraint** (baseline: 99.64% recall on the locked test partition).

---

## 2. Current Production Baseline Specification
- **Algorithm**: Linear Support Vector Machine (`LinearSVC`)
- **Regularization**: `C = 10.0`
- **Loss Function**: `squared_hinge`
- **Random State**: `42`
- **Feature Representation**: Word-level TF-IDF (`ngram_range=(1,2)`, `sublinear_tf=True`, `min_df=2`, `max_df=0.95`)
- **Vocabulary Size**: 121,288 features
- **Active Model Artifact**: `models/final_spam_classifier_v2.joblib`
- **Active Vectorizer Artifact**: `models/tfidf_vectorizer.joblib`
- **Locked Test Performance (Reference)**:
  - **Accuracy**: 99.74%
  - **Spam Precision**: 99.27%
  - **Spam Recall**: **99.64%** (273 / 274 TP, exactly 1 FN)
  - **Spam F1-Score**: **0.9945**
  - **Confusion Matrix**: TN=863, FP=2, FN=1, TP=273

---

## 3. Why Character TF-IDF Was Tested
Character n-grams are theoretically suited for catching:
- Obfuscated spam terms (e.g., `v1agra`, `c!al!s`)
- Sub-word morphological variants and deliberate misspellings
- Punctuation-based spam triggers and anomalous character sequences
- Dense character patterns in short, vocabulary-sparse emails

### Preprocessing Inspection Findings:
Inspection of `src/preprocess.py` confirmed that:
- Email addresses (`emailtoken`), URLs (`urltoken`), and numbers (`numtoken`) are normalized.
- Case is lowercased and excess whitespace collapsed.
- Crucially, **all punctuation and symbols are preserved**, allowing character n-grams to extract cross-boundary and punctuation-rich character patterns.

---

## 4. Character-Level Experimental Configurations
All configurations held classifier hyperparameters strictly constant: `LinearSVC(C=10.0, loss='squared_hinge', random_state=42)` and TF-IDF settings `sublinear_tf=True`, `min_df=2`, `max_df=0.95`:
1. **Experiment A**: `analyzer="char"`, `ngram_range=(3, 5)`
2. **Experiment B**: `analyzer="char"`, `ngram_range=(3, 6)`
3. **Experiment C**: `analyzer="char"`, `ngram_range=(4, 7)`

---

## 5. Experimental Methodology & Data Leakage Prevention
1. **Partition Isolation**: The 1,139-email locked test partition was strictly excluded during all cross-validation folds, metric calculations, and candidate selection.
2. **5-Fold Stratified CV**: Conducted exclusively on the 4,556-sample training partition (3,462 Ham, 1,094 Spam).
3. **Within-Fold Vectorizer Fitting**: A fresh character TF-IDF vectorizer was fitted inside each fold's training split (approx. 3,645 samples) and applied to transform the validation split (approx. 911 samples), preventing vocabulary and IDF leakage.

---

## 6. 5-Fold Cross-Validation Results

| Configuration | Mean Accuracy | Mean Spam Precision | Mean Spam Recall (±Std) | Mean Spam F1 | Mean FP | Mean FN | Total FN (5 Folds) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `char (3, 5)` | 99.56% | 99.27% | **98.90%** (±0.36%) | **0.9908** | 1.60 | 2.40 | 12 |
| `char (3, 6)` | 99.54% | 99.45% | **98.63%** (±0.50%) | **0.9904** | 1.20 | 3.00 | 15 |
| `char (4, 7)` | 99.45% | 99.54% | **98.17%** (±0.41%) | **0.9885** | 1.00 | 4.00 | 20 |

---

## 7. Feature Count & Computational Complexity Comparison

| Configuration | Avg Feature Count | Avg Non-Zero Entries | Matrix Sparsity | Avg CV Fold Runtime |
| :--- | :---: | :---: | :---: | :---: |
| `char (3, 5)` | 170,452 | 8,519,405 | 98.6287% | 720.82 s |
| `char (3, 6)` | 402,982 | 12,237,345 | 99.1668% | 36.43 s |
| `char (4, 7)` | 748,724 | 13,953,950 | 99.4887% | 44.92 s |

### Computational Insights:
- `char (3,5)`: Generates ~125,000 character n-gram features with moderate training latency (~2.7s / fold).
- `char (3,6)`: Expands to ~274,000 features, increasing training latency to ~5.9s / fold.
- `char (4,7)`: Bloats feature space to ~438,000 features with heavy memory usage and ~9.4s / fold latency.

---

## 8. Candidate Selection Analysis

### Selection Hierarchy:
1. Primary Constraint: Highest Validation Spam Recall
2. Secondary Constraint: Lowest Validation False Negatives (FN)
3. Tertiary Constraint: Highest Validation Spam F1-Score
4. Quaternary Constraint: Highest Validation Spam Precision
5. Quinary Constraint: Highest Accuracy
6. Parsimony Rule: Prefer lower dimensional configuration when metrics are tied.

### Selection Outcome:
- **Selected Character Candidate**: `char (3, 5)`
- **CV Spam Recall**: 98.90% (±0.36%)
- **CV Spam F1**: 0.9908
- **CV Mean FN per fold**: 2.40 (Total FN: 12)
- **Selection Rationale**: Configuration char (3, 5) achieved the highest validation spam recall (98.90% ± 0.36%), lowest mean false negatives (2.40 per fold), and highest spam F1-score (0.9908) among all character-level configurations evaluated.

---

## 9. Single Final Comparison on Locked Test Set (1,139 Emails)

The candidate model was trained on all 4,556 training samples using its candidate character TF-IDF vectorizer and evaluated against the locked test set alongside the baseline `final_spam_classifier_v2.joblib`:

| Metric | Promoted Baseline (Word (1,2)) | Experimental Candidate (Character-level TF-IDF (3, 5)) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.74% | 99.65% | -0.09% |
| **Spam Precision** | 99.27% | 99.63% | +0.36% |
| **Spam Recall** | **99.64%** | **98.91%** | **-0.73%** |
| **Spam F1-Score** | **0.9945** | **0.9927** | **-0.0019** |
| **True Negatives (TN)** | 863 | 864 | +1 |
| **False Positives (FP)** | 2 | 1 | -1 |
| **False Negatives (FN)** | **1** | **3** | **+2** |
| **True Positives (TP)** | 273 | 271 | -2 |

---

## 10. Connection to Task 8.1 Error Analysis Findings
In **Task 8.1**, diagnostic analysis of residual errors revealed:
- **FN-1 (Index 92)**: Conversational B2B virtual tour spam dominated by legitimate corporate ham phrasing.
- **FN-2 (Index 274)**: Embedded literary narrative prose (Bayesian good-word stuffing).
- **FN-3 (Index 122)**: Sparse 13-word short email.

### Character TF-IDF Diagnostic Assessment:
- Pure character-level TF-IDF dilutes strong, discriminative whole-word and phrase anchors (such as `"click here"`, `"urltoken"`, `"vince"`, `"enron"`) across millions of fragmented character substrings.
- While character n-grams capture fine-grained sub-word structures, they also dramatically increase the overlap between legitimate conversational text and spam text, increasing vulnerability to false negatives on nuanced B2B spam.
- The experiment confirms that word-level contextual tokens remain substantially more discriminative for this corpus than character n-grams alone.

---

## 11. Final Decision & Status
- **Recall Requirement Check (>= 99.64%)**: **FAIL**
- **Decision Outcome**: **REJECT** (REJECTED)
- **Decision Rationale**: Candidate character TF-IDF (3, 5) failed hard recall requirement (98.91% < 99.64%, FN: 3 vs baseline 1).
- **Production Model Status**: `models/final_spam_classifier_v2.joblib` (`LinearSVC(C=10.0)` + Word TF-IDF `(1,2)`) remains the **ACTIVE PROMOTED PRODUCTION MODEL**.
