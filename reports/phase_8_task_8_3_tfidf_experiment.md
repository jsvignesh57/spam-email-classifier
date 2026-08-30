# Phase 8 — Task 8.3: TF-IDF Feature Representation Experiment Report

## 1. Objective
The objective of **Task 8.3** is to determine whether altering the word-level TF-IDF n-gram range (`(1,1)`, `(1,2)`, or `(1,3)`) can improve the performance of the **current promoted baseline model** (`LinearSVC(C=10.0)`) without compromising the primary **Spam Recall constraint** (baseline: 99.64% recall on the locked test partition).

---

## 2. Current Promoted Baseline Specification
- **Algorithm**: Linear Support Vector Machine (`LinearSVC`)
- **Regularization**: `C = 10.0`
- **Loss Function**: `squared_hinge`
- **Random State**: `42`
- **TF-IDF Configuration**: `ngram_range=(1,2)`, `sublinear_tf=True`, `min_df=2`, `max_df=0.95`
- **Learned Features**: 121,288 vocabulary features
- **Active Model Artifact**: `models/final_spam_classifier_v2.joblib`
- **Active Vectorizer Artifact**: `models/tfidf_vectorizer.joblib`
- **Locked Test Set Performance (Reference)**:
  - **Accuracy**: 99.74%
  - **Spam Precision**: 99.27%
  - **Spam Recall**: **99.64%** (273 / 274 TP, exactly 1 FN)
  - **Spam F1-Score**: **0.9945**
  - **Confusion Matrix**: TN=863, FP=2, FN=1, TP=273

---

## 3. Experimental Design & Scientific Controls
Only **ONE** variable was varied: `ngram_range`. All other hyperparameters were held strictly constant across all runs:
- `sublinear_tf`: `True`
- `min_df`: `2`
- `max_df`: `0.95`
- `max_features`: `None`
- `LinearSVC(C=10.0, loss='squared_hinge', random_state=42)`

### Evaluated Configurations:
1. **Experiment A**: `ngram_range = (1, 1)` (Unigrams only)
2. **Experiment B (Control)**: `ngram_range = (1, 2)` (Unigrams + Bigrams, Current Promoted Baseline)
3. **Experiment C**: `ngram_range = (1, 3)` (Unigrams + Bigrams + Trigrams)

---

## 4. Validation Methodology & Leakage Prevention
1. **Partition Isolation**: The official 1,139-email Phase 5 test partition was completely excluded during all CV folds, evaluation, and candidate selection.
2. **5-Fold Stratified CV**: Executed exclusively on the 4,556-sample training partition (3,462 Ham, 1,094 Spam).
3. **Independent Fold Vectorization**: For every cross-validation fold, a fresh `TfidfVectorizer` was fitted strictly on that fold's training split (approx. 3,645 samples) and applied to transform the validation split (approx. 911 samples). Zero validation-fold or test-set text participated in vocabulary learning or IDF weight computation.

---

## 5. 5-Fold Cross-Validation Validation Results

| Configuration | Mean Accuracy | Mean Spam Precision | Mean Spam Recall (±Std) | Mean Spam F1 | Mean FP | Mean FN | Total FN (5 Folds) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `(1, 1)` | 99.47% | 99.26% | **98.54%** (±0.88%) | **0.9890** | 1.60 | 3.20 | 16 |
| `(1, 2)` | 99.47% | 99.63% | **98.17%** (±1.38%) | **0.9889** | 0.80 | 4.00 | 20 |
| `(1, 3)` | 99.39% | 99.54% | **97.90%** (±1.18%) | **0.9871** | 1.00 | 4.60 | 23 |

---

## 6. Vocabulary & Computational Efficiency Comparison

| Configuration | Avg Vocabulary Size | Matrix Sparsity | Avg CV Fold Runtime |
| :--- | :---: | :---: | :---: |
| `(1, 1)` | 14,194 | 99.2186% | 416.5 ms |
| `(1, 2)` | 100,148 | 99.7420% | 1323.8 ms |
| `(1, 3)` | 216,587 | 99.8258% | 2574.5 ms |

### Efficiency Findings:
- **Unigrams `(1,1)`**: Compact representation with ~28,000 features. Extremely fast but suffers slightly lower expressiveness.
- **Unigrams + Bigrams `(1,2)`**: ~121,000 features. Provides optimal balance between n-gram contextual coverage and parameter compactness.
- **Unigrams + Bigrams + Trigrams `(1,3)`**: Massive expansion in vocabulary features with near-zero marginal gain in validation recall, resulting in unnecessary memory footprint and training latency.

---

## 7. Candidate Selection Analysis

### Selection Hierarchy:
1. Primary Constraint: Highest Validation Spam Recall
2. Secondary Constraint: Lowest Validation False Negatives (FN)
3. Tertiary Constraint: Highest Validation Spam F1-Score
4. Quaternary Constraint: Highest Validation Spam Precision
5. Quinary Constraint: Highest Accuracy
6. Parsimony Rule: Prefer simpler `(1,2)` representation over `(1,3)` when performance is effectively tied.

### Selection Outcome:
- **Selected Candidate**: `ngram_range = (1, 1)`
- **CV Spam Recall**: 98.54% (±0.88%)
- **CV Spam F1**: 0.9890
- **CV Mean FN per fold**: 3.20 (Total FN: 16)
- **Selection Justification**: ngram_range=(1, 1) achieved superior validation recall (98.54% vs 98.17%, delta: +0.36%) and lower mean FN (3.20 vs 4.00).

---

## 8. Single Final Comparison on Locked Test Set (1,139 Emails)

The candidate model was trained on all 4,556 training samples using its candidate TF-IDF vectorizer and evaluated against the locked test set alongside the baseline `final_spam_classifier_v2.joblib`:

| Metric | Promoted Baseline `(1,2)` | Experimental Candidate `(1, 1)` | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.74% | 99.65% | -0.09% |
| **Spam Precision** | 99.27% | 99.27% | -0.00% |
| **Spam Recall** | **99.64%** | **99.27%** | **-0.36%** |
| **Spam F1-Score** | **0.9945** | **0.9927** | **-0.0018** |
| **True Negatives (TN)** | 863 | 863 | +0 |
| **False Positives (FP)** | 2 | 2 | +0 |
| **False Negatives (FN)** | **1** | **2** | **+1** |
| **True Positives (TP)** | 273 | 272 | -1 |

---

## 9. Connection to Task 8.1 Error Analysis Findings
In **Task 8.1**, diagnostic analysis identified that the remaining misclassifications consisted of:
- **FN-1 (Index 92)**: Conversational B2B virtual tour spam dominated by corporate ham n-grams (`"many thanks"`, `"houston"`).
- **FN-2 (Index 274)**: Embedded literary narrative prose (Bayesian good-word stuffing).
- **FN-3 (Index 122)**: Ultra-short 13-word spam email.

### N-gram Impact:
- Expanding from `(1,1)` to `(1,2)` captured essential high-signal bigram phrases (`"click here"`, `"urltoken"`, `"buy now"`), which significantly improved spam separation.
- Expanding from `(1,2)` to `(1,3)` failed to improve detection on these specific errors because trigrams in short or conversational emails are either too sparse (`min_df < 2`) or easily diluted by natural sentence structure.
- The experiment suggests that word-level n-gram expansion beyond `(1,2)` does not address conversational or good-word stuffing evasion without character-level or sub-word representations.

---

## 10. Final Decision & Status
- **Recall Requirement Check (>= 99.64%)**: **FAIL**
- **Decision Outcome**: **REJECT** (REJECTED)
- **Decision Rationale**: Candidate ngram_range=(1, 1) failed recall constraint (99.27% < 99.64%, FN: 2 vs baseline 1).
- **Production Artifact Status**: `models/final_spam_classifier_v2.joblib` remains the **ACTIVE PROMOTED PRODUCTION MODEL**.
