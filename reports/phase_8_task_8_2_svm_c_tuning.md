# Phase 8 — Task 8.2: Linear SVM Hyperparameter Optimization Report

## 1. Objective
The objective of **Task 8.2** is to determine whether tuning the regularization hyperparameter `C` in the `LinearSVC` model can improve classifier performance without compromising the **Spam Recall constraint** (baseline: 98.91% recall on the locked test partition).

---

## 2. Baseline Model Specification & Reference Performance
- **Algorithm**: Linear Support Vector Machine (`LinearSVC`)
- **Loss Function**: `squared_hinge`
- **Baseline Hyperparameter**: `C = 1.0`
- **Random State**: `42`
- **Vectorization**: TF-IDF (Unigrams + Bigrams, Sublinear TF, `min_df=2`, `max_df=0.95`)
- **Phase 5 Locked Test Performance (Reference)**:
  - **Accuracy**: 99.56%
  - **Spam Precision**: 99.27%
  - **Spam Recall**: **98.91%** (271 / 274 TP, exactly 3 FN)
  - **Spam F1-Score**: 0.9909
  - **Confusion Matrix**: TN=863, FP=2, FN=3, TP=271

---

## 3. Validation Methodology & Data Leakage Prevention
To prevent test set contamination and overfitting:
1. **Strict Test-Set Isolation**:
   - The official 1,139-email Phase 5 test partition was **completely excluded** during all hyperparameter evaluation, tuning, and candidate selection.
2. **5-Fold Stratified Cross-Validation**:
   - StratifiedKFold (`n_splits=5`, `shuffle=True`, `random_state=42`) was performed exclusively on the 4,556-email Phase 3 training partition (3,462 Ham, 1,094 Spam).
3. **Leakage-Safe Within-Fold TF-IDF Fitting**:
   - For every cross-validation fold, a fresh `TfidfVectorizer` was fitted strictly on the fold's training split (approx. 3,645 samples) and applied to transform the fold's validation split (approx. 911 samples). No vocabulary or IDF weights from the validation fold or locked test set were accessible to the model.

---

## 4. Candidate C Values Evaluated
Grid of `C` regularization strengths evaluated:
`[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]`

---

## 5. 5-Fold Cross-Validation Results Table

| Regularization `C` | Mean Accuracy | Mean Spam Precision | Mean Spam Recall (±Std) | Mean Spam F1 | Mean FP | Mean FN | Total FN (5 Folds) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.01` | 78.67% | 100.00% | **11.15%** (±1.94%) | **0.2001** | 0.00 | 194.40 | 972 |
| `0.05` | 93.74% | 99.76% | **74.13%** (±1.66%) | **0.8505** | 0.40 | 56.60 | 283 |
| `0.1` | 97.37% | 99.70% | **89.31%** (±1.80%) | **0.9421** | 0.60 | 23.40 | 117 |
| `0.25` | 98.62% | 99.43% | **94.79%** (±1.37%) | **0.9705** | 1.20 | 11.40 | 57 |
| `0.5` | 99.06% | 99.44% | **96.62%** (±1.02%) | **0.9800** | 1.20 | 7.40 | 37 |
| `1.0` | 99.19% | 99.53% | **97.08%** (±1.06%) | **0.9829** | 1.00 | 6.40 | 32 |
| `2.0` | 99.39% | 99.63% | **97.81%** (±1.34%) | **0.9870** | 0.80 | 4.80 | 24 |
| `5.0` | 99.45% | 99.72% | **97.99%** (±1.24%) | **0.9884** | 0.60 | 4.40 | 22 |
| `10.0` | 99.47% | 99.63% | **98.17%** (±1.38%) | **0.9889** | 0.80 | 4.00 | 20 |

---

## 6. Candidate Ranking & Selection Analysis

### Selection Hierarchy:
1. **Primary Constraint**: Highest Mean Spam Recall
2. **Secondary Constraint**: Lowest Mean False Negatives (FN)
3. **Tertiary Constraint**: Highest Mean Spam F1-Score
4. **Quaternary Constraint**: Highest Mean Spam Precision
5. **Quinary Constraint**: Highest Accuracy
6. **Tie-Breaking**: Favor current baseline configuration (`C = 1.0`)

### Top Candidate from CV:
- **Selected Candidate**: `C = 10.0`
- **CV Spam Recall**: 98.17% (±1.38%)
- **CV Spam F1**: 0.9889
- **CV Spam Precision**: 99.63%
- **CV Mean FN per fold**: 4.00 (Total FN across 5 folds: 20)
- **CV Mean FP per fold**: 0.80

### Reason for Selection:
C=10.0 achieved higher/equal validation spam recall (98.17% vs baseline 97.08%, delta: +1.10%) and mean FN of 4.00 vs baseline 6.40 (delta: -2.40).

---

## 7. Single Final Comparison on Locked Test Set (1,139 Emails)

After selecting the optimal candidate from 5-fold CV, both the baseline (`C=1.0`) and the experimental candidate (`C=10.0`) were trained on the entire 4,556-sample training partition with canonical TF-IDF and evaluated on the locked test partition:

| Metric | Current Baseline (`C=1.0`) | Experimental Candidate (`C=10.0`) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.56% | 99.74% | +0.18% |
| **Spam Precision** | 99.27% | 99.27% | +0.01% |
| **Spam Recall** | **98.91%** | **99.64%** | **+0.73%** |
| **Spam F1-Score** | **0.9909** | **0.9945** | **+0.0037** |
| **True Negatives (TN)** | 863 | 863 | +0 |
| **False Positives (FP)** | 2 | 2 | +0 |
| **False Negatives (FN)** | **3** | **1** | **-2** |
| **True Positives (TP)** | 271 | 273 | +2 |

---

## 8. Connection to Task 8.1 Error Analysis Findings
In **Task 8.1**, diagnostic analysis of the 5 misclassifications revealed:
1. **FP-1 (Index 2837)**: Business email discussing commercial price-matching clauses (score `+0.0491`).
2. **FP-2 (Index 2863)**: Short technical RFC link containing `"click here"` (score `+0.0105`).
3. **FN-1 (Index 92)**: Conversational B2B virtual tour spam dominated by ham corporate sign-offs (score `-0.2545`).
4. **FN-2 (Index 274)**: Good-word stuffing story prose diluting pharmacy spam (score `-0.0084`).
5. **FN-3 (Index 122)**: Ultra-short 13-word spam lacking standard keywords (score `-0.0260`).

### Impact of Regularization (`C`):
- Higher `C` values (e.g., `C >= 2.0`) apply less regularization, allowing larger feature weights to fit complex combinations, but also increase vulnerability to high-weight false positive triggers without pulling distant B2B false negatives across the boundary.
- Lower `C` values (e.g., `C <= 0.1`) heavily regularize weights, causing a severe collapse in spam recall (e.g., `C=0.01` drops CV recall dramatically to near-zero as sparse spam features are penalized excessively).
- `C = 1.0` and its immediate neighborhood represent the optimal regularization trade-off on this high-dimensional TF-IDF space (121,288 n-gram features).

---

## 9. Final Decision & Status

- **Recall Requirement Check (>= 98.91%)**: **PASS**
- **Final Candidate Decision**: **ACCEPT**
- **Decision Rationale**: Experimental C=10.0 achieved test recall 99.64% (>= baseline 98.91%) with lower FN (1 vs 3) or improved F1 (0.9945 vs 0.9909).
- **Production Artifact Status**: Baseline `models/final_spam_classifier.joblib` and `models/linear_svm_model.joblib` remain **UNMODIFIED**.

---

## 10. Limitations & Next Steps
- **Limitations**: Modifying only the global regularization constant `C` scales margin penalties uniformly across all features, but cannot inherently resolve class-imbalance boundary shift or address good-word stuffing in adversarial emails without specialized threshold tuning or class weighting.
- **Next Step (Phase 8 Task 8.3 / Beyond)**: Investigate decision threshold calibration or class-weighting adjustments (`class_weight='balanced'`) to target the specific boundary-proximity false negatives identified in Task 8.1.
