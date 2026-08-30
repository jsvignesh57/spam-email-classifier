# Phase 8 — Task 8.6: Decision Boundary & Threshold Analysis Report

## 1. Objective
The objective of **Task 8.6** is to determine whether adjusting the decision boundary threshold of the promoted `LinearSVC(C=10.0)` spam classifier ([final_spam_classifier_v2.joblib](file:///d:/Projects/Spam%20email%20-%20ML%20model/models/final_spam_classifier_v2.joblib)) can improve spam recall without unacceptably inflating false positives, while strictly upholding the project's **Spam Recall constraint** (baseline: **99.64%** recall, exactly **1** false negative on the locked test set).

---

## 2. Current Production Baseline Specification
- **Algorithm**: Linear Support Vector Machine (`LinearSVC`)
- **Regularization**: `C = 10.0`
- **Loss Function**: `squared_hinge`
- **Random State**: `42`
- **Feature Representation**: Word-level TF-IDF (`ngram_range=(1,2)`, `sublinear_tf=True`, `min_df=2`, `max_df=0.95`)
- **Active Model Artifact**: `models/final_spam_classifier_v2.joblib`
- **Active Vectorizer Artifact**: `models/tfidf_vectorizer.joblib`
- **Current Decision Boundary**: `threshold = 0.0`
- **Locked Test Performance (Reference)**:
  - **Accuracy**: 99.74%
  - **Spam Precision**: 99.27%
  - **Spam Recall**: **99.64%** (273 / 274 TP, exactly 1 FN)
  - **Spam F1-Score**: **0.9945**
  - **Confusion Matrix**: TN=863, FP=2, FN=1, TP=273

---

## 3. LinearSVC Decision Function Explanation
Unlike probabilistic classifiers (e.g., Logistic Regression, Naive Bayes), `LinearSVC` does not output posterior class probabilities $P(y=1|x)$. Instead, its prediction is governed by the raw signed decision function:
$$f(x) = w^T x + b$$
- $f(x) > 0$: Sample falls on the positive side of the separating hyperplane ($\hat{y} = \text{Spam}$).
- $f(x) < 0$: Sample falls on the negative side of the separating hyperplane ($\hat{y} = \text{Ham}$).
- $|f(x)|$: Proportional to the Euclidean geometric distance from the margin.

By introducing a decision threshold parameter $\tau \in \mathbb{R}$, we evaluate the generalized decision rule:
$$\hat{y}_\tau(x) = \begin{cases} 1 & \text{if } f(x) \ge \tau \\ 0 & \text{if } f(x) < \tau \end{cases}$$
Lowering $\tau < 0$ makes the classifier more aggressive in capturing spam (improving recall at the cost of false positives), while raising $\tau > 0$ makes it more conservative (improving precision at the cost of missed spam).

---

## 4. Why Threshold Analysis Was Performed
In **Task 8.1 Error Analysis**, diagnostic inspection revealed that several misclassified spam emails exhibited decision scores very close to the zero boundary (e.g., FN-2 at $-0.0084$, FN-3 at $-0.0260$). 

Adjusting $\tau$ provides a direct, post-training mechanism to explore the operating characteristic curve of the classifier without retraining the underlying support vector weights or modifying feature extraction.

---

## 5. Experimental Methodology & Data Leakage Prevention
1. **Zero Test-Set Contamination**: The 1,139-sample locked test partition was strictly excluded during out-of-fold scoring and threshold candidate selection.
2. **5-Fold Stratified Cross-Validation**: Executed exclusively on the 4,556-sample training partition (3,462 Ham, 1,094 Spam).
3. **Independent Fold Vectorization & Training**: In every CV fold, a fresh `TfidfVectorizer` and `LinearSVC(C=10.0)` were fitted strictly on the fold's training split and applied to score the fold's validation split.
4. **OOF Coverage Verification**: All 4,556 training samples received exactly one validation score with 0 duplicates and 0 omissions (`OOF COVERAGE CHECK: PASS`).

---

## 6. Decision-Score Distribution Analysis

### Score Distribution Summary (OOF Training Samples):
- **Ham Scores ($N=3,462$)**:
  - Min: `-2.5292`, Max: `0.5244`
  - Mean: `-1.2165`, Median: `-1.2195`, Std: `0.3820`
  - 5th Percentile: `-1.8195`, 95th Percentile: `-0.5994`
- **Spam Scores ($N=1,094$)**:
  - Min: `-0.5197`, Max: `1.8779`
  - Mean: `0.8523`, Median: `0.9700`, Std: `0.3390`
  - 5th Percentile: `0.1673`, 95th Percentile: `1.2734`
- **Near-Boundary Density ($[-1.0, 1.0]$)**:
  - Total near-boundary samples: `1669` (36.63% of training set)
  - Ham in $[-1.0, 1.0]$: `1004`
  - Spam in $[-1.0, 1.0]$: `665`

---

## 7. Out-of-Fold (OOF) Threshold Sweep Results

| Threshold ($\tau$) | Accuracy | Spam Precision | Spam Recall | Spam F1 | TN | FP | FN | TP | FPR | FNR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| -1.00 | 77.96% | 52.14% | 100.00% | 0.6855 | 2458.0 | 1004.0 | 0.0 | 1094.0 | 29.00% | 0.00% |
| **-0.75** | 91.35% | 73.52% | **100.00%** | **0.8474** | 3068.0 | 394.0 | 0.0 | 1094.0 | 11.38% | 0.00% |
| -0.50 | 97.50% | 90.63% | 99.91% | 0.9504 | 3349.0 | 113.0 | 1.0 | 1093.0 | 3.26% | 0.09% |
| -0.40 | 98.57% | 94.47% | 99.91% | 0.9711 | 3398.0 | 64.0 | 1.0 | 1093.0 | 1.85% | 0.09% |
| -0.30 | 99.19% | 97.06% | 99.63% | 0.9833 | 3429.0 | 33.0 | 4.0 | 1090.0 | 0.95% | 0.37% |
| -0.20 | 99.39% | 98.10% | 99.36% | 0.9873 | 3441.0 | 21.0 | 7.0 | 1087.0 | 0.61% | 0.64% |
| -0.15 | 99.52% | 98.73% | 99.27% | 0.9900 | 3448.0 | 14.0 | 8.0 | 1086.0 | 0.40% | 0.73% |
| -0.10 | 99.56% | 99.09% | 99.09% | 0.9909 | 3452.0 | 10.0 | 10.0 | 1084.0 | 0.29% | 0.91% |
| -0.05 | 99.54% | 99.36% | 98.72% | 0.9904 | 3455.0 | 7.0 | 14.0 | 1080.0 | 0.20% | 1.28% |
| +0.00 | 99.47% | 99.63% | 98.17% | 0.9890 | 3458.0 | 4.0 | 20.0 | 1074.0 | 0.12% | 1.83% |
| +0.05 | 99.39% | 99.72% | 97.71% | 0.9871 | 3459.0 | 3.0 | 25.0 | 1069.0 | 0.09% | 2.29% |
| +0.10 | 99.03% | 99.72% | 96.25% | 0.9795 | 3459.0 | 3.0 | 41.0 | 1053.0 | 0.09% | 3.75% |
| +0.15 | 98.81% | 99.71% | 95.34% | 0.9748 | 3459.0 | 3.0 | 51.0 | 1043.0 | 0.09% | 4.66% |
| +0.20 | 98.51% | 99.71% | 94.06% | 0.9680 | 3459.0 | 3.0 | 65.0 | 1029.0 | 0.09% | 5.94% |
| +0.30 | 98.07% | 99.70% | 92.23% | 0.9582 | 3459.0 | 3.0 | 85.0 | 1009.0 | 0.09% | 7.77% |
| +0.40 | 97.04% | 99.79% | 87.84% | 0.9344 | 3460.0 | 2.0 | 133.0 | 961.0 | 0.06% | 12.16% |
| +0.50 | 95.92% | 99.89% | 83.09% | 0.9072 | 3461.0 | 1.0 | 185.0 | 909.0 | 0.03% | 16.91% |
| +0.75 | 92.84% | 100.00% | 70.20% | 0.8249 | 3462.0 | 0.0 | 326.0 | 768.0 | 0.00% | 29.80% |
| +1.00 | 85.40% | 100.00% | 39.21% | 0.5634 | 3462.0 | 0.0 | 665.0 | 429.0 | 0.00% | 60.79% |

---

## 8. Selected Threshold & Selection Analysis

### Selection Hierarchy:
1. Primary Constraint: Highest Validation Spam Recall
2. Secondary Constraint: Lowest Validation False Negatives (FN)
3. Tertiary Constraint: Highest Validation Spam F1-Score
4. Quaternary Constraint: Highest Validation Spam Precision
5. Quinary Constraint: Lowest False Positives (FP)
6. Senary Constraint: Highest Accuracy
7. Parsimony Rule: Prefer threshold closest to 0.0 when metrics are effectively tied.

### Selection Outcome:
- **Selected Candidate Threshold ($\tau$)**: `-0.75`
- **OOF Validation Spam Recall**: 100.00% (0.0 FN)
- **OOF Validation Spam F1**: 0.8474
- **OOF False Positives**: 394.0
- **Selection Rationale**: Threshold -0.75 was selected from OOF validation as it achieves OOF Spam Recall of 100.00% (0.0 FN), OOF Spam F1 of 0.8474, with 394.0 FP.

---

## 9. Single Final Comparison on Locked Test Set (1,139 Emails)

The selected threshold was evaluated strictly ONCE against the held-out locked test partition:

| Metric | Baseline Threshold ($\tau = 0.0$) | Candidate Threshold ($\tau = -0.75$) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.74% | 92.36% | -7.37% |
| **Spam Precision** | 99.27% | 75.90% | -23.37% |
| **Spam Recall** | **99.64%** | **100.00%** | **+0.36%** |
| **Spam F1-Score** | **0.9945** | **0.8630** | **-0.1315** |
| **True Negatives (TN)** | 863 | 778 | -85 |
| **False Positives (FP)** | 2 | 87 | +85 |
| **False Negatives (FN)** | **1** | **0** | **-1** |
| **True Positives (TP)** | 273 | 274 | +1 |

---

## 10. Recall Analysis & Gate Evaluation
- **Baseline Test Spam Recall**: **99.64%** (1 FN)
- **Candidate Test Spam Recall**: **100.00%** (0 FN)
- **Recall Gate Check (>= 99.64%)**: **PASS**

---

## 11. Connection to Task 8.1 Error Analysis Findings
In **Task 8.1**, diagnostic analysis showed:
- **FN-1 (Index 92)**: Conversational B2B virtual tour email with heavy legitimate corporate vocabulary (`"thanks"`, `"organization"`, `"houston"`). Decision score with C=10 baseline is $-0.2545$.
- Setting $\tau \le -0.26$ would capture FN-1, but shifting the threshold that deep into ham territory sharply increases False Positives in cross-validation (FP increases significantly from 4 up to 10+), degrading precision and F1.
- The standard decision boundary $\tau = 0.0$ already provides near-optimal balance on this dataset.

---

## 12. Limitations
1. **Uncalibrated Margin Scale**: Decision scores from `LinearSVC` depend on the specific norm of $w$ and dataset scaling; they do not represent absolute confidence probabilities.
2. **Distribution Drift**: Optimal threshold boundaries tuned closely on training distributions may be sensitive to slight shifts in spamming tactics or ham terminology in deployment.

---

## 13. Final Decision & Status
- **Recall Requirement Check (>= 99.64%)**: **PASS**
- **Decision Outcome**: **RETAIN_BASELINE**
- **Decision Statement**: Current production threshold retained.
- **Production Model Status**: `models/final_spam_classifier_v2.joblib` with standard threshold `0.0` remains the **ACTIVE PROMOTED PRODUCTION CLASSIFIER**.
