# Phase 8 — Task 8.5: Combined Word + Character TF-IDF Feature Representation Experiment Report

## 1. Objective
The objective of **Task 8.5** is to determine whether combining **word-level TF-IDF** and **character-level TF-IDF** features into a unified, sparse feature representation can improve upon the current promoted production baseline (`LinearSVC(C=10.0)` with word-level TF-IDF `(1,2)` in `models/final_spam_classifier_v2.joblib`) without violating the project's non-negotiable **Spam Recall constraint** (baseline: **99.64%** recall, **1** false negative on the locked 1,139-sample test set).

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

## 3. Motivation for Combining Features
While word-level n-grams capture semantic meaning, phrase context, and domain-specific vocabulary (e.g., `"click here"`, `"urltoken"`, `"vince"`), character n-grams capture fine-grained sub-word structures, obfuscation variants (`v1agra`, `c!al!s`), punctuation triggers, and morphological affixes. 

By combining word-level TF-IDF with character-level TF-IDF via sparse horizontal stacking (`scipy.sparse.hstack`), the classifier theoretically gains access to both macroscopic contextual phrases and microscopic sub-word signals simultaneously.

---

## 4. Previous Task 8.3 Findings
In **Task 8.3** (Word n-gram exploration):
- Unigrams + Bigrams `(1,2)` proved to be the optimal word-level representation.
- Expanding to Trigrams `(1,3)` generated 216,587 features but produced a slight degradation in validation recall (97.90% vs 98.17% for (1,2)), establishing `(1,2)` as the standard word baseline.

---

## 5. Previous Task 8.4 Findings
In **Task 8.4** (Pure character TF-IDF exploration):
- All pure character-only models were **rejected** because character-only representations diluted word-level anchors across millions of character fragments, dropping locked-test recall to 98.91% (3 FN).
- Among character configurations, `char (3,5)` achieved the highest validation recall (98.90% CV recall) and the lowest parameter count (~170,000 features).
- `char (4,7)` produced 748,724 features and severe recall degradation.
- **Key Conclusion**: Character features must *never* replace word features; Task 8.5 isolates whether they can *complement* word features.

---

## 6. Experimental Configurations
All configurations maintained strict controls: `LinearSVC(C=10.0, loss='squared_hinge', random_state=42)`, with word and character TF-IDF using `sublinear_tf=True`, `min_df=2`, `max_df=0.95`:
1. **Baseline Reference**: Word `(1,2)` alone
2. **Primary Combined**: Word `(1,2)` + Char `(3,5)`
3. **Secondary Combined**: Word `(1,2)` + Char `(3,6)`

*(Note: `char (4,7)` was excluded due to excessive dimensionality and established inferiority in Task 8.4).*

---

## 7. Experimental Methodology & Data Leakage Prevention
1. **Partition Isolation**: The 1,139-email locked test partition was strictly excluded from all cross-validation folds, feature selection, and candidate ranking.
2. **5-Fold Stratified CV**: Conducted strictly on the 4,556-sample training partition (3,462 Ham, 1,094 Spam).
3. **Within-Fold Vectorizer Fitting**: Both word and character TF-IDF vectorizers were fitted strictly on each fold's training split (~3,645 samples) and applied to transform the validation split (~911 samples).
4. **Sparse Memory Safety**: Feature matrices were concatenated using `scipy.sparse.hstack(..., format='csr')`. Zero dense array conversions were performed.

---

## 8. 5-Fold Cross-Validation Results

| Configuration | Mean Accuracy | Mean Spam Precision | Mean Spam Recall (±Std) | Mean Spam F1 | Mean FP | Mean FN | Total FN (5 Folds) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Word (1, 2) baseline` | 99.47% | 99.63% | **98.17%** (±1.38%) | **0.9889** | 0.80 | 4.00 | 20 |
| `Word (1, 2) + Char (3, 5)` | 99.65% | 99.72% | **98.81%** (±0.62%) | **0.9926** | 0.60 | 2.60 | 13 |
| `Word (1, 2) + Char (3, 6)` | 99.60% | 99.72% | **98.63%** (±0.58%) | **0.9917** | 0.60 | 3.00 | 15 |

---

## 9. Feature-Count & Matrix Sparsity Analysis

| Configuration | Word Features | Char Features | Total Features | Non-Zero Entries | Matrix Sparsity | CV Fold Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Word (1, 2) baseline` | 100,148 | 0 | **100,148** | 941,857 | 99.7420% | 3.63 s |
| `Word (1, 2) + Char (3, 5)` | 100,148 | 170,452 | **270,601** | 9,461,262 | 99.0407% | 26.19 s |
| `Word (1, 2) + Char (3, 6)` | 100,148 | 402,982 | **503,131** | 13,179,202 | 99.2813% | 41.32 s |

---

## 10. Computational Efficiency Analysis
- **Baseline Word (1,2)**: ~100,148 fold features, ~1.3s training latency per fold.
- **Primary Combined Word (1,2) + Char (3,5)**: ~270,600 fold features (~170k character + ~100k word), training latency increases moderately to ~26.19s per fold.
- **Secondary Combined Word (1,2) + Char (3,6)**: ~503,130 fold features (~403k character + ~100k word), training latency expands to ~41.32s per fold.
- **Memory Footprint**: Memory usage remained entirely bounded due to strict CSR sparse representation throughout training and inference.

---

## 11. Data Leakage Prevention Verification
- Exact data split loaded from `data/processed/train_test_split.npz` (4,556 train / 1,139 locked test).
- Zero test text participated in vectorizer vocabulary building or IDF computation.
- Within-fold vectorization confirmed: `PHASE 8.5 DATA LEAKAGE CHECK: PASS`.

---

## 12. Candidate Selection Analysis

### Selection Hierarchy:
1. Primary Constraint: Highest Validation Spam Recall
2. Secondary Constraint: Lowest Validation False Negatives (FN)
3. Tertiary Constraint: Highest Validation Spam F1-Score
4. Quaternary Constraint: Highest Validation Spam Precision
5. Quinary Constraint: Highest Accuracy
6. Parsimony Rule: Prefer lower dimensional configuration when metrics are tied.

### Selection Outcome:
- **Selected Candidate**: `Word (1, 2) + Char (3, 5)`
- **CV Spam Recall**: 98.81% (±0.62%)
- **CV Spam F1**: 0.9926
- **CV Mean FN per fold**: 2.60 (Total FN: 13)
- **Selection Rationale**: Configuration Word (1, 2) + Char (3, 5) selected as the strongest combined candidate with CV Spam Recall of 98.81% (±0.62%), Mean FN of 2.60 per fold, and Spam F1 of 0.9926.

---

## 13. Single Final Comparison on Locked Test Set (1,139 Emails)

The candidate combined model was trained on all 4,556 training samples and evaluated strictly ONCE against the locked 1,139-sample test set:

| Metric | Promoted Baseline (Word (1,2)) | Experimental Candidate (Word (1, 2) + Char (3, 5)) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.74% | 99.56% | -0.18% |
| **Spam Precision** | 99.27% | 99.27% | -0.01% |
| **Spam Recall** | **99.64%** | **98.91%** | **-0.73%** |
| **Spam F1-Score** | **0.9945** | **0.9909** | **-0.0037** |
| **True Negatives (TN)** | 863 | 863 | +0 |
| **False Positives (FP)** | 2 | 2 | +0 |
| **False Negatives (FN)** | **1** | **3** | **+2** |
| **True Positives (TP)** | 273 | 271 | -2 |

---

## 14. Recall Analysis & Gate Evaluation
- **Baseline Test Spam Recall**: **99.64%** (1 FN)
- **Candidate Test Spam Recall**: **98.91%** (3 FN)
- **Recall Gate Check (>= 99.64%)**: **FAIL**

---

## 15. Connection to Task 8.1 Error Analysis Findings
In **Task 8.1**, error analysis identified three specific error types:
1. **B2B Conversational Spam (FN-1, Index 92)**: Dominated by polite business terms (`"thanks"`, `"organization"`, `"houston"`).
2. **Adversarial Good-Word Stuffing (FN-2, Index 274)**: Diluted with literary narrative prose.
3. **Ultra-Sparse Nonsensical Spam (FN-3, Index 122)**: Very short email with sparse keywords.

### Diagnostic Evaluation:
- Adding character n-grams to word n-grams allows the model to capture sub-word patterns and morphological variations.
- However, for conversational B2B outreach (such as FN-1 at Index 92), the presence of character n-grams from legitimate business vocabulary does not significantly alter the linear decision score because the corporate phrasing itself is genuine natural language.
- The results suggest that while combined word+character representations preserve high precision and strong overall classification capability, they do not resolve the residual conversational B2B false negative without shifting the decision boundary.

---

## 16. Limitations
1. **Feature Space Expansion**: Stacking word and character features expands dimensionality to >290,000 features, increasing vectorization latency and artifact size.
2. **Corpus Characteristics**: In an Enron/clean spam benchmark where spam tokens are relatively clear, character sub-word splitting provides diminishing marginal returns compared to whole-word n-grams.

---

## 17. Final Decision & Status
- **Recall Requirement Check (>= 99.64%)**: **FAIL**
- **Decision Outcome**: **REJECT**
- **Decision Statement**: Combined word + character representation rejected. Current C=10 + word TF-IDF (1,2) model retained.
- **Production Model Status**: `models/final_spam_classifier_v2.joblib` (`LinearSVC(C=10.0)` + Word TF-IDF `(1,2)`) remains the **ACTIVE PROMOTED PRODUCTION MODEL**.
- **Candidate Artifacts Saved**:
  - `models/phase_8_5_candidate_word_tfidf.joblib`
  - `models/phase_8_5_candidate_char_tfidf.joblib`
  - `models/phase_8_5_candidate_combined_svm.joblib`
