# Spam Email Classifier — Official Production Model Card

## 1. Model Overview
- **Model Name**: Spam Email Classifier
- **Active Production Model**: Linear Support Vector Machine (`LinearSVC`, `C=10.0`)
- **Model Version**: `v2.0.0 (Production Frozen)`
- **Production Artifact**: `models/final_spam_classifier_v2.joblib`
- **Vectorizer Artifact**: `models/tfidf_vectorizer.joblib`
- **Algorithm**: `sklearn.svm.LinearSVC`
- **Decision Threshold**: `0.0` (Standard geometric boundary on raw signed margin)
- **Status**: **FROZEN / PRODUCTION**

---

## 2. Model Lineage & Version History

| Version | Artifact Path | Parameters | Spam Recall | Spam Precision | Spam F1 | Test FN | Test FP | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **v1 (Baseline)** | `models/final_spam_classifier.joblib` | `LinearSVC(C=1.0)` | 98.91% | 99.27% | 0.9909 | 3 | 2 | **Preserved Baseline** |
| **v2 (Production)** | `models/final_spam_classifier_v2.joblib` | `LinearSVC(C=10.0)` | **99.64%** | **99.27%** | **0.9945** | **1** | **2** | **Frozen Production** |

### Promotion & Freeze History
1. **Phase 4–6**: LinearSVC ($C=1.0$) was packaged as the baseline classifier after outperforming Multinomial Naive Bayes (98.91% recall vs 48.18%).
2. **Phase 8.2**: Systematic 5-fold cross-validation on training data identified $C=10.0$ as optimal, reducing false negatives from 3 to 1 and increasing spam recall from 98.91% to 99.64% on the locked test set. Promoted to `final_spam_classifier_v2.joblib`.
3. **Phase 8.3–8.6**: Evaluated Word n-grams, Character TF-IDF, Combined Word+Char representations, and Decision Threshold shifts. All alternative candidates were rejected because they either degraded recall (Task 8.4/8.5) or caused catastrophic false positive inflation (Task 8.6).
4. **Final Model Freeze**: `models/final_spam_classifier_v2.joblib` with threshold `0.0` officially frozen as the immutable production model.

---

## 3. Dataset & Split Specifications
- **Data Source**: Enron / Kaggle Spam Email Dataset (`data/raw/internship.csv`)
- **Raw Records**: 5,728 (4,360 Ham, 1,368 Spam)
- **Cleaned Records**: 5,695 (4,327 Ham, 1,368 Spam) — 33 exact duplicate ham emails removed in Phase 2
- **Data Split**: Stratified 80 / 20 split (`random_state=42`)
  - **Training Partition**: 4,556 samples (3,462 Ham [75.99%], 1,094 Spam [24.01%])
  - **Locked Test Partition**: 1,139 samples (865 Ham [75.94%], 274 Spam [24.06%])

---

## 4. Input Preprocessing & Feature Engineering
- **Canonical Preprocessing Pipeline (`src/preprocess.py`)**:
  - `Subject:` prefix removal (`^\s*subject\s*:\s*`)
  - Lowercase normalization
  - Email address replacement (`emailtoken`)
  - URL / domain replacement (`urltoken`)
  - Numeric sequence replacement (`numtoken`)
  - Whitespace collapse and normalization
  - Punctuation and stop words preserved
- **Feature Representation**: Word-level TF-IDF (`TfidfVectorizer`)
  - `analyzer`: `word`
  - `ngram_range`: `(1, 2)` (Unigrams + Bigrams)
  - `sublinear_tf`: `True`
  - `min_df`: `2`
  - `max_df`: `0.95`
  - **Fitted Vocabulary Size**: 121,288 features

---

## 5. Official Locked Test Set Evaluation (1,139 Unseen Emails)

| Metric | Official Score | Recalculated Mathematical Value |
| :--- | :---: | :---: |
| **Accuracy** | **99.74%** | $1136 / 1139 = 99.7366\%$ |
| **Spam Precision** | **99.27%** | $273 / 275 = 99.2727\%$ |
| **Spam Recall** | **99.64%** | $273 / 274 = 99.6350\%$ |
| **Spam F1-Score** | **0.9945** | $0.994535$ |
| **Macro F1-Score** | **0.9964** | $0.996417$ |

### Confusion Matrix Breakdown
- **True Negatives (TN)**: 863
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 1
- **True Positives (TP)**: 273
- **Total Test Samples**: 1,139

---

## 6. Inference Pipeline & Decision Rule
For any input email string:
1. Pass raw text through `canonical_preprocess(text)`.
2. Compute sparse TF-IDF feature vector: `X = vectorizer.transform([cleaned_text])`.
3. Compute signed margin distance: `score = model.decision_function(X)[0]`.
4. Apply decision threshold: $\hat{y} = \text{Spam (1)}$ if $\text{score} \ge 0.0$ else $\text{Ham (0)}$.
5. *Note: LinearSVC decision scores represent geometric signed distances, not calibrated probabilities.*

---

## 7. Known Limitations & Failure Modes
1. **Promotional & Ambiguous Email Sensitivity**: In Phase 7 manual qualitative testing on 24 unseen edge cases, the model achieved 83.33% qualitative accuracy (20/24 correct). The 4 errors occurred on heavily promotional and ambiguous boundary cases (marketing newsletters and borderline solicitations). This limitation is inherent to boundary ambiguity and is documented.
2. **Short / Low-Signal Emails**: Extremely short emails (e.g. 1–2 generic words) lack sufficient n-gram tokens and default to safe legitimate classification ($\text{score} < 0.0$).
3. **Dataset Domain Specificity**: The model was trained and evaluated on Enron/Kaggle email distributions. Performance in modern production environments requires standard mail filtering defense-in-depth (SPF/DKIM checks, IP reputation, sender rate-limiting).
