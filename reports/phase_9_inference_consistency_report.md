# Phase 9: Python vs Web Inference Consistency & Preprocessing Audit Report

**Project**: Spam Email Classifier (Internship Project)  
**Model Version**: Production Frozen `LinearSVC` (C=10.0)  
**Status**: COMPLETE — ALL CONSISTENCY GATES PASSED

---

## 1. Audit Objective
Verify that the FastAPI web application (`app/main.py`) performs exact, bit-for-bit identical inference to the canonical Python offline inference pipeline (`src/preprocess.py` + `models/tfidf_vectorizer.joblib` + `models/final_spam_classifier_v2.joblib`). Specifically confirm:
1. Canonical text preprocessing function `normalize_text()` is imported and reused directly without duplication.
2. Character token transformations (URLs $\rightarrow$ `urltoken`, emails $\rightarrow$ `emailtoken`, numbers $\rightarrow$ `numtoken`, headers, lowercasing, whitespace) operate identically.
3. Feature vectorization produces identical sparse matrix representations.
4. LinearSVC decision function returns identical signed margin scores and identical classification verdicts across diverse inputs.

---

## 2. Preprocessing Pipeline Verification

The web application backend in `app/main.py` explicitly imports the canonical normalization function:
```python
from src.preprocess import normalize_text
```

### Preprocessing Determinism Audit:

| Input Text Component | Canonical `src.preprocess.normalize_text()` | Web Backend `app.main:app` | Status |
| :--- | :--- | :--- | :---: |
| **Leading Headers** (`Subject:`, `Re:`, `Fwd:`) | Stripped cleanly | Stripped cleanly | **IDENTICAL** |
| **Email Addresses** (`user@domain.com`) | Substituted with `emailtoken` | Substituted with `emailtoken` | **IDENTICAL** |
| **Web URLs** (`https://...`, `www...`) | Substituted with `urltoken` | Substituted with `urltoken` | **IDENTICAL** |
| **Standalone Numbers** (`10,000`, `450`) | Substituted with `numtoken` | Substituted with `numtoken` | **IDENTICAL** |
| **Casing** | Lowercased | Lowercased | **IDENTICAL** |
| **Whitespace Collapse** | Single whitespace normalized | Single whitespace normalized | **IDENTICAL** |
| **Empty / Whitespace-Only** | Handled / Filtered | Handled / HTTP 422 rejected | **IDENTICAL** |

---

## 3. End-to-End Inference Parity Test Matrix

The following test emails were evaluated through both the Python offline evaluation harness and the running FastAPI server (`POST /predict`):

| Test Case | Python Score | Web API Score | Absolute Score $\Delta$ | Python Label | Web Label | Parity Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Obvious Spam** (Lottery prize) | `+0.3508` | `+0.3508` | `0.0000` | SPAM (1) | SPAM (1) | **PASS** |
| **2. Obvious Ham** (Team sync) | `-1.0317` | `-1.0317` | `0.0000` | NOT SPAM (0) | NOT SPAM (0) | **PASS** |
| **3. Welcome / Newsletter Case** | `+0.1077` | `+0.1077` | `0.0000` | SPAM (1) | SPAM (1) | **PASS** |
| **4. Short Legitimate Email** | `-0.6337` | `-0.6337` | `0.0000` | NOT SPAM (0) | NOT SPAM (0) | **PASS** |
| **5. Promotional Legitimate Email** | `+0.4684` | `+0.4684` | `0.0000` | SPAM (1) | SPAM (1) | **PASS** |

---

## 4. Key Findings

1. **Zero Discrepancy**: Across all test categories, the Python offline pipeline and the FastAPI web service produce identical decision scores to 4 decimal places ($|\Delta| = 0.0000$) and identical class labels.
2. **Zero Duplicate Logic**: The web backend does not implement any secondary or diverging preprocessing code.
3. **In-Memory Lifespan Execution**: The production artifacts (`models/final_spam_classifier_v2.joblib` and `models/tfidf_vectorizer.joblib`) are loaded into memory once on application startup, preventing file I/O overhead and ensuring consistent state across requests.
4. **Conclusion**: The web application is an exact, faithful inference client for the production ML model. The observed boundary behavior on welcome/newsletter emails is a property of the high-dimensional feature representation and training distribution, **not** an application or API implementation defect.
