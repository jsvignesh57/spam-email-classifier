# Spam Email Classifier — Machine Learning System & Web Application

An end-to-end, production-grade Machine Learning classification system and web application for automated, high-precision detection of spam emails.

> [!IMPORTANT]
> **Production Status: FROZEN**: The machine learning model training, tuning, feature engineering, and preprocessing pipelines are 100% complete and immutable. The web application layer acts as a pure, zero-persistence **inference client** wrapping the production artifacts.

---

```
┌────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM FLOW                               │
│                                                                        │
│   User / Web UI                                                        │
│        │                                                               │
│        ▼ (JSON Request: email text)                                    │
│   FastAPI Server (app/main.py)                                         │
│        │                                                               │
│        ▼                                                               │
│   Canonical Preprocessing (src/preprocess.py -> normalize_text())     │
│        │                                                               │
│        ▼                                                               │
│   TF-IDF Transformation (models/tfidf_vectorizer.joblib)               │
│        │                                                               │
│        ▼                                                               │
│   Production Classifier (models/final_spam_classifier_v2.joblib)       │
│        │                                                               │
│        ▼                                                               │
│   Decision Score & Boundary Evaluation (threshold = 0.0)               │
│        │                                                               │
│        ▼                                                               │
│   Output Result: SPAM (>= 0.0) or NOT SPAM (< 0.0)                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

# PART A: Machine Learning Pipeline (Frozen)

## 1. Problem Statement & Objectives
Spam and phishing emails degrade productivity, compromise credentials, and threaten infrastructure. The machine learning goal is to achieve near-zero false negatives (maximizing **Spam Recall**) while strictly constraining false positives (maintaining **Spam Precision** above 99.0%) on high-dimensional text representations.

## 2. Dataset & Partitioning
- **Dataset**: Kaggle / Enron Spam Dataset (`data/raw/internship.csv`).
- **Raw Records**: 5,728 emails (4,360 Ham, 1,368 Spam).
- **Cleaned Records**: 5,695 emails (33 exact duplicates removed in Phase 2).
- **Split Strategy**: 80/20 Stratified Partition (`random_state=42`, strictly locked):
  - **Training Partition**: 4,556 samples (3,462 Ham, 1,094 Spam).
  - **Locked Test Partition**: 1,139 samples (865 Ham, 274 Spam).

## 3. Canonical Text Preprocessing (`src/preprocess.py`)
All text normalization is handled by `normalize_text()`:
1. Strips leading `Subject:` headers.
2. Normalizes email addresses to `emailtoken`.
3. Normalizes web URLs to `urltoken`.
4. Normalizes numeric sequences to `numtoken`.
5. Converts all characters to lowercase.
6. Collapses redundant whitespace into single spaces.

## 4. TF-IDF Feature Engineering (`models/tfidf_vectorizer.joblib`)
- **Analyzer**: Word-level unigrams and bigrams (`ngram_range=(1, 2)`).
- **Sublinear TF**: Enabled (`sublinear_tf=True`) to dampen repeated spam keywords.
- **Frequency Filtering**: `min_df=2`, `max_df=0.95`.
- **Vocabulary Size**: Exactly **121,288 features**.

## 5. Production Model Specification (`models/final_spam_classifier_v2.joblib`)
- **Classifier**: `sklearn.svm.LinearSVC`
- **Hyperparameters**: $C = 10.0$, $\text{loss} = \text{'squared\_hinge'}$, $\text{random\_state} = 42$.
- **Decision Threshold**: $\tau = 0.0$ (Score $\ge 0.0 \rightarrow \text{Spam}$, Score $< 0.0 \rightarrow \text{Not Spam}$).
- **Status**: Production Frozen.

## 6. Official Test Set Performance Benchmark

| Metric | Score | Recalculated Fraction | Evaluation Details |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **99.74%** | $1136 / 1139$ | 1,136 of 1,139 test emails correctly classified |
| **Spam Recall** | **99.64%** | $273 / 274$ | Caught 273 out of 274 test spam emails (1 FN) |
| **Spam Precision** | **99.27%** | $273 / 275$ | Only 2 legitimate emails flagged (2 FP) |
| **Spam F1-Score** | **0.9945** | $0.994535$ | Harmonic balance between Precision & Recall |
| **Macro F1-Score** | **0.9964** | $0.996417$ | Unweighted macro average |

```
                       CONFUSION MATRIX (LOCKED TEST SET)
                       PREDICTED HAM        PREDICTED SPAM
ACTUAL HAM (865)          863 (TN)               2 (FP)
ACTUAL SPAM (274)           1 (FN)             273 (TP)
```

---

## 7. Phase 8 Controlled Improvement Experiments Summary

| Subtask | Focus Area | Tested Configurations | Finding | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **8.1 Error Analysis** | Root cause diagnostics | Historical 3 FN, 2 FP cases | Boundary proximity and corporate overlap | Completed diagnostics |
| **8.2 SVM C Tuning** | Regularization strength | $C \in [0.01, 100.0]$ | $C = 10.0$ maximized recall (99.64%, 1 FN) | **PROMOTED (v2)** |
| **8.3 Word N-grams** | Contextual depth | $(1,1), (1,2), (1,3)$ | $(1,2)$ provides optimal accuracy/feature ratio | **RETAINED $(1,2)$** |
| **8.4 Char TF-IDF** | Sub-word character grams | Char $(3,5), (3,6), (4,7)$ | Degraded Spam Recall to 98.91% (3 FN) | **REJECTED** |
| **8.5 Combined TF-IDF** | Word + Char stacking | Word $(1,2)$ + Char $(3,5)$ | Identical recall, 3.6x larger artifact, 2.4x latency | **REJECTED** |
| **8.6 Threshold Sweep** | Boundary shifting | $\tau \in [-1.0, +1.0]$ | $\tau = -0.75$ triggered 43.5x FP surge (87 FP) | **REJECTED ($\tau=0.0$)** |

---

## 8. Phase 9 Real-World HAM Robustness Evaluation

A separate, controlled robustness test on **80 legitimate real-world emails** across 16 categories:
- **Real-World HAM Accuracy**: **86.25% (69 / 80 correct)**
- **Real-World False Positive Rate**: **13.75% (11 / 80 FP)**
- **100% Accuracy Domains**: Personal, University, Work/Project, Calendar, Technical Newsletters, Announcements, Short emails, URL-containing, Numeric-heavy.
- **Boundary Vulnerability**: 63.6% of false positives cluster in the narrow near-boundary interval $[0.00, +0.25]$, concentrated in marketing-adjacent categories (Welcome emails, Promotional discounts, Account alerts, Subscription billing).

---

# PART B: Web Application (Inference Client)

The Web Application provides an interactive, beginner-friendly yet technically robust interface for classifying email messages in real time.

## 1. Application Architecture & Technology Stack
- **Backend Framework**: Python 3.10+, **FastAPI** (`0.141.1`), **Uvicorn** (`0.52.3`).
- **ML Runtime**: `scikit-learn` (`1.9.0`), `joblib` (`1.5.3`), `numpy` (`2.5.2`), `pandas` (`3.0.5`).
- **Frontend**: Semantic **HTML5**, Modern **Vanilla CSS** (Custom Slate/Indigo theme, glassmorphism, responsive grid), **Vanilla JavaScript** (Fetch API, zero frameworks).
- **Inference Integration**: Strictly imports `normalize_text()` directly from `src/preprocess.py` and loads frozen joblib artifacts in-memory on application startup.

## 2. Key Features
- **Instant Real-Time Classification**: Classifies emails within milliseconds.
- **Strict Single Preprocessing Pipeline**: Backend reuses `src/preprocess.py` without code duplication.
- **Honest Margin-Aware Descriptions**: LinearSVC decision scores are explicitly reported as signed margin values with boundary proximity warnings, not mislabelled probabilities.
- **Quick Sample Testing**: One-click chips for loading realistic Spam and Legitimate email test cases.
- **Zero-Storage Privacy Guarantee**: Submitted emails are processed in-memory and **never** written to database, disk, or logs.
- **Safe DOM APIs**: Strictly uses `textContent` to eliminate XSS vulnerabilities.
- **Responsive & Accessible**: Fully usable across mobile (320px), tablet (768px), and desktop (1366px+), complete with `aria-live` screen-reader announcements.

---

## 3. Installation & Setup

### Step 1: Clone or Navigate to the Workspace
```bash
cd "Spam email - ML model"
```

### Step 2: Create and Activate Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On macOS / Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Running the Web Application

### Start the Uvicorn Server:
```bash
uvicorn app.main:app --reload --port 8000
```

### Open the Application:
- **Web Interface**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc Docs**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 5. REST API Reference

### Health Check: `GET /health`
Returns operational status and active model metadata.

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "model": "LinearSVC",
  "model_version": "v2.0.0",
  "decision_threshold": 0.0,
  "max_input_length": 50000
}
```

---

### Email Classification: `POST /predict`
Evaluates submitted email against the frozen decision boundary.

**Request Headers**: `Content-Type: application/json`

**Request Body**:
```json
{
  "email": "Subject: Congratulations! You won a $10,000 prize. Click http://claim.example.com to claim your reward."
}
```

**Response (200 OK — Spam)**:
```json
{
  "prediction": "SPAM",
  "label": 1,
  "decision_score": 0.8162
}
```

**Response (200 OK — Not Spam)**:
```json
{
  "prediction": "NOT SPAM",
  "label": 0,
  "decision_score": -0.8102
}
```

**Validation Error Response (422 Unprocessable Content)**:
```json
{
  "detail": "Email text cannot be empty or whitespace-only."
}
```

---

### Example cURL Commands

**Classify Spam Email:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"email": "URGENT: Claim your $5,000 lottery winnings at http://free-prize.example.com"}'
```

**Classify Legitimate Email:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"email": "Hi team, please find attached the agenda for our project sync tomorrow at 10 AM."}'
```

---

## 6. Running Automated Tests

Run the complete unit and discovery test suites:

```bash
# Automated discovery tests
python -m unittest discover -s tests -v

# Live browser acceptance and regression tests
python tests/test_browser_acceptance.py
```

All test cases verify end-to-end functionality without mutating any model or dataset files.

---

## 7. Project Structure

```
Spam email - ML model/
│
├── data/
│   ├── raw/
│   │   └── internship.csv                      # Raw Kaggle/Enron dataset (read-only)
│   └── processed/
│       ├── cleaned_internship.csv              # Deduplicated cleaned dataset
│       ├── manual_test_cases.csv               # Unseen qualitative test cases
│       ├── phase_8_error_cases.csv             # Diagnostic error cases
│       └── train_test_split.npz                # Stratified 80/20 train/test split
│
├── models/
│   ├── final_spam_classifier_v2.joblib         # [PRODUCTION FROZEN] LinearSVC (C=10.0)
│   ├── tfidf_vectorizer.joblib                 # [PRODUCTION FROZEN] Fitted Word TF-IDF
│   ├── model_metadata.json                     # Complete model lineage & metrics
│   └── [historical candidate model checkpoints...]
│
├── src/
│   ├── preprocess.py                           # [CANONICAL] Data cleaning & normalize_text()
│   ├── feature_engineering.py                  # TF-IDF extraction & train/test splitting
│   ├── train_models.py                         # Baseline candidate model training
│   ├── evaluate_models.py                      # Model evaluation
│   ├── select_final_model.py                   # Model selection & initial packaging
│   ├── test_classifier.py                      # Qualitative test harness
│   ├── error_analysis.py                       # Error analysis
│   ├── tune_svm_c.py                           # SVM regularization optimization
│   ├── verify_svm_c10.py                       # Promotion verification
│   ├── experiment_tfidf_ngrams.py              # Word n-gram experiment
│   ├── experiment_char_tfidf.py                # Character TF-IDF experiment
│   ├── experiment_combined_tfidf.py            # Combined TF-IDF experiment
│   ├── analyze_decision_threshold.py           # Threshold analysis
│   ├── phase_9_ham_robustness_audit.py         # Phase 9 HAM robustness engine
│   └── final_project_audit.py                  # Final project audit script
│
├── app/
│   ├── __init__.py                             # Application package init
│   ├── main.py                                 # FastAPI backend & inference endpoints
│   ├── templates/
│   │   └── index.html                          # Single-page semantic UI template
│   └── static/
│       ├── css/
│       │   └── style.css                       # Responsive UI styles & theme
│       └── js/
│           └── app.js                          # Client fetch logic & DOM rendering
│
├── tests/
│   ├── test_app.py                             # Automated regression test suite
│   └── test_browser_acceptance.py              # End-to-end acceptance runner
│
├── reports/
│   ├── model_card.md                           # Production model card
│   ├── phase_9_final_audit.md                  # Phase 9 robustness audit report
│   ├── phase_9_ham_robustness_report.md        # Category metrics & score breakdown
│   ├── phase_9_inference_consistency_report.md # Inference parity report
│   ├── phase_9_real_world_ham_robustness.csv   # 80-sample evaluation dataset
│   ├── phase_10_final_project_audit.md         # Final comprehensive pre-submission audit
│   └── [historical experiment reports & charts...]
│
├── requirements.txt                            # Verified dependencies
├── README.md                                   # Project documentation (Part A & B)
└── .gitignore                                  # Git ignore definitions
```

---

## 8. Artifact Integrity Verification

The SHA-256 hashes of all frozen production assets are verified before and after application execution:

| Artifact | SHA-256 Checksum | Status |
| :--- | :--- | :---: |
| `models/final_spam_classifier_v2.joblib` | `daaf9e4ae4f92ea688f8ae589518431414df67692b698d5d2c7bdeff2a8fc10b` | **VERIFIED UNCHANGED** |
| `models/tfidf_vectorizer.joblib` | `4db48a627fd8588a6cb9f09dfbbed54a3b7376c8508bce7fcc1fea86119688f2` | **VERIFIED UNCHANGED** |
| `data/raw/internship.csv` | `a5d0d75d15d370ef2dd3229b30204b18deab6d3fd249206e89a2c58f901bcc77` | **VERIFIED UNCHANGED** |
| `data/processed/cleaned_internship.csv` | `72de2b54b15db7eeb249cf00043c95e460b109a3077fc20816fb625db8be5e8d` | **VERIFIED UNCHANGED** |
| `data/processed/train_test_split.npz` | `1fd473c855032c872b6096e0ce19dde21c7ed861630a9716562764ba415a7fbf` | **VERIFIED UNCHANGED** |
