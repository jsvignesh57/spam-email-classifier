# Spam Email Classifier — Inference Pipeline Architecture

## 1. Overview & Pipeline Flowchart

The Spam Email Classifier inference pipeline accepts raw, unclassified email text, transforms it using the canonical preprocessing and TF-IDF feature extraction pipeline, and generates a binary classification decision using the saved production **Linear Support Vector Machine (LinearSVC)** model.

```
       ┌───────────────────────────────┐
       │        NEW RAW EMAIL          │
       └──────────────┬────────────────┘
                      │
                      ▼
       ┌───────────────────────────────┐
       │     INPUT VALIDATION          │
       │ (Null/Empty/Whitespace Check) │
       └──────────────┬────────────────┘
                      │
                      ▼
       ┌───────────────────────────────┐
       │      TEXT PREPROCESSING       │
       │  - Strip 'Subject:' prefix    │
       │  - Normalize URLs -> urltoken │
       │  - Normalize Emails -> emailtoken
       │  - Normalize Numbers -> numtoken
       │  - Lowercasing & Whitespace   │
       └──────────────┬────────────────┘
                      │
                      ▼
       ┌───────────────────────────────┐
       │    SAVED TF-IDF VECTORIZER    │
       │ (121,288 Vocabulary Features) │
       │  * vectorizer.transform() *   │
       └──────────────┬────────────────┘
                      │
                      ▼
       ┌───────────────────────────────┐
       │     SAVED LINEAR SVM MODEL    │
       │   * model.predict() *         │
       │   * model.decision_function()*│
       └──────────────┬────────────────┘
                      │
                      ▼
       ┌───────────────────────────────┐
       │       FINAL PREDICTION        │
       │  0: NOT SPAM / HAM            │
       │  1: SPAM                      │
       └───────────────────────────────┘
```

---

## 2. Step-by-Step Pipeline Mechanics

### Step 1: Input Validation
- **Safeguard**: Prevents malformed, empty, or whitespace-only inputs from silently passing through classification.
- **Handling**:
  - `None` or non-string inputs return: `"ERROR: Email text must be a non-null string."`
  - Empty or whitespace strings return: `"ERROR: Email text cannot be empty or whitespace-only."`
  - Valid string inputs proceed to Step 2.

### Step 2: Canonical Text Preprocessing (`src/preprocess.py`)
To prevent train/inference skew, the inference pipeline strictly reuses the exact same preprocessing logic (`normalize_text`) used during model training:
1. **Header Stripping**: Removes leading `Subject:` prefix patterns while preserving the subject content.
2. **Email Masking**: Replaces email addresses with semantic token `emailtoken`.
3. **URL Masking**: Replaces web domains and URLs with semantic token `urltoken`.
4. **Number Normalization**: Normalizes standalone digit sequences to `numtoken`.
5. **Case Normalization**: Converts all text to lowercase.
6. **Whitespace Normalization**: Collapses repeated spaces, tabs, and newlines into single spaces.

### Step 3: TF-IDF Feature Extraction (`models/tfidf_vectorizer.joblib`)
- The preprocessed text is transformed into a high-dimensional sparse feature vector of length **121,288**.
- **Crucial Rule**: The vectorizer only calls `transform([cleaned_text])`. It **never** calls `fit()` or `fit_transform()`.
- The extracted vector represents n-gram term frequencies (unigrams and bigrams) weighted by their inverse document frequencies computed during training.

### Step 4: Classification via Linear Support Vector Machine (`models/final_spam_classifier.joblib`)
- The feature vector is passed to the saved Linear SVM model.
- **Decision Boundary**: The model computes the signed hyperplane distance using `decision_function(X)`:
  $$\text{Score} = \mathbf{w}^T \mathbf{x} + b$$
- **Decision Rule**:
  - If $\text{Score} > 0 \implies \text{Label } 1 \text{ (SPAM)}$
  - If $\text{Score} \le 0 \implies \text{Label } 0 \text{ (NOT SPAM / HAM)}$

### Step 5: Output Presentation
- **Label `0`**: **NOT SPAM / HAM** — Legitimate email (work, personal, transactional, or academic communication).
- **Label `1`**: **SPAM** — Unsolicited, phishing, scam, or malicious commercial communication.
- **Decision Score**: Signed numerical distance from the separating hyperplane (positive for spam, negative for ham). Note: Linear SVM does not provide calibrated probabilities without Platt scaling.

---

## 3. Python Usage Example

```python
import joblib
from src.preprocess import normalize_text
from src.test_classifier import predict_email

# 1. Load artifacts
model = joblib.load("models/final_spam_classifier.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# 2. Predict on new email
email = "Congratulations! You have won a $10,000 cash prize. Click here to claim."
result = predict_email(email, vectorizer, model)

print(result)
# Output:
# {
#     'is_valid': True,
#     'error': None,
#     'label': 1,
#     'prediction': 'SPAM',
#     'decision_score': 1.8421,
#     'cleaned_text': 'congratulations! you have won a $numtoken prize. click here to claim.'
# }
```

---

## 4. Operational Guardrails

1. **Zero Data Leakage**: Vectorizer vocabulary and IDF weights are immutable and fixed at 121,288 features.
2. **Artifact Immutability**: All original training datasets, intermediate splits, and models remain untouched.
3. **No Retraining**: All inference relies purely on pre-serialized model weights ($C=1.0$, `loss='squared_hinge'`, `random_state=42`).
