# Phase 9: Real-World Legitimate (HAM) Robustness & False-Positive Audit Report

**Project**: Spam Email Classifier — Machine Learning System  
**Model Architecture**: Linear Support Vector Machine (`LinearSVC`, $C=10.0$, `loss='squared_hinge'`, $\text{threshold} = 0.0$)  
**Feature Extraction**: Word-level TF-IDF ($(1, 2)$ n-grams, $\text{sublinear\_tf} = \text{True}$, 121,288 features)  
**Status**: COMPLETE — EMPIRICAL ROBUSTNESS EVALUATION

---

## 1. Executive Summary & Benchmark Distinction

> [!IMPORTANT]
> **Strict Benchmark Distinction**:
> - **Locked Test Set Performance (Phase 5 / Task 8.2)**: Evaluated on the frozen, held-out 1,139-sample test partition (865 Ham, 274 Spam). Results: **99.74% Accuracy**, **99.64% Spam Recall (1 FN)**, **99.27% Spam Precision (2 FP)**, **0.9945 Spam F1**.
> - **Real-World HAM Robustness Test Set (Phase 9)**: Evaluated on a new, diverse collection of **80 legitimate HAM emails** across 16 real-world categories (5 samples per category). Results: **86.25% Ham Accuracy (69/80)**, **13.75% False Positive Rate (11/80 FP)**.

The audit demonstrates that the model maintains **100% precision (0% FPR)** across standard interpersonal, academic, corporate, calendar, and technical communications. However, false positives cluster in marketing-adjacent categories (Welcome emails, Promotional offers, Account alerts, and Subscription billing) where vocabulary heavily overlaps with spam training markers. Crucially, **63.6% of all false positives lie immediately adjacent to the decision boundary in the $[0.00, +0.25]$ margin window**.

---

## 2. Real-World HAM Category Breakdown

The 80 test emails span 16 realistic categories with synthetic, redacted personal data:

| Category | Samples | Correct HAM | False Positives (FP) | False Positive Rate (%) | Mean Decision Score | Min Score | Max Score | Vulnerability Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A. Personal Communication** | 5 | 5 | 0 | **0.0%** | `-0.4214` | `-0.73` | `-0.10` | Robust (Safe) |
| **B. College / University** | 5 | 5 | 0 | **0.0%** | `-0.6581` | `-0.70` | `-0.54` | Highly Robust (Safe) |
| **C. Work / Project** | 5 | 5 | 0 | **0.0%** | `-0.5619` | `-0.81` | `-0.39` | Highly Robust (Safe) |
| **D. Meeting / Calendar** | 5 | 5 | 0 | **0.0%** | `-0.6584` | `-0.85` | `-0.45` | Highly Robust (Safe) |
| **E. Technical Newsletters** | 5 | 5 | 0 | **0.0%** | `-0.2273` | `-0.35` | `-0.13` | Robust (Safe) |
| **F. Legitimate Welcome Emails** | 5 | 3 | 2 | **40.0%** | `-0.0194` | `-0.57` | `+0.63` | **High Vulnerability** |
| **G. Legitimate Promotional / Discounts** | 5 | 3 | 2 | **40.0%** | `+0.0322` | `-0.02` | `+0.18` | **High Vulnerability** |
| **H. Transactional Notifications** | 5 | 5 | 0 | **0.0%** | `-0.2534` | `-0.45` | `-0.01` | Robust (Safe) |
| **I. Account Notifications** | 5 | 3 | 2 | **40.0%** | `-0.0050` | `-0.18` | `+0.27` | **High Vulnerability** |
| **J. Delivery / Order Notifications** | 5 | 4 | 1 | **20.0%** | `-0.0883` | `-0.34` | `+0.44` | Moderate Vulnerability |
| **K. Subscription Emails** | 5 | 3 | 2 | **40.0%** | `+0.0123` | `-0.29` | `+0.35` | **High Vulnerability** |
| **L. Informational Announcements** | 5 | 5 | 0 | **0.0%** | `-0.4043` | `-0.82` | `-0.18` | Robust (Safe) |
| **M. Short Legitimate Emails** | 5 | 5 | 0 | **0.0%** | `-0.5085` | `-0.93` | `-0.05` | Robust (Safe) |
| **N. URL-Containing Legitimate** | 5 | 5 | 0 | **0.0%** | `-0.3492` | `-0.60` | `-0.06` | Robust (Safe) |
| **O. Numeric-Heavy Legitimate** | 5 | 5 | 0 | **0.0%** | `-0.3539` | `-0.76` | `-0.15` | Robust (Safe) |
| **P. Unicode-Containing Legitimate** | 5 | 3 | 2 | **40.0%** | `-0.1992` | `-0.65` | `+0.24` | Moderate / Emoji Vulnerability |
| **TOTAL / OVERALL** | **80** | **69** | **11** | **13.75%** | `-0.3340` | **-0.93** | **+0.63** | **86.25% Real-World Accuracy** |

---

## 3. Decision-Score Distribution Analysis

The signed decision scores for all 80 legitimate emails were categorized into 8 margin intervals relative to the $\tau = 0.0$ threshold:

| Decision Score Bucket | Interpretation | Count | Percentage | Cumulative | Visual Density |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **$< -1.00$** | Deep Legitimate Margin | 0 | 0.0% | 0.0% | |
| **$[-1.00, -0.50)$** | Strong Legitimate Margin | 23 | 28.7% | 28.7% | `##############` |
| **$[-0.50, -0.25)$** | Moderate Legitimate Margin | 18 | 22.5% | 51.2% | `###########` |
| **$[-0.25, 0.00)$** | Near-Boundary Legitimate | 28 | 35.0% | 86.2% | `#################` |
| **$[0.00, +0.25)$** | **Near-Boundary False Positive** | **7** | **8.8%** | 95.0% | `####` |
| **$[+0.25, +0.50)$** | Moderate False Positive | 3 | 3.8% | 98.8% | `#` |
| **$[+0.50, +1.00)$** | High False Positive | 1 | 1.2% | 100.0% | |
| **$> +1.00$** | Severe False Positive | 0 | 0.0% | 100.0% | |

### Key Distribution Insights:
1. **69 of 80 (86.25%)** legitimate emails receive negative scores and are correctly filtered as **NOT SPAM**.
2. **51.2%** of legitimate emails reside comfortably in the secure $[-1.00, -0.25]$ region.
3. Of the 11 false positives, **7 (63.6%) are borderline cases** scoring between `+0.00` and `+0.25`.
4. Only a single email scored above `+0.50` (`+0.63`, a welcome email containing multiple generic hosting call-to-action tokens).
5. Zero legitimate emails scored in the severe $> +1.00$ false-positive regime.

---

## 4. Visual Evidence

### Decision Score Distribution
![HAM Decision Score Distribution](file:///d:/Projects/Spam%20email%20-%20ML%20model/reports/phase_9_ham_decision_score_distribution.png)

### Category False Positive Rates
![HAM Category False Positive Rates](file:///d:/Projects/Spam%20email%20-%20ML%20model/reports/phase_9_ham_category_false_positive.png)

---

## 5. Root Cause Analysis of Boundary False Positives

### Feature Attribution Breakdown
Inspecting the feature weights for near-boundary false positives (e.g., welcome and subscription emails scoring $\approx +0.0617$ to $+0.1077$) reveals three primary root causes:
1. **Marketing Call-to-Action N-grams**: Phrases like `"check out"`, `"our"`, `"out our"`, `"have you"`, `"welcome to"` carry positive weights in the linear model because they occur frequently in unsolicited commercial email datasets.
2. **Presence of Normalized URLs (`urltoken`)**: A web link token contributes $+0.1178$ to the decision margin. When combined with promotional phrasing, it is often sufficient to push a welcome or password reset email over the $0.0$ threshold.
3. **Absence of Domain-Specific Ham Lexicon**: Unlike corporate project updates (which contain ham-heavy tokens like `meeting`, `project`, `attached`, `schedule`, `review`), onboarding emails use generic consumer vocabulary with fewer offsetting negative weights.

---

## 6. Scientific Conclusion & Production Guidance

1. **No Application Bug**: The inference engine is functioning exactly as designed and trained.
2. **Boundary Sensitivity**: Real-world consumer onboarding, discount, and notification emails reside close to the hyperplane boundary.
3. **Threshold Stability (Recall Protection Gate)**: Shifting the threshold downward (e.g., to $-0.75$) was previously proven in Phase 8.6 to trigger an unacceptable 43.5x explosion in false positives (87 FP on locked test set). Shifting the threshold upward would degrade Spam Recall below the project's non-negotiable requirement ($99.64\%$).
4. **UI Adaptation**: Rather than modifying the frozen model, the Web UI must honestly reflect boundary proximity by informing users when a score falls within the marginal $[0.0, +0.50]$ range.
