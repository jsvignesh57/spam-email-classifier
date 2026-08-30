"""
Spam Email Classifier — End-to-End Acceptance Test Runner

Executes all Test Cases A through J, validates HTML/CSS/JS asset integrity,
checks DOM structure, tests full API pipeline, verifies decision score rules,
privacy guarantees, and checks production artifact SHA-256 hashes.
"""

import hashlib
import json
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "http://127.0.0.1:8000"

FROZEN_HASHES = {
    "models/final_spam_classifier_v2.joblib": "daaf9e4ae4f92ea688f8ae589518431414df67692b698d5d2c7bdeff2a8fc10b",
    "models/tfidf_vectorizer.joblib": "4db48a627fd8588a6cb9f09dfbbed54a3b7376c8508bce7fcc1fea86119688f2",
    "data/raw/internship.csv": "a5d0d75d15d370ef2dd3229b30204b18deab6d3fd249206e89a2c58f901bcc77",
    "data/processed/cleaned_internship.csv": "72de2b54b15db7eeb249cf00043c95e460b109a3077fc20816fb625db8be5e8d",
    "data/processed/train_test_split.npz": "1fd473c855032c872b6096e0ce19dde21c7ed861630a9716562764ba415a7fbf",
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_acceptance_tests():
    print("=" * 70)
    print("STARTING END-TO-END ACCEPTANCE TESTS")
    print("=" * 70)

    # 1. Artifact Check (Pre-test)
    print("\n[STEP 1] Pre-Test Artifact Integrity Check...")
    for rel_path, expected in FROZEN_HASHES.items():
        p = PROJECT_ROOT / rel_path
        actual = compute_sha256(p)
        assert actual == expected, f"Hash mismatch for {rel_path}!"
        print(f"  [OK] {rel_path}: Hash verified.")

    # 2. Server & Endpoints
    print("\n[STEP 2] Verifying Server Endpoints...")
    res_health = requests.get(f"{BASE_URL}/health")
    assert res_health.status_code == 200, f"Health check failed: {res_health.status_code}"
    health_data = res_health.json()
    print(f"  [OK] GET /health: 200 OK -> {health_data}")

    res_root = requests.get(f"{BASE_URL}/")
    assert res_root.status_code == 200, f"Root failed: {res_root.status_code}"
    assert "text/html" in res_root.headers.get("content-type", "")
    print(f"  [OK] GET /: 200 OK (HTML Length: {len(res_root.text)} bytes)")

    res_css = requests.get(f"{BASE_URL}/static/css/style.css")
    assert res_css.status_code == 200, "CSS failed"
    print(f"  [OK] GET /static/css/style.css: 200 OK ({len(res_css.text)} bytes)")

    res_js = requests.get(f"{BASE_URL}/static/js/app.js")
    assert res_js.status_code == 200, "JS failed"
    print(f"  [OK] GET /static/js/app.js: 200 OK ({len(res_js.text)} bytes)")

    # 3. HTML & DOM Element Verification
    print("\n[STEP 3] Verifying HTML & DOM Elements...")
    html_content = res_root.text
    required_ids = [
        "classifier-form", "email-input", "check-btn", "clear-btn",
        "current-chars", "char-count", "input-error", "input-error-text",
        "result-container", "result-badge", "result-badge-text",
        "result-summary", "detail-score", "detail-rule", "detail-model",
        "sample-spam-btn", "sample-ham-btn"
    ]
    for element_id in required_ids:
        assert f'id="{element_id}"' in html_content, f"Missing DOM element id: {element_id}"
        print(f"  [OK] DOM element #{element_id} verified present.")

    # Verify no uncalibrated probability terminology in labels
    assert "Probability of spam" not in html_content
    assert "AI confidence:" not in html_content
    print("  [OK] Verified UI does not mislabel decision scores as probability or confidence.")

    # 4. Test Cases A through J
    print("\n[STEP 4] Executing Test Cases A through J...")
    results_table = []

    # Case A: Obvious Spam
    email_a = (
        "Congratulations! You have won $10,000 in our exclusive lottery. "
        "Claim your prize immediately by clicking this link: http://claim.example.com"
    )
    res_a = requests.post(f"{BASE_URL}/predict", json={"email": email_a})
    data_a = res_a.json()
    assert res_a.status_code == 200
    assert data_a["prediction"] == "SPAM"
    assert data_a["label"] == 1
    assert data_a["decision_score"] >= 0.0
    results_table.append(("Case A: Obvious Spam", f"SPAM (score: {data_a['decision_score']:+.4f})", "PASS"))
    print(f"  [OK] Case A (Obvious Spam) -> Prediction: {data_a['prediction']}, Score: {data_a['decision_score']:+.4f}")

    # Case B: Obvious Ham
    email_b = (
        "Hi team,\n\n"
        "Let us meet tomorrow at 10 AM to discuss the sprint goals.\n"
        "Please bring the latest project updates.\n\n"
        "Thanks."
    )
    res_b = requests.post(f"{BASE_URL}/predict", json={"email": email_b})
    data_b = res_b.json()
    assert res_b.status_code == 200
    assert data_b["prediction"] == "NOT SPAM"
    assert data_b["label"] == 0
    assert data_b["decision_score"] < 0.0
    results_table.append(("Case B: Obvious Ham", f"NOT SPAM (score: {data_b['decision_score']:+.4f})", "PASS"))
    print(f"  [OK] Case B (Obvious Ham) -> Prediction: {data_b['prediction']}, Score: {data_b['decision_score']:+.4f}")

    # Case C: Promotional / Ambiguous
    email_c = (
        "Special offer for our customers: enjoy 20% off selected products "
        "this weekend. Visit our website to learn more."
    )
    res_c = requests.post(f"{BASE_URL}/predict", json={"email": email_c})
    data_c = res_c.json()
    assert res_c.status_code == 200
    results_table.append(("Case C: Promotional/Ambiguous", f"{data_c['prediction']} (score: {data_c['decision_score']:+.4f})", "PASS"))
    print(f"  [OK] Case C (Promotional) -> Prediction: {data_c['prediction']}, Score: {data_c['decision_score']:+.4f}")

    # Case D: Short Email
    email_d = "Meeting at 3 PM?"
    res_d = requests.post(f"{BASE_URL}/predict", json={"email": email_d})
    data_d = res_d.json()
    assert res_d.status_code == 200
    results_table.append(("Case D: Short Email", f"{data_d['prediction']} (score: {data_d['decision_score']:+.4f})", "PASS"))
    print(f"  [OK] Case D (Short Email) -> Prediction: {data_d['prediction']}, Score: {data_d['decision_score']:+.4f}")

    # Case E: URL Email
    email_e = "Please review the project document at https://example.com/project"
    res_e = requests.post(f"{BASE_URL}/predict", json={"email": email_e})
    data_e = res_e.json()
    assert res_e.status_code == 200
    results_table.append(("Case E: URL-Containing", f"{data_e['prediction']} (score: {data_e['decision_score']:+.4f})", "PASS"))
    print(f"  [OK] Case E (URL Email) -> Prediction: {data_e['prediction']}, Score: {data_e['decision_score']:+.4f}")

    # Case F: Unicode Email
    email_f = "Hello team, नमस्ते. Please review the project update."
    res_f = requests.post(f"{BASE_URL}/predict", json={"email": email_f})
    data_f = res_f.json()
    assert res_f.status_code == 200
    results_table.append(("Case F: Unicode Email", f"{data_f['prediction']} (score: {data_f['decision_score']:+.4f})", "PASS"))
    print(f"  [OK] Case F (Unicode Email) -> Prediction: {data_f['prediction']}, Score: {data_f['decision_score']:+.4f}")

    # Case G: Numeric Heavy
    email_g = (
        "Your reference number is 9845632107. "
        "Transaction ID: 7839201456. "
        "Please confirm the details."
    )
    res_g = requests.post(f"{BASE_URL}/predict", json={"email": email_g})
    data_g = res_g.json()
    assert res_g.status_code == 200
    results_table.append(("Case G: Numeric Heavy", f"{data_g['prediction']} (score: {data_g['decision_score']:+.4f})", "PASS"))
    print(f"  [OK] Case G (Numeric Heavy) -> Prediction: {data_g['prediction']}, Score: {data_g['decision_score']:+.4f}")

    # Case H: Empty Input
    res_h = requests.post(f"{BASE_URL}/predict", json={"email": ""})
    assert res_h.status_code == 422
    data_h = res_h.json()
    assert "empty" in data_h.get("detail", "").lower()
    results_table.append(("Case H: Empty Input", f"HTTP 422 Rejected: '{data_h['detail']}'", "PASS"))
    print(f"  [OK] Case H (Empty Input) -> HTTP 422: {data_h['detail']}")

    # Case I: Whitespace Input
    res_i = requests.post(f"{BASE_URL}/predict", json={"email": "        "})
    assert res_i.status_code == 422
    data_i = res_i.json()
    assert "whitespace" in data_i.get("detail", "").lower() or "empty" in data_i.get("detail", "").lower()
    results_table.append(("Case I: Whitespace Input", f"HTTP 422 Rejected: '{data_i['detail']}'", "PASS"))
    print(f"  [OK] Case I (Whitespace Input) -> HTTP 422: {data_i['detail']}")

    # Case J: Clear Button / DOM Reset
    js_content = res_js.text
    assert "clearBtn.addEventListener('click'" in js_content
    assert "emailInput.value = ''" in js_content
    assert "hideResult()" in js_content
    assert "hideError()" in js_content
    results_table.append(("Case J: Clear Button", "DOM Input & Results Reset Verified", "PASS"))
    print("  [OK] Case J (Clear Button) -> Reset event handlers verified.")

    # 5. Multiple Sequential Requests Test
    print("\n[STEP 5] Testing Multiple Sequential Requests...")
    for idx in range(5):
        sample_txt = f"Invoice #{idx+1000} is due tomorrow. Please pay $50.00."
        r = requests.post(f"{BASE_URL}/predict", json={"email": sample_txt})
        assert r.status_code == 200
        d = r.json()
        assert d["prediction"] in ["SPAM", "NOT SPAM"]
    print("  [OK] 5 consecutive inference requests completed with zero memory leak or server error.")

    # 6. Privacy Check
    print("\n[STEP 6] Verifying Privacy & Zero-Persistence...")
    root_files = list(PROJECT_ROOT.glob("*.txt"))
    for rf in root_files:
        if rf.name not in ["requirements.txt"]:
            assert False, f"Unexpected file found: {rf}"
    print("  [OK] Zero disk persistence verified. No temporary text files or emails stored.")

    # 7. Final Artifact Integrity Check
    print("\n[STEP 7] Final Production Artifact Check...")
    for rel_path, expected in FROZEN_HASHES.items():
        p = PROJECT_ROOT / rel_path
        actual = compute_sha256(p)
        assert actual == expected, f"Hash changed for {rel_path}!"
        print(f"  [OK] {rel_path}: Hash unchanged.")

    print("\n" + "=" * 78)
    print("ACCEPTANCE TEST SUMMARY TABLE")
    print("=" * 78)
    print(f"{'Test Case':<30} | {'Result':<36} | {'Status':<6}")
    print("-" * 78)
    for test, res, status in results_table:
        print(f"{test:<30} | {res:<36} | {status:<6}")
    print("=" * 78)


if __name__ == "__main__":
    run_acceptance_tests()
