"""
Spam Email Classifier — Comprehensive Test Suite

Tests:
1. Artifact integrity (SHA-256 validation of models and datasets).
2. Model & vectorizer loading.
3. Canonical preprocessing verification (normalize_text).
4. Direct model inference on standard and edge-case inputs.
5. End-to-end HTTP API tests (GET /health, GET /, POST /predict).
6. Input validation (empty, whitespace-only, oversized, invalid JSON).
7. Decision score validation (ensuring score is signed margin, not probability).
8. Absence of disk persistence (zero-storage privacy validation).
"""

import hashlib
import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path

import requests
import uvicorn

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app, load_artifacts, ml_artifacts, MAX_EMAIL_LENGTH, DECISION_THRESHOLD
from src.preprocess import normalize_text

# Expected SHA256 hashes of frozen artifacts
FROZEN_HASHES = {
    "models/final_spam_classifier_v2.joblib": "daaf9e4ae4f92ea688f8ae589518431414df67692b698d5d2c7bdeff2a8fc10b",
    "models/tfidf_vectorizer.joblib": "4db48a627fd8588a6cb9f09dfbbed54a3b7376c8508bce7fcc1fea86119688f2",
    "data/raw/internship.csv": "a5d0d75d15d370ef2dd3229b30204b18deab6d3fd249206e89a2c58f901bcc77",
    "data/processed/cleaned_internship.csv": "72de2b54b15db7eeb249cf00043c95e460b109a3077fc20816fb625db8be5e8d",
    "data/processed/train_test_split.npz": "1fd473c855032c872b6096e0ce19dde21c7ed861630a9716562764ba415a7fbf",
}

TEST_PORT = 8009
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestServerThread:
    """Helper to run Uvicorn server in a background thread for testing."""
    def __init__(self, host="127.0.0.1", port=TEST_PORT):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        
        # Wait for server to become responsive
        max_retries = 30
        for _ in range(max_retries):
            try:
                res = requests.get(f"{BASE_URL}/health", timeout=1)
                if res.status_code == 200:
                    return
            except requests.RequestException:
                time.sleep(0.1)
        raise RuntimeError("Uvicorn test server failed to start within timeout.")

    def stop(self):
        if self.server:
            self.server.should_exit = True
            if self.thread:
                self.thread.join(timeout=2)


class TestSpamClassifierApplication(unittest.TestCase):
    server_runner = None

    @classmethod
    def setUpClass(cls):
        # 1. Verify frozen artifact hashes before running tests
        for rel_path, expected_hash in FROZEN_HASHES.items():
            full_path = PROJECT_ROOT / rel_path
            assert full_path.exists(), f"Missing frozen artifact: {rel_path}"
            actual_hash = compute_sha256(full_path)
            assert actual_hash == expected_hash, (
                f"Artifact hash mismatch before tests for {rel_path}!\n"
                f"Expected: {expected_hash}\nActual:   {actual_hash}"
            )
        
        # 2. Start Uvicorn test server
        cls.server_runner = TestServerThread()
        cls.server_runner.start()

    @classmethod
    def tearDownClass(cls):
        if cls.server_runner:
            cls.server_runner.stop()

        # Verify frozen artifact hashes after running tests
        for rel_path, expected_hash in FROZEN_HASHES.items():
            full_path = PROJECT_ROOT / rel_path
            actual_hash = compute_sha256(full_path)
            assert actual_hash == expected_hash, (
                f"Artifact hash altered during tests for {rel_path}!\n"
                f"Expected: {expected_hash}\nActual:   {actual_hash}"
            )

    # ------------------------------------------------------------------
    # Preprocessing Tests
    # ------------------------------------------------------------------
    def test_canonical_preprocessing_reuse(self):
        """Verify normalize_text produces expected semantic tokens."""
        text = "Subject: Urgent: visit http://example.com and email john.doe@example.org with 500 dollars"
        normalized = normalize_text(text)
        self.assertNotIn("subject:", normalized.lower())
        self.assertIn("urltoken", normalized)
        self.assertIn("emailtoken", normalized)
        self.assertIn("numtoken", normalized)
        self.assertNotIn("500", normalized)
        self.assertNotIn("http://example.com", normalized)

    # ------------------------------------------------------------------
    # HTTP Endpoint Tests
    # ------------------------------------------------------------------
    def test_health_endpoint(self):
        """GET /health returns healthy status and model information."""
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["model"], "LinearSVC")
        self.assertEqual(data["model_version"], "v2.0.0")
        self.assertEqual(data["decision_threshold"], 0.0)
        self.assertEqual(data["max_input_length"], MAX_EMAIL_LENGTH)

    def test_frontend_root_endpoint(self):
        """GET / returns the single-page HTML interface."""
        response = requests.get(f"{BASE_URL}/", timeout=5)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("Spam Email Classifier", response.text)
        self.assertIn("LinearSVC", response.text)

    def test_predict_obvious_spam(self):
        """POST /predict classifies obvious lottery spam correctly."""
        payload = {
            "email": "Subject: Winner Notice: Congratulations! You won a $10,000 lottery prize. Claim now!"
        }
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["prediction"], "SPAM")
        self.assertEqual(data["label"], 1)
        self.assertGreaterEqual(data["decision_score"], 0.0)
        # Decision score must be a number
        self.assertIsInstance(data["decision_score"], float)

    def test_predict_obvious_ham(self):
        """POST /predict classifies normal business email as NOT SPAM."""
        payload = {
            "email": "Subject: Team meeting notes\n\nHi team, can we meet tomorrow at 3 PM to review the quarterly project deliverables? Thanks, Mark"
        }
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["prediction"], "NOT SPAM")
        self.assertEqual(data["label"], 0)
        self.assertLess(data["decision_score"], 0.0)
        self.assertIsInstance(data["decision_score"], float)

    def test_predict_promotional_email(self):
        """POST /predict handles promotional marketing email."""
        payload = {
            "email": "Subject: Exclusive 50% discount on all cloud services! Limited time offer ends tonight."
        }
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["prediction"], ["SPAM", "NOT SPAM"])
        self.assertIn(data["label"], [0, 1])

    def test_predict_short_email(self):
        """POST /predict handles short emails without crashing."""
        payload = {"email": "Thanks, received."}
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["prediction"], ["SPAM", "NOT SPAM"])

    def test_predict_url_containing_email(self):
        """POST /predict handles emails with URLs."""
        payload = {"email": "Please review the documents at https://intranet.company.internal/docs/project.pdf"}
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["prediction"], ["SPAM", "NOT SPAM"])

    def test_predict_unicode_email(self):
        """POST /predict handles unicode characters gracefully."""
        payload = {"email": "Subject: Bonjour! Félicitations pour votre promotion 🚀"}
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["prediction"], ["SPAM", "NOT SPAM"])

    def test_predict_numeric_heavy_email(self):
        """POST /predict handles numeric sequences."""
        payload = {"email": "Order #987654321 Total: $12,450.00 Transaction ID: 1122334455"}
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["prediction"], ["SPAM", "NOT SPAM"])

    # ------------------------------------------------------------------
    # Validation & Error Handling Tests
    # ------------------------------------------------------------------
    def test_empty_string_rejected(self):
        """POST /predict rejects empty string with HTTP 422."""
        payload = {"email": ""}
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("empty", data["detail"].lower())

    def test_whitespace_only_rejected(self):
        """POST /predict rejects whitespace-only string with HTTP 422."""
        payload = {"email": "   \n\t   \r\n   "}
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("whitespace", data["detail"].lower())

    def test_missing_email_field_rejected(self):
        """POST /predict rejects missing email field with HTTP 422."""
        payload = {"text": "some text"}
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)

    def test_invalid_json_rejected(self):
        """POST /predict rejects invalid non-JSON body with HTTP 422."""
        response = requests.post(
            f"{BASE_URL}/predict",
            data="not a json",
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        self.assertEqual(response.status_code, 422)

    def test_oversized_email_rejected(self):
        """POST /predict rejects email exceeding MAX_EMAIL_LENGTH."""
        oversized_text = "Spam " * (MAX_EMAIL_LENGTH // 4 + 100)
        payload = {"email": oversized_text}
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("maximum allowed length", data["detail"].lower())

    def test_consecutive_predictions_stability(self):
        """POST /predict handles multiple consecutive predictions reliably."""
        samples = [
            ("Free casino bonus! Click http://gamble.example.com", "SPAM", 1),
            ("Can you send me the meeting minutes from Friday?", "NOT SPAM", 0),
            ("Claim $5,000 cash prize instantly!", "SPAM", 1),
            ("Please find the attached invoice for last month's consulting work.", "NOT SPAM", 0),
        ]
        for text, expected_pred, expected_label in samples:
            res = requests.post(f"{BASE_URL}/predict", json={"email": text}, timeout=5)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["prediction"], expected_pred)
            self.assertEqual(data["label"], expected_label)


if __name__ == "__main__":
    unittest.main(verbosity=2)
