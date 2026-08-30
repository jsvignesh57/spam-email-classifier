"""
Spam Email Classifier — Phase 9 Real-World HAM Robustness Audit & Parity Engine

Conducts:
1. Production Artifact Hash Verification (Pre & Post)
2. Python Inference vs FastAPI Web Inference Parity Testing
3. Preprocessing Consistency Verification
4. Real-World 16-Category HAM Robustness Evaluation (80 samples)
5. Decision-Score Distribution & Category Breakdown Analysis
6. Diagnostic Feature Attribution for near-boundary predictions (+0.0617 welcome email)
7. Generation of visualizations and CSV evaluation records
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.preprocess import normalize_text

# ----------------------------------------------------------------------
# Frozen Production Artifact Paths & Checksums
# ----------------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "models" / "final_spam_classifier_v2.joblib"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_HASHES = {
    "models/final_spam_classifier_v2.joblib": "daaf9e4ae4f92ea688f8ae589518431414df67692b698d5d2c7bdeff2a8fc10b",
    "models/tfidf_vectorizer.joblib": "4db48a627fd8588a6cb9f09dfbbed54a3b7376c8508bce7fcc1fea86119688f2",
    "data/raw/internship.csv": "a5d0d75d15d370ef2dd3229b30204b18deab6d3fd249206e89a2c58f901bcc77",
    "data/processed/cleaned_internship.csv": "72de2b54b15db7eeb249cf00043c95e460b109a3077fc20816fb625db8be5e8d",
    "data/processed/train_test_split.npz": "1fd473c855032c872b6096e0ce19dde21c7ed861630a9716562764ba415a7fbf",
}

BASE_URL = "http://127.0.0.1:8000"


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_artifact_hashes() -> bool:
    for rel_path, expected in FROZEN_HASHES.items():
        p = PROJECT_ROOT / rel_path
        if not p.exists():
            print(f"[ERROR] Missing artifact: {rel_path}")
            return False
        actual = compute_sha256(p)
        if actual != expected:
            print(f"[ERROR] Hash mismatch for {rel_path}!\nExpected: {expected}\nActual:   {actual}")
            return False
    return True


# ----------------------------------------------------------------------
# 80-Sample Controlled Real-World HAM Dataset Across 16 Categories
# ----------------------------------------------------------------------
HAM_ROBUSTNESS_SAMPLES = [
    # A. Personal communication (5)
    {"category": "A. Personal Communication", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Hey Sarah, hope you're having a great week! Are we still on for lunch this Thursday at noon? Let me know what time works best for you."},
    {"category": "A. Personal Communication", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Hi Mom, just wanted to check in and see how Dad's doctor appointment went today. Give me a call when you have a free moment."},
    {"category": "A. Personal Communication", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Happy birthday David! Wishing you a fantastic year ahead filled with health, joy, and success. Hope you celebrate well with the family."},
    {"category": "A. Personal Communication", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Thanks so much for hosting dinner last night, the food was delicious and it was wonderful catching up with everyone!"},
    {"category": "A. Personal Communication", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Hey Alex, I left my blue umbrella in your car after the hike yesterday. Can I grab it from you sometime this weekend?"},

    # B. College/University communication (5)
    {"category": "B. College/University", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: CS402 Assignment 3 Submission Deadline Extended\n\nDear Students,\nThe deadline for Assignment 3 has been extended to Friday at 11:59 PM. Please make sure your code runs on the department lab servers before submitting."},
    {"category": "B. College/University", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Office Hours Update for Prof. Miller\n\nHi everyone, I will be holding additional office hours this Wednesday from 2 to 4 PM in Room 314 for anyone with questions regarding the upcoming midterm exam."},
    {"category": "B. College/University", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Spring Semester Course Registration Schedule\n\nDear Student, Course registration for the upcoming semester opens on November 15 at 8:00 AM. Please review your degree audit and meet with your advisor prior to enrollment."},
    {"category": "B. College/University", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Campus Library Holiday Hours\n\nThe University Main Library will operate on modified hours during the holiday break. Online research databases and journal access remain available 24/7."},
    {"category": "B. College/University", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Academic Advising Appointment Confirmation\n\nYour advising appointment with Dr. Johnson is confirmed for Tuesday, Oct 12 at 10:30 AM in Hall 204. Please bring your draft course plan."},

    # C. Work/Project communication (5)
    {"category": "C. Work/Project", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Q3 Sprint Review and Roadmap Planning\n\nHi team, attached is the presentation deck summarizing our Q3 sprint velocity and deliverables. We will review customer feedback and set Q4 priorities on Monday."},
    {"category": "C. Work/Project", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Code Review for PR #142 - User Authentication Module\n\nHi John, I left a few comments on your pull request regarding token expiration handling. Overall the architecture looks solid. Let's sync once tests pass."},
    {"category": "C. Work/Project", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Production Release Deployment Notice\n\nTeam, the release of v2.4.0 is scheduled for Thursday at 9 PM UTC. Maintenance window is expected to last 30 minutes with zero planned downtime."},
    {"category": "C. Work/Project", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Updated Project Requirements Document\n\nHi all, I have updated the functional specification document based on yesterday's client workshop. Please review Section 4 before our engineering sync."},
    {"category": "C. Work/Project", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Weekly Bug Triage Summary\n\nHere is our weekly bug triage breakdown: 14 closed, 3 in progress, 0 critical blockers. Great progress this week everyone."},

    # D. Meeting/Calendar messages (5)
    {"category": "D. Meeting/Calendar", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Invitation: Monthly Architecture Review @ Thu Oct 14, 2026 2pm - 3pm\n\nYou have been invited to Monthly Architecture Review. Agenda: Database indexing strategies and caching layers."},
    {"category": "D. Meeting/Calendar", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Accepted: 1-on-1 Sync - David / Marcus\n\nMarcus has accepted your calendar invitation for the 1-on-1 sync on Wednesday at 11:00 AM."},
    {"category": "D. Meeting/Calendar", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Rescheduled: Design System Workshop\n\nThe Design System Workshop has been moved to Thursday at 3 PM due to a conference room scheduling conflict. New meeting room is Studio B."},
    {"category": "D. Meeting/Calendar", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Daily Standup Reminder - 9:30 AM\n\nGood morning team! Please join our daily standup meeting at 9:30 AM. Be prepared to share your yesterday progress and today's focus."},
    {"category": "D. Meeting/Calendar", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Meeting Notes & Action Items - Client Onboarding\n\nThanks everyone for the productive call. Action items: Lisa to finalize SLA draft by Friday, Tom to provision test environment credentials."},

    # E. Technical/Community newsletters (5)
    {"category": "E. Technical Newsletters", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Python Weekly Issue #580\n\nWelcome to this week's issue of Python Weekly! In this edition: Building fast async APIs with FastAPI, understanding Python 3.13 GIL improvements, and advanced NumPy indexing techniques."},
    {"category": "E. Technical Newsletters", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: JavaScript & Web Dev Digest #142\n\nHere are the top articles this week: CSS subgrid practical guide, optimizing React re-renders, and the state of WebAssembly in 2026."},
    {"category": "E. Technical Newsletters", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Machine Learning Research Round-up - October Edition\n\nWelcome to our monthly ML digest covering new papers in transformer distillation, efficient linear SVM solvers, and synthetic data validation strategies."},
    {"category": "E. Technical Newsletters", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: DevOps & Cloud Architecture Monthly\n\nThis month: Kubernetes autoscaling best practices, comparing Terraform vs OpenTofu, and zero-trust VPC security architecture patterns."},
    {"category": "E. Technical Newsletters", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Open Source Software Community Dispatch\n\nHighlights from around the open-source ecosystem: Scikit-learn release candidate notes, PostgreSQL 17 benchmark analysis, and community contribution guides."},

    # F. Legitimate Welcome Emails (5)
    {"category": "F. Legitimate Welcome Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Welcome to Developer Community! Thank you for joining us.\n\nHi there, welcome to our open-source developer forum. We are excited to have you! You can introduce yourself in the welcome channel and browse community discussions at http://forum.example.org"},
    {"category": "F. Legitimate Welcome Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Welcome to our newsletter! Thank you for subscribing.\n\nWelcome! Thank you for signing up to receive our monthly engineering articles and case studies. Visit http://example.com to explore our archive."},
    {"category": "F. Legitimate Welcome Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Welcome to CloudHost! Your account is ready.\n\nWelcome to CloudHost. We are thrilled you chose our developer hosting platform. Here is your quickstart guide to provisioning your first virtual machine."},
    {"category": "F. Legitimate Welcome Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Welcome to BookClub! Thanks for becoming a member.\n\nWelcome to our neighborhood book club. Our next book discussion is scheduled for November 12th. Let us know if you need a copy of the reading list."},
    {"category": "F. Legitimate Welcome Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Welcome to FitnessTracker! Let's get started.\n\nWelcome to FitnessTracker. To set up your weekly workout routine and sync your device, open your settings and select your activity goals."},

    # G. Legitimate Promotional / Discounts (5)
    {"category": "G. Legitimate Promotional/Discounts", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Fall Book Sale: 20% off all computer science titles this weekend\n\nDear customer, enjoy 20% off all programming and data science textbooks this weekend at our campus bookstore. Visit our website for details."},
    {"category": "G. Legitimate Promotional/Discounts", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Special Subscriber Offer: Annual subscription discount\n\nAs a valued subscriber to our tech magazine, renew your subscription this month and save $15 on our annual print and digital bundle."},
    {"category": "G. Legitimate Promotional/Discounts", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Hardware Store Weekend Special on Garden Tools\n\nCheck out our weekend specials on gardening tools and supplies. Available at all local retail branches through Sunday."},
    {"category": "G. Legitimate Promotional/Discounts", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Coffee Club Member Appreciation Week\n\nThis week only, members receive double reward points on all whole-bean coffee orders placed online or in-store. Thank you for being a loyal member."},
    {"category": "G. Legitimate Promotional/Discounts", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Seasonal Clearance Sale on Outdoor Gear\n\nOur end-of-season clearance sale is now live. Save on select hiking apparel, camping equipment, and accessories while supplies last."},

    # H. Transactional notifications (5)
    {"category": "H. Transactional Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Your Order Receipt - Order #4829103\n\nThank you for your purchase! We have received your order #4829103 totaling $42.50. You will receive another notification once your items ship."},
    {"category": "H. Transactional Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Payment Confirmation for Invoice #9021\n\nWe have successfully received your payment of $120.00 for Invoice #9021. Thank you for your business. Your updated balance is $0.00."},
    {"category": "H. Transactional Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Electronic Ticket & Receipt - Flight Booking #TX789\n\nYour flight reservation TX789 is confirmed. Passenger: Emily Watson. Departure: Oct 20 at 8:15 AM from Terminal 2."},
    {"category": "H. Transactional Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Utility Bill Auto-Pay Notification\n\nYour monthly electric utility bill of $68.40 has been scheduled for auto-payment on October 25 from your checking account ending in 4102."},
    {"category": "H. Transactional Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Subscription Renewal Confirmation\n\nYour monthly cloud backup subscription has renewed successfully for $9.99. Your next billing date will be November 28, 2026."},

    # I. Account Notifications (5)
    {"category": "I. Account Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Security Alert: New Login from Chrome on Windows\n\nWe noticed a new login to your account from a Chrome browser on Windows. If this was you, no action is needed. If not, please update your password immediately."},
    {"category": "I. Account Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Password Reset Request\n\nWe received a request to reset the password for your account. If you made this request, please click the link below to set a new password: http://auth.example.com/reset?token=xyz"},
    {"category": "I. Account Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Two-Factor Authentication Enabled Successfully\n\nTwo-factor authentication (2FA) has been successfully activated on your account. Backup codes have been generated for account recovery."},
    {"category": "I. Account Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Account Email Address Updated\n\nThis email confirms that the primary contact address associated with your developer profile has been updated to alex@example.org."},
    {"category": "I. Account Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Privacy Policy Updates Notice\n\nWe are updating our Terms of Service and Privacy Policy to provide greater transparency on data handling. Please review the updated policy on our website."},

    # J. Delivery/Order Notifications (5)
    {"category": "J. Delivery/Order Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Your Package Has Shipped! Tracking #940011189956\n\nGreat news! Your package containing Order #78190 has shipped via Priority Mail. Estimated delivery date: Thursday, Oct 15."},
    {"category": "J. Delivery/Order Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Out for Delivery: Package #8472910\n\nYour courier is out for delivery with package #8472910. The driver is expected to arrive between 1:00 PM and 3:30 PM today."},
    {"category": "J. Delivery/Order Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Package Delivered to Front Porch\n\nYour order from BookDepot was delivered to your front porch at 2:15 PM today. Thank you for shopping with us."},
    {"category": "J. Delivery/Order Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Delivery Rescheduled for Order #38291\n\nYour package delivery was rescheduled due to adverse weather conditions. New estimated arrival date is tomorrow afternoon."},
    {"category": "J. Delivery/Order Notifications", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Prescription Refill Ready for Pickup\n\nYour prescription refill #RX-99214 is ready for pickup at your local pharmacy counter. Operating hours: 9 AM - 8 PM."},

    # K. Subscription emails (5)
    {"category": "K. Subscription Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Your Weekly Cloud Usage Report\n\nHere is your compute resource breakdown for the past 7 days: 120 CPU hours utilized, 45 GB storage consumed, estimated month-to-date cost: $14.20."},
    {"category": "K. Subscription Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: StreamService: New Movies & Shows Added This Month\n\nCheck out the new documentary and series titles added to your streaming library for October. Stream anytime on all your connected devices."},
    {"category": "K. Subscription Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Podcast Subscriber Digest - Episode #88 Released\n\nEpisode #88 is live: Deep Dive into Distributed Consensus Algorithms. Listen now on your favorite podcast app."},
    {"category": "K. Subscription Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Software License Renewal Notice\n\nYour team license for CodeEditor Pro will expire in 30 days on November 15. Click here to review your renewal options: http://example.com/renew"},
    {"category": "K. Subscription Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Monthly Membership Statement - Community Gym\n\nYour monthly gym membership dues of $45.00 for October have been processed. Thank you for being a member of our community center."},

    # L. Informational announcements (5)
    {"category": "L. Informational Announcements", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Scheduled Power Maintenance Notice - Downtown District\n\nThe local power utility will perform scheduled grid upgrades on Sunday between 1 AM and 4 AM. Expect brief intermittent service interruptions."},
    {"category": "L. Informational Announcements", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Annual Neighborhood Block Party & Food Drive\n\nJoin us for our annual block party next Saturday from 11 AM to 4 PM at Oak Park! Please consider bringing canned food items for our local shelter drive."},
    {"category": "L. Informational Announcements", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Public Transit Route #14 Schedule Changes\n\nEffective Monday, Route #14 buses will run every 12 minutes during morning and evening rush hours to accommodate increased passenger demand."},
    {"category": "L. Informational Announcements", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: City Water Department Annual Quality Report\n\nThe City Water Department has published its 2026 drinking water quality report. The complete document is available at http://citywater.example.gov/report"},
    {"category": "L. Informational Announcements", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Blood Donation Drive at Community Center\n\nThe Red Cross is hosting a blood donation drive at the West End Community Center on October 22. Walk-ins welcome or schedule an appointment online."},

    # M. Short legitimate emails (5)
    {"category": "M. Short Legitimate Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Sounds good to me, see you at 3 PM."},
    {"category": "M. Short Legitimate Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Thanks! Received the file."},
    {"category": "M. Short Legitimate Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Got it, I will review and reply by tomorrow morning."},
    {"category": "M. Short Legitimate Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Can you please re-send the meeting invite? I didn't get it."},
    {"category": "M. Short Legitimate Emails", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Yes, approved from my side. Proceed with the order."},

    # N. Emails containing URLs (5)
    {"category": "N. URL-Containing Legitimate", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Hi team, please review the draft architecture proposal on our wiki: https://wiki.internal.example.com/display/ARCH/Database+Sharding+Plan"},
    {"category": "N. URL-Containing Legitimate", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Here is the link to the Python documentation on dataclasses: https://docs.python.org/3/library/dataclasses.html for our discussion."},
    {"category": "N. URL-Containing Legitimate", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "You can check the conference schedule and speaker lineup at https://pycon.example.org/2026/schedule/talks"},
    {"category": "N. URL-Containing Legitimate", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Please upload your project presentation slides to our shared Google Drive folder: https://drive.example.com/drive/folders/project-sync"},
    {"category": "N. URL-Containing Legitimate", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "The pull request has been merged to main: https://github.com/example-org/repo/pull/582. Thanks for the quick review!"},

    # O. Emails containing numbers (5)
    {"category": "O. Numeric-Heavy Legitimate", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "The server CPU load peaked at 88.5% with 14,230 active connections. Memory usage remained steady at 12.4 GB out of 32 GB total."},
    {"category": "O. Numeric-Heavy Legitimate", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Sprint metrics: 42 story points completed, 8 points carried over, velocity index 1.15, test coverage at 94.2% across 1,840 tests."},
    {"category": "O. Numeric-Heavy Legitimate", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Please ship 150 units of Part #984-210 and 75 units of Model #550-A to Warehouse #4 before 5:00 PM on October 18."},
    {"category": "O. Numeric-Heavy Legitimate", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Your quarterly 401(k) retirement statement: Beginning balance $45,210.80, contributions $3,500.00, gain $2,140.15, ending balance $50,850.95."},
    {"category": "O. Numeric-Heavy Legitimate", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Flight 1482 departs at 07:45 from Gate B22. Seat assignment 14C. Baggage claim carousel 4 upon arrival."},

    # P. Unicode-containing legitimate emails (5)
    {"category": "P. Unicode-Containing", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Rendez-vous confirmé pour l'entretien\n\nBonjour Jean, nous confirmons votre entretien d'embauche pour mardi prochain à 14h00 dans nos locaux de Paris."},
    {"category": "P. Unicode-Containing", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Project team greetings! 🚀\n\nHello team, great job on wrapping up the sprint today! 🎉 Let's celebrate our successful release this Friday."},
    {"category": "P. Unicode-Containing", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Überweisung eingegangen - Rechnung Nr. 8831\n\nGuten Tag, wir bestätigen den Eingang Ihrer Zahlung über 150,00 € für die Rechnung 8831. Vielen Dank!"},
    {"category": "P. Unicode-Containing", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: Team update: Welcome our new engineers! 🌟\n\nPlease give a warm welcome to Priya and Carlos who joined our machine learning infrastructure team today."},
    {"category": "P. Unicode-Containing", "expected_label": 0, "source_type": "synthetic_safe",
     "text": "Subject: 会議のスケジュール確認 (Meeting Confirmation)\n\n田中様、来週火曜日15時のプロジェクト定例会議の件、承知いたしました。よろしくお願いいたします。"}
]


def run_parity_tests(model, vectorizer) -> bool:
    """Verify that Python canonical inference and FastAPI /predict produce identical results."""
    print("\n" + "=" * 70)
    print("STEP 3 & 4: PYTHON VS WEB INFERENCE PARITY & PREPROCESSING AUDIT")
    print("=" * 70)

    test_emails = [
        ("Obvious Spam", "Congratulations! You have won a $10,000 lottery prize. Claim your reward now at http://claim.example.com"),
        ("Obvious Ham", "Hi team, can we meet tomorrow at 10 AM to discuss the quarterly sprint deliverables? Thanks."),
        ("Welcome +0.0617 Case", "Subject: Welcome to our newsletter! We are happy to have you on board. Check out our latest articles and resources at http://example.com"),
        ("Short Ham", "Sounds good to me, see you at 3 PM."),
        ("Promotional Ham", "Special offer for our customers: enjoy 20% off selected products this weekend. Visit our website to learn more.")
    ]

    all_passed = True
    for name, email_text in test_emails:
        # Python canonical inference
        norm_py = normalize_text(email_text)
        X_py = vectorizer.transform([norm_py])
        score_py = float(model.decision_function(X_py)[0])
        pred_py = "SPAM" if score_py >= 0.0 else "NOT SPAM"
        label_py = 1 if score_py >= 0.0 else 0
        rounded_score_py = round(score_py, 4)

        # Web API inference
        try:
            res = requests.post(f"{BASE_URL}/predict", json={"email": email_text}, timeout=5)
            if res.status_code != 200:
                print(f"[FAIL] {name}: Web API returned HTTP {res.status_code}")
                all_passed = False
                continue
            web_data = res.json()
            pred_web = web_data["prediction"]
            label_web = web_data["label"]
            score_web = web_data["decision_score"]

            score_diff = abs(rounded_score_py - score_web)
            parity_ok = (pred_py == pred_web) and (label_py == label_web) and (score_diff < 1e-4)

            status = "[PASS]" if parity_ok else "[FAIL]"
            if not parity_ok:
                all_passed = False
            print(f"  {status} {name:<22} | Python: {pred_py:<8} (score: {rounded_score_py:+.4f}) | Web: {pred_web:<8} (score: {score_web:+.4f})")

        except Exception as e:
            print(f"  [ERROR] Web server connection failed: {e}")
            all_passed = False

    return all_passed


def evaluate_ham_robustness(model, vectorizer) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate 80-sample HAM robustness dataset and compute detailed metrics."""
    records = []
    for idx, sample in enumerate(HAM_ROBUSTNESS_SAMPLES, 1):
        text = sample["text"]
        cat = sample["category"]
        exp = sample["expected_label"]
        source = sample["source_type"]

        norm = normalize_text(text)
        X = vectorizer.transform([norm])
        raw_score = float(model.decision_function(X)[0])
        pred_label = 1 if raw_score >= 0.0 else 0
        pred_text = "SPAM" if pred_label == 1 else "NOT SPAM"
        is_fp = (pred_label == 1)
        is_correct = (pred_label == exp)

        records.append({
            "sample_id": idx,
            "category": cat,
            "expected_label": exp,
            "predicted_label": pred_label,
            "prediction": pred_text,
            "decision_score": round(raw_score, 4),
            "is_correct": is_correct,
            "is_false_positive": is_fp,
            "text_length": len(text),
            "email_preview": text.replace("\n", " ")[:75] + ("..." if len(text) > 75 else "")
        })

    df = pd.DataFrame(records)
    csv_path = REPORTS_DIR / "phase_9_real_world_ham_robustness.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved evaluation dataset to: {csv_path}")

    # Summary Statistics
    total_ham = len(df)
    correct_ham = int(df["is_correct"].sum())
    total_fp = int(df["is_false_positive"].sum())
    fp_rate = total_fp / total_ham
    fp_pct = fp_rate * 100.0
    ham_accuracy = correct_ham / total_ham * 100.0

    print("\n" + "=" * 70)
    print("REAL-WORLD HAM ROBUSTNESS EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Real-World HAM Samples: {total_ham}")
    print(f"Correctly Classified as HAM: {correct_ham} ({ham_accuracy:.2f}%)")
    print(f"False Positives (Misclassified as SPAM): {total_fp} ({fp_pct:.2f}%)")
    print(f"False Positive Rate (FPR): {fp_rate:.4f} ({fp_pct:.2f}%)")

    # Category Breakdown
    cat_summary = df.groupby("category").agg(
        total_samples=("sample_id", "count"),
        correct=("is_correct", "sum"),
        false_positives=("is_false_positive", "sum"),
        mean_score=("decision_score", "mean"),
        min_score=("decision_score", "min"),
        max_score=("decision_score", "max")
    ).reset_index()
    cat_summary["fp_rate"] = cat_summary["false_positives"] / cat_summary["total_samples"]
    cat_summary["fp_pct"] = cat_summary["fp_rate"] * 100.0

    print("\n" + "-" * 78)
    print(f"{'Category':<32} | {'Total':<5} | {'FP':<4} | {'FPR %':<7} | {'Mean Score':<10} | {'Score Range':<14}")
    print("-" * 78)
    for _, row in cat_summary.iterrows():
        print(f"{row['category']:<32} | {row['total_samples']:<5} | {row['false_positives']:<4} | {row['fp_pct']:>5.1f}% | {row['mean_score']:>+9.4f} | [{row['min_score']:>+5.2f}, {row['max_score']:>+5.2f}]")
    print("-" * 78)

    # Score Bucket Distribution
    buckets = [
        ("< -1.00", lambda s: s < -1.0),
        ("-1.00 to -0.50", lambda s: -1.0 <= s < -0.5),
        ("-0.50 to -0.25", lambda s: -0.5 <= s < -0.25),
        ("-0.25 to 0.00", lambda s: -0.25 <= s < 0.0),
        ("0.00 to +0.25 (Near-Boundary FP)", lambda s: 0.0 <= s < 0.25),
        ("+0.25 to +0.50 (Moderate FP)", lambda s: 0.25 <= s < 0.5),
        ("+0.50 to +1.00 (High FP)", lambda s: 0.5 <= s < 1.0),
        ("> +1.00 (Severe FP)", lambda s: s >= 1.0)
    ]
    bucket_counts = {}
    for name, cond in buckets:
        cnt = int(df["decision_score"].apply(cond).sum())
        pct = (cnt / total_ham) * 100.0
        bucket_counts[name] = {"count": cnt, "percent": pct}

    print("\nDECISION-SCORE BUCKET DISTRIBUTION (80 HAM SAMPLES):")
    for name, data in bucket_counts.items():
        bar = "#" * int(data["percent"] / 2)
        print(f"  {name:<34} : {data['count']:>2} ({data['percent']:>5.1f}%) | {bar}")

    stats = {
        "total_ham": total_ham,
        "correct_ham": correct_ham,
        "total_fp": total_fp,
        "fp_rate": fp_rate,
        "fp_pct": fp_pct,
        "cat_summary": cat_summary,
        "bucket_counts": bucket_counts
    }
    return df, stats


def generate_visualizations(df: pd.DataFrame, cat_summary: pd.DataFrame):
    """Generate high-resolution plots for score distribution and category false-positive rates."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Decision Score Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    scores = df["decision_score"]

    ham_scores = df[df["is_false_positive"] == False]["decision_score"]
    fp_scores = df[df["is_false_positive"] == True]["decision_score"]

    bins = np.linspace(-1.8, 1.0, 29)
    ax.hist(ham_scores, bins=bins, color="#10b981", alpha=0.8, edgecolor="black", label=f"Correct HAM (n={len(ham_scores)})")
    if len(fp_scores) > 0:
        ax.hist(fp_scores, bins=bins, color="#ef4444", alpha=0.8, edgecolor="black", label=f"False Positive SPAM (n={len(fp_scores)})")

    ax.axvline(x=0.0, color="#dc2626", linestyle="--", linewidth=2, label="Production Threshold (0.0)")
    ax.axvspan(0.0, 0.25, color="#fef08a", alpha=0.3, label="Near-Boundary Margin [0.0, +0.25]")

    ax.set_title("Phase 9: Real-World HAM Decision Score Distribution (LinearSVC C=10.0)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("LinearSVC Decision Score (Signed Hyperplane Margin)", fontsize=11)
    ax.set_ylabel("Email Count", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    dist_plot_path = REPORTS_DIR / "phase_9_ham_decision_score_distribution.png"
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"Generated plot: {dist_plot_path}")

    # Plot 2: Category False Positive Rates
    fig, ax = plt.subplots(figsize=(12, 6))
    cats = [c.split(". ", 1)[-1] for c in cat_summary["category"]]
    fp_pcts = cat_summary["fp_pct"]
    colors = ["#ef4444" if pct > 0 else "#3b82f6" for pct in fp_pcts]

    bars = ax.barh(cats, fp_pcts, color=colors, edgecolor="black", height=0.65)
    ax.set_xlim(0, max(max(fp_pcts) + 10, 30))
    ax.set_xlabel("False Positive Rate (%)", fontsize=11)
    ax.set_title("Phase 9: False Positive Rate by Email Category (n=5 per category)", fontsize=13, fontweight="bold", pad=12)

    for bar, pct in zip(bars, fp_pcts):
        width = bar.get_width()
        ax.text(width + 1.0, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", ha="left", va="center", fontsize=9, fontweight="bold")

    ax.invert_yaxis()
    plt.tight_layout()

    cat_plot_path = REPORTS_DIR / "phase_9_ham_category_false_positive.png"
    plt.savefig(cat_plot_path, dpi=300)
    plt.close()
    print(f"Generated plot: {cat_plot_path}")


def analyze_welcome_newsletter_case(model, vectorizer):
    """Diagnose feature contributions for welcome newsletter emails with scores around +0.0617."""
    print("\n" + "=" * 70)
    print("STEP 9: DETAILED FEATURE ATTRIBUTION FOR WELCOME / NEWSLETTER EMAIL")
    print("=" * 70)

    email_text = (
        "Subject: Welcome to our newsletter! We are happy to have you on board. "
        "Check out our latest articles and resources at http://example.com"
    )
    norm = normalize_text(email_text)
    X = vectorizer.transform([norm])
    score = float(model.decision_function(X)[0])

    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]  # shape: (121288,)
    intercept = float(model.intercept_[0])

    # Non-zero feature indices for this sample
    non_zero_indices = X.nonzero()[1]
    feature_contributions = []

    for idx in non_zero_indices:
        feat_name = feature_names[idx]
        tfidf_val = X[0, idx]
        weight = coef[idx]
        contribution = tfidf_val * weight
        feature_contributions.append({
            "feature": feat_name,
            "tfidf_value": round(float(tfidf_val), 4),
            "svm_weight": round(float(weight), 4),
            "net_contribution": round(float(contribution), 4),
            "leaning": "Spam (+)" if contribution > 0 else "Ham (-)"
        })

    feat_df = pd.DataFrame(feature_contributions).sort_values("net_contribution", ascending=False)

    print(f"Target Email Text: \"{email_text}\"")
    print(f"Normalized Text:   \"{norm}\"")
    print(f"LinearSVC Intercept: {intercept:+.4f}")
    print(f"Total Computed Decision Score: {score:+.4f}")
    print("\nActive Feature Contributions (Top Positive / Spam-leaning vs Negative / Ham-leaning):")
    print("-" * 72)
    print(f"{'Feature Token':<24} | {'TF-IDF':<8} | {'Weight':<8} | {'Contribution':<12} | {'Leaning':<8}")
    print("-" * 72)
    for _, r in feat_df.iterrows():
        print(f"{r['feature']:<24} | {r['tfidf_value']:>7.4f} | {r['svm_weight']:>+7.4f} | {r['net_contribution']:>+11.4f} | {r['leaning']}")
    print("-" * 72)

    return feat_df, score, intercept


def main():
    print("============================================================")
    print("PHASE 9 — REAL-WORLD ROBUSTNESS & CONSISTENCY AUDIT")
    print("============================================================")

    # 1. Pre-test Hash Verification
    print("\nSTEP 2: Verifying Pre-Test Artifact Hashes...")
    if not verify_artifact_hashes():
        sys.exit(1)
    print("[PASS] All 5 production artifact SHA-256 hashes verified bit-for-bit unchanged.")

    # 2. Load Production Artifacts
    print("\nLoading Production Model & Vectorizer...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"Loaded {type(model).__name__} (C={model.C}) and {type(vectorizer).__name__} ({len(vectorizer.get_feature_names_out())} features).")

    # 3. Python vs Web Parity Test
    parity_ok = run_parity_tests(model, vectorizer)
    if not parity_ok:
        print("[WARNING] Python vs Web inference parity test encountered issues.")

    # 4. Evaluate 80-Sample HAM Robustness Dataset
    df_results, stats = evaluate_ham_robustness(model, vectorizer)

    # 5. Visualizations
    generate_visualizations(df_results, stats["cat_summary"])

    # 6. Welcome Newsletter Feature Attribution
    feat_df, target_score, intercept = analyze_welcome_newsletter_case(model, vectorizer)

    # 7. Post-test Hash Verification
    print("\nSTEP 2 (Post): Verifying Post-Test Artifact Hashes...")
    if not verify_artifact_hashes():
        sys.exit(1)
    print("[PASS] All production artifacts remain bit-for-bit unchanged after Phase 9 audit.")


if __name__ == "__main__":
    main()
