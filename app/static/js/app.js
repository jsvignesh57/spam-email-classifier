/**
 * Spam Email Classifier — Frontend Application Logic
 * 
 * Handles client-side validation, asynchronous fetch to /predict,
 * character counting, sample loading, dynamic DOM rendering, and accessibility.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('classifier-form');
    const emailInput = document.getElementById('email-input');
    const checkBtn = document.getElementById('check-btn');
    const clearBtn = document.getElementById('clear-btn');
    const btnText = document.getElementById('btn-text');
    const btnIcon = document.getElementById('btn-icon');
    const btnSpinner = document.getElementById('btn-spinner');
    const currentChars = document.getElementById('current-chars');
    const charCounter = document.getElementById('char-count');
    const inputError = document.getElementById('input-error');
    const inputErrorText = document.getElementById('input-error-text');
    
    // Result Card Elements
    const resultContainer = document.getElementById('result-container');
    const resultBadge = document.getElementById('result-badge');
    const resultBadgeIcon = document.getElementById('result-badge-icon');
    const resultBadgeText = document.getElementById('result-badge-text');
    const resultSummary = document.getElementById('result-summary');
    const detailScore = document.getElementById('detail-score');
    const detailRule = document.getElementById('detail-rule');

    // Quick Test Chips
    const sampleSpamBtn = document.getElementById('sample-spam-btn');
    const sampleHamBtn = document.getElementById('sample-ham-btn');

    // Constants
    const MAX_LENGTH = 50000;

    // Sample Texts
    const SAMPLES = {
        spam: "Subject: Urgent: Claim your $10,000 Amazon Gift Card reward now!\n\nCongratulations! Your email has been randomly selected to receive an exclusive $10,000 reward. Click here http://claim-prize-now.example.com to verify your account and claim your prize immediately before it expires!",
        ham: "Subject: Project update and sprint review meeting notes\n\nHi team,\n\nThanks for attending today's project review meeting. The sprint deliverables are on track for next Tuesday. Please review the updated documentation attached and let me know if you have any questions.\n\nBest regards,\nAlex"
    };

    // ------------------------------------------------------------------
    // Character Counter
    // ------------------------------------------------------------------
    function updateCharCount() {
        const length = emailInput.value.length;
        currentChars.textContent = length.toLocaleString();

        charCounter.classList.remove('limit-near', 'limit-reached');
        if (length >= MAX_LENGTH) {
            charCounter.classList.add('limit-reached');
        } else if (length >= MAX_LENGTH * 0.9) {
            charCounter.classList.add('limit-near');
        }
    }

    emailInput.addEventListener('input', () => {
        updateCharCount();
        hideError();
    });

    // ------------------------------------------------------------------
    // Error & UI State Management
    // ------------------------------------------------------------------
    function showError(message) {
        inputErrorText.textContent = message;
        inputError.style.display = 'flex';
    }

    function hideError() {
        inputError.style.display = 'none';
        inputErrorText.textContent = '';
    }

    function hideResult() {
        resultContainer.style.display = 'none';
    }

    function setLoading(isLoading) {
        if (isLoading) {
            checkBtn.disabled = true;
            clearBtn.disabled = true;
            btnText.textContent = 'Analyzing...';
            btnIcon.style.display = 'none';
            btnSpinner.style.display = 'inline-block';
        } else {
            checkBtn.disabled = false;
            clearBtn.disabled = false;
            btnText.textContent = 'Check Email';
            btnIcon.style.display = 'inline-block';
            btnSpinner.style.display = 'none';
        }
    }

    // ------------------------------------------------------------------
    // Display Prediction Result
    // ------------------------------------------------------------------
    function displayResult(data) {
        const { prediction, label, decision_score } = data;
        const isSpam = label === 1 || prediction.toUpperCase() === 'SPAM';
        const score = typeof decision_score === 'number' ? decision_score : parseFloat(decision_score);

        // Update Badge & Styles
        resultBadge.className = 'result-badge ' + (isSpam ? 'badge-spam' : 'badge-ham');
        resultBadgeIcon.textContent = isSpam ? '⚠️' : '🛡️';
        resultBadgeText.textContent = isSpam ? 'SPAM EMAIL' : 'NOT SPAM';

        // Nuanced, boundary-aware summary descriptions
        if (score >= 0.50) {
            resultSummary.textContent = 'This email is classified as spam by the model.';
        } else if (score >= 0.0) {
            resultSummary.textContent = 'This email is classified as spam, but its score is close to the decision boundary.';
        } else if (score > -0.50) {
            resultSummary.textContent = 'This email is classified as not spam, but its score is close to the decision boundary.';
        } else {
            resultSummary.textContent = 'This email is classified as not spam by the model.';
        }

        // Format Signed Decision Score
        const formattedScore = (score >= 0 ? '+' : '') + score.toFixed(4);
        detailScore.textContent = formattedScore;

        // Rule Explanation
        detailRule.textContent = isSpam
            ? 'Score >= 0.00 → Spam'
            : 'Score < 0.00 → Not Spam';

        // Reveal Result Card
        resultContainer.style.display = 'block';
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // ------------------------------------------------------------------
    // Form Submission & API Request
    // ------------------------------------------------------------------
    async function handleSubmit(event) {
        event.preventDefault();
        hideError();
        hideResult();

        const rawText = emailInput.value;
        const trimmedText = rawText.trim();

        // Client-side validation: empty / whitespace
        if (!trimmedText) {
            showError('Please enter an email to analyze.');
            emailInput.focus();
            return;
        }

        // Client-side validation: max length
        if (rawText.length > MAX_LENGTH) {
            showError(`Email exceeds the maximum allowed length of ${MAX_LENGTH.toLocaleString()} characters.`);
            return;
        }

        setLoading(true);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ email: rawText })
            });

            const data = await response.json();

            if (!response.ok) {
                const errorMessage = data.detail || 'An error occurred while analyzing the email.';
                showError(errorMessage);
                return;
            }

            displayResult(data);

        } catch (error) {
            console.error('Classification error:', error);
            showError('Unable to connect to the classification service. Please ensure the backend server is running.');
        } finally {
            setLoading(false);
        }
    }

    form.addEventListener('submit', handleSubmit);

    // ------------------------------------------------------------------
    // Clear Button Handler
    // ------------------------------------------------------------------
    clearBtn.addEventListener('click', () => {
        emailInput.value = '';
        updateCharCount();
        hideError();
        hideResult();
        emailInput.focus();
    });

    // ------------------------------------------------------------------
    // Quick Sample Loaders
    // ------------------------------------------------------------------
    sampleSpamBtn.addEventListener('click', () => {
        emailInput.value = SAMPLES.spam;
        updateCharCount();
        hideError();
        hideResult();
        emailInput.focus();
    });

    sampleHamBtn.addEventListener('click', () => {
        emailInput.value = SAMPLES.ham;
        updateCharCount();
        hideError();
        hideResult();
        emailInput.focus();
    });

    // Initial character count
    updateCharCount();
});
