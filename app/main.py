"""
Spam Email Classifier — FastAPI Web Application Backend

This module serves the REST API and the frontend application for the Spam Email
Classifier using the frozen production machine learning model (LinearSVC v2) and
canonical text preprocessing imported directly from `src.preprocess.normalize_text`.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("spam_classifier_app")

# ----------------------------------------------------------------------
# Path Resolution & Project Structure
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_spam_classifier_v2.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

# Import canonical preprocessing directly from src.preprocess
try:
    import sys
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    from src.preprocess import normalize_text
except ImportError as e:
    logger.error("Failed to import canonical normalize_text from src.preprocess: %s", e)
    raise RuntimeError(
        f"Could not import normalize_text from src.preprocess. Ensure {BASE_DIR} is in Python path."
    ) from e

# Input Constraints
MAX_EMAIL_LENGTH = 50_000
DECISION_THRESHOLD = 0.0

# In-memory model cache
ml_artifacts: Dict[str, Any] = {
    "model": None,
    "vectorizer": None,
    "is_ready": False,
}


def load_artifacts() -> None:
    """Load production model and vectorizer once into memory."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Production model artifact not found at {MODEL_PATH}")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Production vectorizer artifact not found at {VECTORIZER_PATH}")

    logger.info("Loading production model from %s...", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    logger.info("Loading production TF-IDF vectorizer from %s...", VECTORIZER_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    ml_artifacts["model"] = model
    ml_artifacts["vectorizer"] = vectorizer
    ml_artifacts["is_ready"] = True
    logger.info(
        "Artifacts loaded successfully. Model: %s, Features: %d",
        type(model).__name__,
        len(vectorizer.get_feature_names_out()),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to handle startup and shutdown."""
    load_artifacts()
    yield
    # Clean-up on shutdown
    ml_artifacts.clear()
    ml_artifacts["is_ready"] = False
    logger.info("Application shut down cleanly.")


# ----------------------------------------------------------------------
# FastAPI Application Configuration
# ----------------------------------------------------------------------
app = FastAPI(
    title="Spam Email Classifier API",
    description=(
        "Production inference service for classifying email messages as SPAM or NOT SPAM. "
        "Powered by a frozen LinearSVC model (C=10.0, word-level (1,2) TF-IDF) and "
        "canonical semantic token preprocessing. "
        "\n\n**Note**: The returned `decision_score` is the signed distance to the "
        "decision hyperplane (Ham < 0.0, Spam >= 0.0), NOT an uncalibrated probability."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Static file serving
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ----------------------------------------------------------------------
# Schemas & Validation Models
# ----------------------------------------------------------------------
class PredictRequest(BaseModel):
    email: str = Field(
        ...,
        description="The raw email text content to classify.",
        examples=["Congratulations! You have won a $1,000 gift card. Click here to claim your reward."],
    )

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Email input must be a string.")
        stripped = value.strip()
        if not stripped:
            raise ValueError("Email text cannot be empty or whitespace-only.")
        if len(value) > MAX_EMAIL_LENGTH:
            raise ValueError(
                f"Email exceeds maximum allowed length of {MAX_EMAIL_LENGTH:,} characters "
                f"(received {len(value):,} characters)."
            )
        return value


class PredictResponse(BaseModel):
    prediction: str = Field(
        ...,
        description="Classification label: 'SPAM' or 'NOT SPAM'.",
        examples=["SPAM"],
    )
    label: int = Field(
        ...,
        description="Numeric class identifier: 1 for SPAM, 0 for NOT SPAM.",
        examples=[1],
    )
    decision_score: float = Field(
        ...,
        description=(
            "Signed decision margin value from LinearSVC. "
            "Scores >= 0.0 indicate SPAM; scores < 0.0 indicate NOT SPAM. "
            "This is a raw geometric distance, NOT a probability."
        ),
        examples=[2.8542],
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Current service health status.")
    model: str = Field(..., description="Name of the production classifier architecture.")
    model_version: str = Field(..., description="Production model version.")
    decision_threshold: float = Field(..., description="Operating decision boundary threshold.")
    max_input_length: int = Field(..., description="Maximum allowed input character length.")


# ----------------------------------------------------------------------
# Custom Exception Handlers
# ----------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return clean, user-friendly JSON error messages for validation failures."""
    errors = exc.errors()
    error_messages = []
    for err in errors:
        msg = err.get("msg", "Invalid input")
        # Clean Pydantic prefix if present
        if msg.startswith("Value error, "):
            msg = msg.replace("Value error, ", "")
        error_messages.append(msg)

    detail_message = "; ".join(error_messages) if error_messages else "Invalid request data."
    logger.warning("Input validation failed: %s", detail_message)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": detail_message},
    )


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the single-page application interface."""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend index.html template not found.",
        )
    return FileResponse(index_file, media_type="text/html")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health and Model Status",
    description="Check the operational status of the inference service and view active model metadata.",
)
async def health_check() -> HealthResponse:
    """Return application health and production model configuration."""
    if not ml_artifacts["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML artifacts are not initialized or ready.",
        )

    return HealthResponse(
        status="healthy",
        model="LinearSVC",
        model_version="v2.0.0",
        decision_threshold=DECISION_THRESHOLD,
        max_input_length=MAX_EMAIL_LENGTH,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify Email as Spam or Not Spam",
    description=(
        "Processes the submitted raw email through canonical text normalization "
        "(src.preprocess.normalize_text), transforms it with the frozen TF-IDF vectorizer, "
        "and evaluates it against the LinearSVC decision boundary."
    ),
    responses={
        200: {
            "description": "Successful classification result.",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": "SPAM",
                        "label": 1,
                        "decision_score": 2.4518,
                    }
                }
            },
        },
        422: {
            "description": "Validation error (empty text, whitespace only, or oversized email).",
            "content": {
                "application/json": {
                    "example": {"detail": "Email text cannot be empty or whitespace-only."}
                }
            },
        },
        500: {
            "description": "Unexpected server-side inference failure.",
            "content": {
                "application/json": {
                    "example": {"detail": "An internal inference error occurred. Please try again."}
                }
            },
        },
    },
)
async def predict_spam(request: PredictRequest) -> PredictResponse:
    """
    Classify an email message using the frozen production model.
    
    Email content is processed in memory and never persisted.
    """
    if not ml_artifacts["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is currently unavailable.",
        )

    try:
        model = ml_artifacts["model"]
        vectorizer = ml_artifacts["vectorizer"]

        # Step 1: Canonical Preprocessing
        normalized = normalize_text(request.email)

        # Step 2: TF-IDF Feature Extraction
        X_features = vectorizer.transform([normalized])

        # Step 3: LinearSVC Decision Margin
        # Decision function returns signed distance to separating hyperplane
        decision_scores = model.decision_function(X_features)
        raw_score = float(decision_scores[0])

        # Step 4: Decision Boundary Rule (threshold = 0.0)
        # Class 1 = SPAM (score >= 0.0), Class 0 = NOT SPAM (score < 0.0)
        if raw_score >= DECISION_THRESHOLD:
            prediction_label = 1
            prediction_text = "SPAM"
        else:
            prediction_label = 0
            prediction_text = "NOT SPAM"

        # Round decision score to 4 decimal places for clean representation
        rounded_score = round(raw_score, 4)

        logger.info(
            "Classification completed: %s (label=%d, score=%.4f, length=%d chars)",
            prediction_text,
            prediction_label,
            rounded_score,
            len(request.email),
        )

        return PredictResponse(
            prediction=prediction_text,
            label=prediction_label,
            decision_score=rounded_score,
        )

    except Exception as exc:
        logger.error("Unexpected error during inference: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal inference error occurred. Please try again.",
        ) from None
