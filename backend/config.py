"""Shared paths and constants for the AI essay detector backend."""

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

for d in (DATA_DIR, ARTIFACTS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, DOCUMENTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Signal-layer model: small, local, open. "GPT-2 class is enough to start"
SIGNAL_MODEL_NAME = "gpt2"
# Second model for the Binoculars-style cross-model ratio (stretch feature).
SIGNAL_MODEL_NAME_B = "gpt2-medium"

MAX_MODEL_TOKENS = 1024          # gpt2 context window
SLIDING_WINDOW_STRIDE = 512      # overlap when a doc exceeds the window

RANDOM_SEED = 42

CLASSIFIER_ARTIFACT_PATH = ARTIFACTS_DIR / "classifier.joblib"
SCALER_ARTIFACT_PATH = ARTIFACTS_DIR / "scaler.joblib"
FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
HONESTY_REPORT_PATH = DOCUMENTS_DIR / "HONESTY_REPORT.md"
