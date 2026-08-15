"""Ties the three layers together for a single live request: signal
layer + feature layer -> saved scaler/classifier -> explain. This exact
function is the only place "the verdict" is produced, and it is pure
code + a trained linear model -- no LLM call anywhere in this path.
"""

import json

import joblib
import numpy as np

import features
import signal_layer
from config import CLASSIFIER_ARTIFACT_PATH, FEATURE_SCHEMA_PATH, SCALER_ARTIFACT_PATH
from explain import explain_sentence
from feature_schema import ALL_FEATURE_NAMES

_model = None
_scaler = None
_schema = None


def _load_artifacts():
    global _model, _scaler, _schema
    if _model is None:
        _model = joblib.load(CLASSIFIER_ARTIFACT_PATH)
        _scaler = joblib.load(SCALER_ARTIFACT_PATH)
        _schema = json.loads(FEATURE_SCHEMA_PATH.read_text())
    return _model, _scaler, _schema


def analyze_essay(text: str) -> dict:
    model, scaler, schema = _load_artifacts()

    sentences, sig_feats, doc_level = signal_layer.score_document(text)
    if not sentences:
        return {"sentences": [], "doc_summary": {"n_sentences": 0}}
    _, style_feats = features.extract_document_features(text)

    combined_rows = [{**sf, **tf} for sf, tf in zip(sig_feats, style_feats)]
    X = np.array([[row.get(name, 0.0) for name in ALL_FEATURE_NAMES] for row in combined_rows], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    threshold = schema["flag_threshold"]
    results = []
    for sentence, row, prob in zip(sentences, combined_rows, probs):
        predicted_ai = bool(prob >= threshold)
        reasons = explain_sentence(
            row, schema["feature_names"], schema["logreg_coefficients"],
            schema["scaler_mean"], schema["scaler_scale"], predicted_ai,
        )
        results.append({
            "text": sentence,
            "prob_ai": float(prob),
            "predicted_ai": predicted_ai,
            "reasons": reasons,
        })

    doc_summary = {
        "n_sentences": len(sentences),
        "mean_prob_ai": float(np.mean(probs)),
        "n_flagged": int(sum(r["predicted_ai"] for r in results)),
        "doc_perplexity": doc_level.get("doc_perplexity"),
    }
    return {"sentences": results, "doc_summary": doc_summary}
