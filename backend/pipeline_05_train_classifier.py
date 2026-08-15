"""
Pipeline stage 5: the decision layer. Trains the classifier whose output
IS the verdict -- nothing downstream asks an LLM to confirm or override
it. Two models are trained for comparison (logistic regression and a
gradient-boosted tree); logistic regression ships in the live app
because its coefficients give an exact, auditable per-feature
contribution for every prediction, which is what explain.py uses to
generate the "why" for each flagged sentence. Also writes the honesty
report: held-out precision/recall/F1, confidently-wrong examples, and
the non-native-English over-flagging probe.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from config import (
    ARTIFACTS_DIR, CLASSIFIER_ARTIFACT_PATH, FEATURE_SCHEMA_PATH,
    PROCESSED_DATA_DIR, RANDOM_SEED, SCALER_ARTIFACT_PATH,
)
from feature_schema import ALL_FEATURE_NAMES

FEATURES_PATH = PROCESSED_DATA_DIR / "sentence_features.parquet"
ELL_FEATURES_PATH = PROCESSED_DATA_DIR / "ell_sentence_features.parquet"
GBT_ARTIFACT_PATH = ARTIFACTS_DIR / "gbt_classifier.joblib"

FLAG_THRESHOLD = 0.5


def load_xy(df: pd.DataFrame):
    X = df[ALL_FEATURE_NAMES].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].to_numpy(dtype=int)
    return X, y


def metrics_block(y_true, y_prob, threshold=FLAG_THRESHOLD) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


def find_confidently_wrong_docs(df_test: pd.DataFrame, y_prob: np.ndarray, k: int = 3) -> list[dict]:
    """Aggregate sentence-level predictions to the document level for
    PURE documents only (whole essay is genuinely all-human or all-AI --
    excludes polished-mixed docs, where a doc-level label doesn't mean
    'the detector was wrong', since the doc IS partially AI by design).
    """
    df = df_test.copy()
    df["pred_prob"] = y_prob
    pure_buckets = {"daigt_human", "daigt_ai", "self_ai_fresh", "ell_nonnative"}
    pure = df[df["bucket"].isin(pure_buckets)]

    doc_level = pure.groupby("doc_id").agg(
        mean_prob=("pred_prob", "mean"),
        true_label=("label", "first"),
        bucket=("bucket", "first"),
        n_sentences=("pred_prob", "count"),
        text_preview=("sentence_text", lambda s: " ".join(s)[:400]),
    ).reset_index()

    doc_level["pred_label"] = (doc_level["mean_prob"] >= 0.5).astype(int)
    wrong = doc_level[doc_level["pred_label"] != doc_level["true_label"]].copy()
    wrong["confidence"] = (wrong["mean_prob"] - 0.5).abs()
    wrong = wrong.sort_values("confidence", ascending=False).head(k)
    return wrong.to_dict(orient="records")


def ell_bias_probe(scaler, model) -> dict | None:
    if not ELL_FEATURES_PATH.exists():
        return None
    ell = pd.read_parquet(ELL_FEATURES_PATH)
    X_ell, y_ell = load_xy(ell)
    X_ell_scaled = scaler.transform(X_ell)
    prob_ell = model.predict_proba(X_ell_scaled)[:, 1]
    ell_flag_rate = float((prob_ell >= FLAG_THRESHOLD).mean())

    main = pd.read_parquet(FEATURES_PATH)
    native_human_test = main[(main["split"] == "test") & (main["bucket"] == "daigt_human")]
    X_nat, y_nat = load_xy(native_human_test)
    prob_nat = model.predict_proba(scaler.transform(X_nat))[:, 1]
    native_flag_rate = float((prob_nat >= FLAG_THRESHOLD).mean())

    return {
        "ell_n_sentences": int(len(ell)),
        "ell_n_essays": int(ell["doc_id"].nunique()),
        "ell_false_positive_rate": ell_flag_rate,
        "native_human_n_sentences": int(len(native_human_test)),
        "native_human_false_positive_rate": native_flag_rate,
        "gap_percentage_points": (ell_flag_rate - native_flag_rate) * 100,
    }


def main():
    df = pd.read_parquet(FEATURES_PATH)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    print(f"train sentences: {len(train)} | test sentences: {len(test)}")

    X_train, y_train = load_xy(train)
    X_test, y_test = load_xy(test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=RANDOM_SEED)
    logreg.fit(X_train_s, y_train)
    prob_test_lr = logreg.predict_proba(X_test_s)[:, 1]
    lr_metrics = metrics_block(y_test, prob_test_lr)
    print("Logistic regression:", json.dumps({k: v for k, v in lr_metrics.items() if k != "confusion_matrix"}, indent=2))

    gbt = HistGradientBoostingClassifier(random_state=RANDOM_SEED, max_iter=200)
    gbt.fit(X_train_s, y_train)
    prob_test_gbt = gbt.predict_proba(X_test_s)[:, 1]
    gbt_metrics = metrics_block(y_test, prob_test_gbt)
    print("Gradient boosted trees:", json.dumps({k: v for k, v in gbt_metrics.items() if k != "confusion_matrix"}, indent=2))

    joblib.dump(logreg, CLASSIFIER_ARTIFACT_PATH)
    joblib.dump(gbt, GBT_ARTIFACT_PATH)
    joblib.dump(scaler, SCALER_ARTIFACT_PATH)

    schema = {
        "feature_names": ALL_FEATURE_NAMES,
        "logreg_coefficients": logreg.coef_[0].tolist(),
        "logreg_intercept": float(logreg.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "flag_threshold": FLAG_THRESHOLD,
    }
    FEATURE_SCHEMA_PATH.write_text(json.dumps(schema, indent=2))
    print(f"Saved classifier artifacts -> {ARTIFACTS_DIR}")

    wrong_examples = find_confidently_wrong_docs(test, prob_test_lr, k=3)
    bias_probe = ell_bias_probe(scaler, logreg)

    report = {
        "logreg_metrics": lr_metrics,
        "gbt_metrics": gbt_metrics,
        "confidently_wrong_examples": wrong_examples,
        "ell_bias_probe": bias_probe,
        "n_train_sentences": len(train),
        "n_test_sentences": len(test),
        "n_train_docs": train["doc_id"].nunique(),
        "n_test_docs": test["doc_id"].nunique(),
    }
    (PROCESSED_DATA_DIR / "eval_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("\nSaved raw eval report -> processed/eval_report.json")
    if bias_probe:
        print(f"\nELL false-positive rate: {bias_probe['ell_false_positive_rate']:.3f} vs native-human {bias_probe['native_human_false_positive_rate']:.3f}")


if __name__ == "__main__":
    main()
