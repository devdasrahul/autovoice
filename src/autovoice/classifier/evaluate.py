"""Evaluation module for AutoVoice Severity Classifier.

Computes precision, recall, F1 (macro/weighted), confusion matrix, and calibration
metrics against the held-out test split, writing reports/clf_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from autovoice.config import MODELS_DIR, REPORTS_DIR
from autovoice.data.labeling import SEVERITY_CLASSES
from autovoice.logging_conf import get_logger

log = get_logger(__name__)

DEFAULT_METRICS_PATH = REPORTS_DIR / "clf_metrics.json"


def evaluate_model(
    model_dir: Path,
    test_split_path: Path | None = None,
    out_json: Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    """Evaluate severity classifier on held-out test data.

    Args:
        model_dir: Directory containing model.pkl, vectorizer.pkl, and test_split.json.
        test_split_path: Optional explicit path to test_split.json.
        out_json: Path to save the evaluation metrics JSON.

    Returns:
        Dictionary of computed metrics.
    """
    vec_path = model_dir / "vectorizer.pkl"
    clf_path = model_dir / "model.pkl"
    split_p = test_split_path or (model_dir / "test_split.json")

    if not vec_path.exists() or not clf_path.exists():
        raise FileNotFoundError(f"Model artifacts not found in {model_dir}. Run training first.")
    if not split_p.exists():
        raise FileNotFoundError(f"Test split not found at {split_p}. Run training first.")

    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(clf_path, "rb") as f:
        model = pickle.load(f)

    split = json.loads(split_p.read_text(encoding="utf-8"))
    x_test = split["x_test"]
    y_test = split["y_test"]

    x_test_vec = vectorizer.transform(x_test)
    y_pred = model.predict(x_test_vec)
    y_probs = model.predict_proba(x_test_vec)

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=list(SEVERITY_CLASSES),
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(SEVERITY_CLASSES)).tolist()

    # Mean predicted confidence
    mean_confidence = float(np.mean(np.max(y_probs, axis=1)))

    results: dict[str, Any] = {
        "model_type": "tfidf_logreg_baseline",
        "test_samples": len(y_test),
        "classes": list(SEVERITY_CLASSES),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "mean_confidence": mean_confidence,
        "per_class": {
            cls_name: {
                "precision": float(report[cls_name]["precision"]),
                "recall": float(report[cls_name]["recall"]),
                "f1_score": float(report[cls_name]["f1-score"]),
                "support": int(report[cls_name]["support"]),
            }
            for cls_name in SEVERITY_CLASSES
            if cls_name in report
        },
        "confusion_matrix": {
            "labels": list(SEVERITY_CLASSES),
            "matrix": cm,
        },
        "calibration_notes": (
            "Model is calibrated using class-weighted balanced logistic regression "
            "to account for severe NHTSA indicator class imbalance."
        ),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("classifier.metrics_written", path=str(out_json), macro_f1=results["macro_f1"])
    return results


def main() -> None:
    """CLI entry point for severity classifier evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate AutoVoice Severity Classifier")
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR / "clf")
    parser.add_argument("--test-split", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()

    res = evaluate_model(args.model_dir, args.test_split, args.out_json)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
