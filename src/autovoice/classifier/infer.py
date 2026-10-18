"""Inference engine for AutoVoice Severity Classification.

Provides fast, CPU-friendly inference returning predicted severity class,
calibrated probability distribution, and identified hazard indicators.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from autovoice.config import MODELS_DIR
from autovoice.data.labeling import SEVERITY_CLASSES
from autovoice.logging_conf import get_logger

log = get_logger(__name__)

# Heuristic hazard terms for explainable risk annotation
HAZARD_PATTERNS = {
    "crash": re.compile(r"\b(crash(ed)?|collision|guardrail|rollover|struck)\b", re.I),
    "fire": re.compile(r"\b(fire|flames|smoke|smoking|burning|thermal runaway)\b", re.I),
    "injury": re.compile(r"\b(injur(y|ed|ies)|hospital|ambulance|concussion)\b", re.I),
    "loss_of_control": re.compile(r"\b(steering (lock|stiff)|brakes? (failed|loss))\b", re.I),
    "sudden_stall": re.compile(
        r"\b(stalled?|shut off|died) (while driving|at highway speed|on highway)\b", re.I
    ),
}


class SeverityClassifierInferencer:
    """Lazy-loading severity classifier inference engine (CPU-optimized)."""

    def __init__(self, model_dir: Path | str | None = None):
        """Initialize inferencer with model directory."""
        self.model_dir = Path(model_dir) if model_dir else (MODELS_DIR / "clf")
        self._vectorizer = None
        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Load vectorizer and classifier from disk (idempotent)."""
        if self._loaded:
            return

        vec_path = self.model_dir / "vectorizer.pkl"
        clf_path = self.model_dir / "model.pkl"

        if vec_path.exists() and clf_path.exists():
            log.info("classifier.loading", path=str(self.model_dir))
            with open(vec_path, "rb") as f:
                self._vectorizer = pickle.load(f)
            with open(clf_path, "rb") as f:
                self._model = pickle.load(f)
            self._loaded = True
        else:
            log.warning("classifier.artifacts_missing_using_heuristics", path=str(self.model_dir))
            self._loaded = False

    def warmup(self) -> None:
        """Run a single prediction to warm up CPU and caches on startup."""
        self.predict("Vehicle experienced brake failure and crash on highway.")

    def predict(self, text: str) -> dict[str, Any]:
        """Predict severity class and probabilities for a complaint narrative.

        Args:
            text: Complaint narrative text.

        Returns:
            Dictionary with predicted_severity, confidence, probabilities, and hazards.
        """
        self.load()

        # Detect explainable domain hazards
        hazards = [name for name, pattern in HAZARD_PATTERNS.items() if pattern.search(text)]

        if self._loaded and self._vectorizer and self._model:
            x_vec = self._vectorizer.transform([text])
            probs = self._model.predict_proba(x_vec)[0]
            classes = self._model.classes_

            prob_dict = {cls_name: float(p) for cls_name, p in zip(classes, probs, strict=False)}
            # Ensure all canonical classes are present
            for c in SEVERITY_CLASSES:
                prob_dict.setdefault(c, 0.0)

            pred_class = str(self._model.predict(x_vec)[0])
            confidence = float(prob_dict[pred_class])
        else:
            # Fallback heuristic prediction when model is not yet trained
            if any(h in hazards for h in ("fire", "injury")):
                pred_class = "HIGH"
                prob_dict = {"LOW": 0.05, "MEDIUM": 0.15, "HIGH": 0.80}
            elif "crash" in hazards or "loss_of_control" in hazards:
                pred_class = "MEDIUM"
                prob_dict = {"LOW": 0.15, "MEDIUM": 0.70, "HIGH": 0.15}
            else:
                pred_class = "LOW"
                prob_dict = {"LOW": 0.80, "MEDIUM": 0.15, "HIGH": 0.05}
            confidence = prob_dict[pred_class]

        return {
            "predicted_severity": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "hazard_indicators": hazards,
        }
