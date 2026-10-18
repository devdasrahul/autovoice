"""Unit tests for Phase-4 Severity Classification and Labeling."""

from __future__ import annotations

import pandas as pd
import pytest

from autovoice.classifier.infer import SeverityClassifierInferencer
from autovoice.data.labeling import (
    SEVERITY_CLASSES,
    add_severity_labels,
    derive_severity_label,
)


class TestSeverityLabeling:
    def test_derive_severity_high_on_fire_injury_death(self):
        assert derive_severity_label({"fire": 1, "injured": 0, "deaths": 0, "crash": 0}) == "HIGH"
        assert derive_severity_label({"fire": 0, "injured": 2, "deaths": 0, "crash": 0}) == "HIGH"
        assert derive_severity_label({"fire": 0, "injured": 0, "deaths": 1, "crash": 0}) == "HIGH"

    def test_derive_severity_medium_on_crash(self):
        assert derive_severity_label({"fire": 0, "injured": 0, "deaths": 0, "crash": 1}) == "MEDIUM"

    def test_derive_severity_low_otherwise(self):
        assert derive_severity_label({"fire": 0, "injured": 0, "deaths": 0, "crash": 0}) == "LOW"

    def test_add_severity_labels_columns(self):
        df = pd.DataFrame(
            [
                {
                    "narrative": "Airbag warning light on dashboard.",
                    "crash": 0,
                    "fire": 0,
                    "injured": 0,
                    "deaths": 0,
                },
                {
                    "narrative": "Car caught fire while parked.",
                    "crash": 0,
                    "fire": 1,
                    "injured": 0,
                    "deaths": 0,
                },
                {
                    "narrative": "Vehicle collided with guardrail.",
                    "crash": 1,
                    "fire": 0,
                    "injured": 0,
                    "deaths": 0,
                },
            ]
        )
        labeled = add_severity_labels(df)
        assert "severity" in labeled.columns
        assert "severity_id" in labeled.columns
        assert list(labeled["severity"]) == ["LOW", "HIGH", "MEDIUM"]


class TestSeverityInferencer:
    def test_heuristic_fallback(self, tmp_path):
        inferencer = SeverityClassifierInferencer(model_dir=tmp_path / "empty")
        res_high = inferencer.predict("The vehicle battery caught fire and began smoking.")
        assert res_high["predicted_severity"] == "HIGH"
        assert "fire" in res_high["hazard_indicators"]

        res_med = inferencer.predict(
            "Vehicle was involved in a collision and crashed into a barrier."
        )
        assert res_med["predicted_severity"] == "MEDIUM"
        assert "crash" in res_med["hazard_indicators"]

        res_low = inferencer.predict("Radio touchscreen freezes occasionally.")
        assert res_low["predicted_severity"] == "LOW"

    def test_loaded_model_predict_structure(self):
        inferencer = SeverityClassifierInferencer()
        res = inferencer.predict("The vehicle battery caught fire.")
        assert "predicted_severity" in res
        assert "probabilities" in res
        assert "confidence" in res
        assert "hazard_indicators" in res

    def test_probability_distribution_contract(self):
        inferencer = SeverityClassifierInferencer(model_dir=None)
        res = inferencer.predict("Engine stalled on highway causing near miss.")
        assert "probabilities" in res
        probs = res["probabilities"]
        for c in SEVERITY_CLASSES:
            assert c in probs
            assert 0.0 <= probs[c] <= 1.0
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-2)
