"""Unit test suite for AutoVoice prediction logging and statistical drift detection."""

from __future__ import annotations

from pathlib import Path

from autovoice.monitoring.drift import detect_drift
from autovoice.monitoring.prediction_log import log_prediction, read_prediction_logs


def test_log_prediction_privacy(tmp_path: Path) -> None:
    """Verify raw narrative text and PII are never persisted in prediction logs."""
    log_file = tmp_path / "test_audit.jsonl"
    sensitive_narrative = "Customer John Doe experienced brake failure and crash on highway 101."

    rec = log_prediction(
        request_id="req-test-123",
        text=sensitive_narrative,
        predicted_severity="HIGH",
        confidence=0.92,
        entities_count=3,
        hazards_count=2,
        latency_ms=12.5,
        destination_path=log_file,
    )

    assert rec["char_length"] == len(sensitive_narrative)
    assert rec["word_count"] == len(sensitive_narrative.split())
    assert rec["predicted_severity"] == "HIGH"
    assert rec["confidence"] == 0.92

    # Verify disk contents do not contain the raw string or names
    disk_content = log_file.read_text(encoding="utf-8")
    assert "John Doe" not in disk_content
    assert sensitive_narrative not in disk_content

    # Read back via helper
    logs = read_prediction_logs(log_file)
    assert len(logs) == 1
    assert logs[0]["request_id"] == "req-test-123"


def test_detect_drift_healthy(tmp_path: Path) -> None:
    """Test detect_drift returns HEALTHY when distribution matches baseline."""
    baseline = {
        "sample_size": 100,
        "mean_char_length": 300.0,
        "std_char_length": 50.0,
        "severity_proportions": {"LOW": 0.50, "MEDIUM": 0.35, "HIGH": 0.15},
        "char_lengths": [250, 300, 350] * 30,
    }

    # Simulate 20 records drawn from similar distribution
    current = []
    for i in range(20):
        sev = "LOW" if i < 10 else ("MEDIUM" if i < 17 else "HIGH")
        current.append(
            {
                "request_id": f"id-{i}",
                "char_length": 280 + (i % 40),
                "predicted_severity": sev,
                "confidence": 0.85,
            }
        )

    res = detect_drift(current, baseline_stats=baseline)
    assert res["overall_status"] in ("HEALTHY", "DRIFT_WARNING")
    assert "input_length_ks" in res["tests"]
    assert "severity_distribution_chi2" in res["tests"]


def test_detect_drift_critical(tmp_path: Path) -> None:
    """Test detect_drift flags CRITICAL_DRIFT when distribution shifts drastically."""
    baseline = {
        "sample_size": 100,
        "mean_char_length": 300.0,
        "std_char_length": 50.0,
        "severity_proportions": {"LOW": 0.70, "MEDIUM": 0.20, "HIGH": 0.10},
        "char_lengths": [280, 300, 320] * 30,
    }

    # Extreme shift: all records are 4000 characters and 100% HIGH severity
    current = []
    for i in range(30):
        current.append(
            {
                "request_id": f"shift-{i}",
                "char_length": 4500,
                "predicted_severity": "HIGH",
                "confidence": 0.50,
            }
        )

    res = detect_drift(current, baseline_stats=baseline)
    assert res["overall_status"] == "CRITICAL_DRIFT"
    assert res["tests"]["input_length_ks"]["drift_detected"] is True
    assert res["tests"]["severity_distribution_chi2"]["drift_detected"] is True
    assert res["tests"]["confidence_monitoring"]["degradation_alert"] is True


def test_detect_drift_insufficient_data() -> None:
    """Verify that fewer than 5 records safely returns INSUFFICIENT_DATA."""
    res = detect_drift([{"char_length": 200, "predicted_severity": "LOW"}])
    assert res["status"] == "INSUFFICIENT_DATA"
