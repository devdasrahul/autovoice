"""Structured prediction audit logger for AutoVoice inference services.

Enforces GDPR & privacy compliance: NEVER logs raw user narrative text or PII.
Logs strictly anonymous statistical telemetry: character length, token count,
predicted severity, model confidence, entity count, and latency.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autovoice.config import REPORTS_DIR
from autovoice.logging_conf import get_logger

log = get_logger(__name__)

DEFAULT_LOG_PATH = REPORTS_DIR / "prediction_audit.jsonl"


def log_prediction(
    *,
    request_id: str,
    text: str,
    predicted_severity: str,
    confidence: float,
    entities_count: int = 0,
    hazards_count: int = 0,
    latency_ms: float = 0.0,
    destination_path: Path | str | None = None,
) -> dict[str, Any]:
    """Append structured telemetry record to audit log without raw narrative text.

    Args:
        request_id: Request correlation identifier.
        text: Input text (only length and token count are derived; raw text is discarded).
        predicted_severity: Predicted triage severity (LOW, MEDIUM, HIGH).
        confidence: Prediction confidence score [0, 1].
        entities_count: Number of extracted NER spans.
        hazards_count: Number of safety hazard flags identified.
        latency_ms: Inference execution time in milliseconds.
        destination_path: Path to target JSONL file.

    Returns:
        The logged telemetry dictionary.
    """
    dest = Path(destination_path) if destination_path else DEFAULT_LOG_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    words = text.split()
    record = {
        "request_id": request_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "char_length": len(text),
        "word_count": len(words),
        "predicted_severity": predicted_severity,
        "confidence": round(float(confidence), 4),
        "entities_count": int(entities_count),
        "hazards_count": int(hazards_count),
        "latency_ms": round(float(latency_ms), 2),
    }

    try:
        with open(dest, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        log.error("prediction_log.write_failed", error=str(exc), path=str(dest))

    return record


def read_prediction_logs(source_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Read all structured prediction audit records from JSONL file."""
    src = Path(source_path) if source_path else DEFAULT_LOG_PATH
    if not src.exists():
        return []

    records = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records
