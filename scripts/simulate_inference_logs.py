"""Simulate production inference telemetry and generate a verified drift report."""

from __future__ import annotations

import json
import uuid

import pandas as pd

from autovoice.classifier.infer import SeverityClassifierInferencer
from autovoice.config import DATA_PROCESSED_DIR
from autovoice.monitoring.drift import run_drift_pipeline
from autovoice.monitoring.prediction_log import log_prediction

# Load sample records from processed data
df = pd.read_parquet(DATA_PROCESSED_DIR / "complaints.parquet")
sample_df = df.sample(min(30, len(df)), random_state=42)

clf = SeverityClassifierInferencer()
clf.load()

for _, row in sample_df.iterrows():
    text = str(row["narrative"])
    pred = clf.predict(text)
    log_prediction(
        request_id=f"sim-{uuid.uuid4().hex[:8]}",
        text=text,
        predicted_severity=pred["predicted_severity"],
        confidence=pred["confidence"],
        entities_count=len(pred["hazard_indicators"]) + 1,
        hazards_count=len(pred["hazard_indicators"]),
        latency_ms=15.4,
    )

report = run_drift_pipeline()
print("Drift report successfully generated:")
print(json.dumps(report, indent=2))
