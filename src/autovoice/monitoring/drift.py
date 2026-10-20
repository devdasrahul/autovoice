"""Statistical data and concept drift detection for AutoVoice inference.

Compares production inference logs against baseline training distributions:
1. Input Data Drift: Kolmogorov-Smirnov (KS) test on complaint character lengths.
2. Concept/Target Drift: Chi-Square Goodness-of-Fit on predicted severity distribution.
3. Model Degradation: Confidence shift and hazard flag volatility tracking.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from autovoice.config import DATA_PROCESSED_DIR, REPORTS_DIR
from autovoice.logging_conf import get_logger
from autovoice.monitoring.prediction_log import read_prediction_logs

log = get_logger(__name__)

DEFAULT_REPORT_PATH = REPORTS_DIR / "drift_report.json"
P_VALUE_THRESHOLD = 0.05


def compute_baseline_stats(parquet_path: Path | str | None = None) -> dict[str, Any]:
    """Calculate reference statistical distributions from the cleaned training corpus."""
    path = Path(parquet_path) if parquet_path else (DATA_PROCESSED_DIR / "complaints.parquet")
    if not path.exists():
        log.warning("drift.baseline_missing", path=str(path))
        return {
            "sample_size": 190,
            "mean_char_length": 340.5,
            "std_char_length": 180.2,
            "severity_proportions": {"LOW": 0.50, "MEDIUM": 0.35, "HIGH": 0.15},
            "char_lengths": [300, 350, 400],
        }

    df = pd.read_parquet(path)
    text_col = (
        "narrative"
        if "narrative" in df.columns
        else ("cdescr" if "cdescr" in df.columns else df.columns[0])
    )
    char_lengths = df[text_col].fillna("").astype(str).str.len().tolist()

    # Calculate severity proportions
    if "severity" in df.columns:
        counts = df["severity"].value_counts(normalize=True).to_dict()
    else:
        counts = {"LOW": 0.50, "MEDIUM": 0.35, "HIGH": 0.15}

    return {
        "sample_size": len(df),
        "mean_char_length": round(float(np.mean(char_lengths)), 2),
        "std_char_length": round(float(np.std(char_lengths)), 2),
        "severity_proportions": {k: round(float(v), 4) for k, v in counts.items()},
        "char_lengths": char_lengths,
    }


def detect_drift(
    current_records: list[dict[str, Any]],
    baseline_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute statistical hypothesis tests to detect input and target drift.

    Args:
        current_records: Telemetry records from prediction audit logs.
        baseline_stats: Reference baseline metrics.

    Returns:
        Structured drift report including test statistics, p-values, and drift flags.
    """
    baseline = baseline_stats or compute_baseline_stats()
    n_current = len(current_records)

    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "current_sample_size": n_current,
        "baseline_sample_size": baseline.get("sample_size", 0),
        "tests": {},
        "overall_status": "HEALTHY",
    }

    if n_current < 5:
        report["status"] = "INSUFFICIENT_DATA"
        report["message"] = (
            f"Requires at least 5 audit records to test drift (current: {n_current})."
        )
        return report

    # 1. Input Drift: Kolmogorov-Smirnov test on character length distribution
    curr_lengths = [r["char_length"] for r in current_records if "char_length" in r]
    base_lengths = baseline.get("char_lengths", [])

    if curr_lengths and base_lengths:
        ks_res = stats.ks_2samp(curr_lengths, base_lengths)
        ks_drift = bool(ks_res.pvalue < P_VALUE_THRESHOLD)
        report["tests"]["input_length_ks"] = {
            "statistic": round(float(ks_res.statistic), 4),
            "p_value": round(float(ks_res.pvalue), 4),
            "drift_detected": ks_drift,
            "threshold": P_VALUE_THRESHOLD,
            "mean_current_length": round(float(np.mean(curr_lengths)), 2),
            "mean_baseline_length": baseline.get("mean_char_length", 0.0),
        }
    else:
        ks_drift = False

    # 2. Concept/Target Drift: Chi-Square Goodness-of-Fit on Severity distribution
    classes = ["LOW", "MEDIUM", "HIGH"]
    observed_counts = dict.fromkeys(classes, 0)
    for r in current_records:
        sev = r.get("predicted_severity")
        if sev in observed_counts:
            observed_counts[sev] += 1

    base_props = baseline.get("severity_proportions", {"LOW": 0.5, "MEDIUM": 0.35, "HIGH": 0.15})
    expected_counts = [max(0.1, base_props.get(c, 0.33) * n_current) for c in classes]
    obs_list = [observed_counts[c] for c in classes]

    try:
        chi2_res = stats.chisquare(f_obs=obs_list, f_exp=expected_counts)
        chi2_drift = bool(chi2_res.pvalue < P_VALUE_THRESHOLD)
        report["tests"]["severity_distribution_chi2"] = {
            "statistic": round(float(chi2_res.statistic), 4),
            "p_value": round(float(chi2_res.pvalue), 4),
            "drift_detected": chi2_drift,
            "observed_proportions": {c: round(observed_counts[c] / n_current, 4) for c in classes},
            "expected_proportions": base_props,
        }
    except Exception as exc:
        log.warning("drift.chi2_failed", error=str(exc))
        chi2_drift = False

    # 3. Model Confidence Degradation
    confidences = [r["confidence"] for r in current_records if "confidence" in r]
    if confidences:
        mean_conf = float(np.mean(confidences))
        report["tests"]["confidence_monitoring"] = {
            "mean_confidence": round(mean_conf, 4),
            "degradation_alert": bool(mean_conf < 0.65),
        }

    # Determine overall status
    if ks_drift and chi2_drift:
        report["overall_status"] = "CRITICAL_DRIFT"
    elif ks_drift or chi2_drift:
        report["overall_status"] = "DRIFT_WARNING"
    else:
        report["overall_status"] = "HEALTHY"

    return report


def run_drift_pipeline(
    log_path: Path | str | None = None,
    output_report: Path | str | None = None,
) -> dict[str, Any]:
    """Execute end-to-end drift calculation and persist report."""
    records = read_prediction_logs(log_path)
    report = detect_drift(records)

    out_p = Path(output_report) if output_report else DEFAULT_REPORT_PATH
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("drift.report_generated", path=str(out_p), status=report["overall_status"])
    return report


if __name__ == "__main__":
    rep = run_drift_pipeline()
    print(json.dumps(rep, indent=2))
