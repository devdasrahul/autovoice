"""Rule-derived proxy severity label generation for automotive complaints.

Severity rules (strictly uses indicator metadata, never narrative tokens to prevent leakage):
    HIGH: Any reported fatalities, injuries, or vehicle fire (life-safety critical).
    MEDIUM: Crash reported with no fire or injuries (property damage / kinetic event).
    LOW: Operational/component issue without crash, fire, injury, or death.

Disclosure:
    These are rule-derived proxy labels from consumer-reported indicators, NOT verified
    engineering failure determinations or court findings.
"""

from __future__ import annotations

import pandas as pd

from autovoice.logging_conf import get_logger

log = get_logger(__name__)

SEVERITY_CLASSES: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")
SEVERITY2ID: dict[str, int] = {name: i for i, name in enumerate(SEVERITY_CLASSES)}
ID2SEVERITY: dict[int, str] = dict(enumerate(SEVERITY_CLASSES))


def derive_severity_label(row: pd.Series | dict) -> str:
    """Derive proxy severity label from indicator fields.

    Args:
        row: Series or dict containing 'deaths', 'injured', 'fire', 'crash'.

    Returns:
        One of 'LOW', 'MEDIUM', 'HIGH'.
    """
    deaths = int(row.get("deaths") or 0)
    injured = int(row.get("injured") or 0)
    fire = int(row.get("fire") or 0)
    crash = int(row.get("crash") or 0)

    if deaths > 0 or injured > 0 or fire > 0:
        return "HIGH"
    if crash > 0:
        return "MEDIUM"
    return "LOW"


def add_severity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'severity' and 'severity_id' columns to a dataframe based on indicator fields.

    Args:
        df: DataFrame containing indicator fields.

    Returns:
        DataFrame with new 'severity' and 'severity_id' columns.
    """
    out = df.copy()
    out["severity"] = out.apply(derive_severity_label, axis=1)
    out["severity_id"] = out["severity"].map(SEVERITY2ID)
    counts = out["severity"].value_counts().to_dict()
    log.info("labeling.severity_derived", counts=counts)
    return out
