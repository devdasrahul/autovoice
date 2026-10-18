"""Training pipeline for AutoVoice Severity Classification.

Trains:
1. Primary baseline: TF-IDF + class-weighted LogisticRegression
   (fast, deterministic, strong baseline).
2. Transformer sequence classifier: DistilBERT SequenceClassification with class weighting.

Data:
    Inputs: 'narrative' text only (zero indicator leakage).
    Labels: 'severity' ('LOW', 'MEDIUM', 'HIGH') derived from NHTSA indicators.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from autovoice.config import DATA_PROCESSED_DIR, MODELS_DIR
from autovoice.data.labeling import SEVERITY_CLASSES, add_severity_labels
from autovoice.logging_conf import get_logger

log = get_logger(__name__)

DEFAULT_OUTPUT_DIR = MODELS_DIR / "clf"


@dataclass
class ClfTrainConfig:
    """Hyperparameters and configuration for severity classifier training."""

    model_type: str = "tfidf_logreg"  # 'tfidf_logreg' or 'distilbert'
    output_dir: Path = DEFAULT_OUTPUT_DIR
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    c_reg: float = 1.0


def train_baseline(
    df: pd.DataFrame, cfg: ClfTrainConfig
) -> tuple[TfidfVectorizer, LogisticRegression, dict[str, Any]]:
    """Train TF-IDF + LogisticRegression baseline with class weighting.

    Args:
        df: Cleaned dataframe with narratives and indicators.
        cfg: Training configuration parameters.

    Returns:
        Tuple of (vectorizer, classifier, test_split_dict).
    """
    labeled_df = add_severity_labels(df)
    labeled_df = labeled_df[labeled_df["narrative"].str.len() > 10].reset_index(drop=True)

    x_train, x_test, y_train, y_test = train_test_split(
        labeled_df["narrative"],
        labeled_df["severity"],
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labeled_df["severity"],
    )

    vectorizer = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        sublinear_tf=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train)

    classes = np.array(SEVERITY_CLASSES)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights, strict=False))

    model = LogisticRegression(
        C=cfg.c_reg,
        class_weight=class_weight_dict,
        max_iter=1000,
        random_state=cfg.random_state,
    )
    model.fit(x_train_vec, y_train)

    log.info(
        "classifier.baseline_trained",
        train_rows=len(x_train),
        test_rows=len(x_test),
        vocab_size=len(vectorizer.vocabulary_),
    )

    split_data = {
        "x_test": x_test.tolist(),
        "y_test": y_test.tolist(),
    }
    return vectorizer, model, split_data


def save_baseline(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    cfg: ClfTrainConfig,
    out_dir: Path | None = None,
) -> Path:
    """Save vectorizer, model, and metadata to output directory.

    Args:
        vectorizer: Fitted TfidfVectorizer.
        model: Fitted LogisticRegression model.
        cfg: Configuration used during training.
        out_dir: Destination path; defaults to cfg.output_dir.

    Returns:
        Path to output directory.
    """
    dest = out_dir or cfg.output_dir
    dest.mkdir(parents=True, exist_ok=True)

    with open(dest / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(dest / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    config_dict = asdict(cfg)
    config_dict["output_dir"] = str(dest)
    config_dict["classes"] = list(SEVERITY_CLASSES)
    (dest / "clf_config.json").write_text(json.dumps(config_dict, indent=2))
    log.info("classifier.baseline_saved", path=str(dest))
    return dest


def main() -> None:
    """CLI entry point for training severity classifier."""
    parser = argparse.ArgumentParser(description="Train AutoVoice Severity Classifier")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PROCESSED_DIR / "complaints.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"{args.data_path} not found. Run `make data` first.")

    df = pd.read_parquet(args.data_path)
    cfg = ClfTrainConfig(output_dir=args.output_dir)
    vec, model, split_data = train_baseline(df, cfg)
    save_baseline(vec, model, cfg)

    # Cache test split for evaluation
    test_split_path = args.output_dir / "test_split.json"
    test_split_path.write_text(json.dumps(split_data, indent=2))
    log.info("classifier.test_split_cached", path=str(test_split_path))


if __name__ == "__main__":
    main()
