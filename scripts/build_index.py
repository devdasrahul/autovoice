"""Build persistent vector index from cleaned complaints parquet (idempotent)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from autovoice.config import DATA_PROCESSED_DIR
from autovoice.logging_conf import get_logger
from autovoice.rag.index import ComplaintVectorIndex

log = get_logger(__name__)


def main() -> None:
    """CLI script to build ChromaDB index."""
    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector index from complaints.parquet"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PROCESSED_DIR / "complaints.parquet",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DATA_PROCESSED_DIR / "chroma",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        log.error("build_index.data_missing", path=str(args.data_path))
        print(f"Error: {args.data_path} not found. Run `make data` first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(args.data_path)
    index = ComplaintVectorIndex(persist_dir=args.index_dir)
    count = index.build_from_dataframe(df)

    log.info("build_index.complete", indexed=count, dir=str(args.index_dir))
    print(f"OK: Successfully indexed {count} complaints into {args.index_dir}")


if __name__ == "__main__":
    main()
