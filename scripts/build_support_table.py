"""Build an auditable evolutionary-support table from precomputed metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evotail.data.support_score import (  # noqa: E402
    SupportScoreConfig,
    compute_support_table,
    load_metadata,
    write_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV, TSV, or Parquet metadata table")
    parser.add_argument("--output", required=True, help="CSV, TSV, or Parquet output table")
    parser.add_argument("--strata", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--minimum-weight", type=float, default=0.25)
    parser.add_argument("--maximum-weight", type=float, default=4.0)
    parser.add_argument("--allow-ambiguous-amino-acids", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SupportScoreConfig(
        strata=args.strata,
        alpha=args.alpha,
        minimum_weight=args.minimum_weight,
        maximum_weight=args.maximum_weight,
        allow_ambiguous_amino_acids=args.allow_ambiguous_amino_acids,
    )
    support_table = compute_support_table(load_metadata(args.input), config)
    write_metadata(support_table, args.output)
    print(
        f"Wrote {len(support_table)} proteins to {args.output}; "
        f"support range={support_table.evolutionary_support_score.min():.3f}.."
        f"{support_table.evolutionary_support_score.max():.3f}"
    )


if __name__ == "__main__":
    main()

