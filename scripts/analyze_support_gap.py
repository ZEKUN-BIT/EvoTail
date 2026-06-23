"""Merge ProteinGym score files with support evidence and run Phase 1 statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evotail.analysis.figures import plot_support_gap  # noqa: E402
from evotail.analysis.support_gap import run_support_gap_analysis  # noqa: E402
from evotail.data.support_score import load_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="Merged score table or directory of per-assay files")
    parser.add_argument("--support-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--assay-metadata")
    parser.add_argument("--min-variants", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--skip-figure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_support_gap_analysis(
        score_path=args.scores,
        support_path=args.support_table,
        output_dir=args.output_dir,
        min_variants=args.min_variants,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        assay_metadata_path=args.assay_metadata,
    )
    if not args.skip_figure:
        output_dir = Path(args.output_dir)
        plot_support_gap(
            load_metadata(output_dir / "per_assay_spearman.csv"),
            load_metadata(output_dir / "support_stratum_summary.csv"),
            output_dir / "figure1_support_gap.pdf",
        )
    print(f"Analysed {stats['n_assays']} assays; support gap={stats['support_gap']}")


if __name__ == "__main__":
    main()
