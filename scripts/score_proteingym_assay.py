"""Score one ProteinGym assay selected directly or by Slurm-array manifest index."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evotail.data.support_score import load_metadata, write_metadata  # noqa: E402
from evotail.scoring.esm_masked_marginal import EsmMaskedMarginalScorer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="CSV/TSV manifest with DMS_id, assay_path, target_seq, protein_id")
    source.add_argument("--assay", help="Single substitutions assay CSV/TSV")
    parser.add_argument("--array-index", type=int, default=None, help="Zero-based manifest row for Slurm arrays")
    parser.add_argument("--dms-id")
    parser.add_argument("--protein-id")
    parser.add_argument("--target-seq")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def select_assay(args: argparse.Namespace) -> tuple[str, str, str, Path]:
    if args.manifest:
        if args.array_index is None:
            raise ValueError("--array-index is required with --manifest")
        manifest = load_metadata(args.manifest)
        required = {"DMS_id", "assay_path", "target_seq", "protein_id"}
        if missing := required - set(manifest.columns):
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        if not 0 <= args.array_index < len(manifest):
            raise ValueError(f"array index {args.array_index} outside manifest with {len(manifest)} rows")
        row = manifest.iloc[args.array_index]
        return str(row.DMS_id), str(row.protein_id), str(row.target_seq), Path(row.assay_path)
    if args.array_index is not None:
        raise ValueError("--array-index only applies to --manifest")
    if not (args.dms_id and args.protein_id and args.target_seq):
        raise ValueError("single --assay mode requires --dms-id, --protein-id, and --target-seq")
    return args.dms_id, args.protein_id, args.target_seq, Path(args.assay)


def main() -> None:
    args = parse_args()
    dms_id, protein_id, target_seq, assay_path = select_assay(args)
    output_path = Path(args.output_dir) / f"{_safe_filename(dms_id)}.csv"
    if output_path.exists() and not args.overwrite:
        print(f"Skipping existing assay score file: {output_path}")
        return
    variants = load_metadata(assay_path)
    required = {"mutant", "DMS_score"}
    if missing := required - set(variants.columns):
        raise ValueError(f"assay missing columns: {sorted(missing)}")
    scorer = EsmMaskedMarginalScorer(args.model, args.device, args.cache_dir)
    scored = variants.copy()
    scored["evo_score"] = scorer.score_variants(scored, target_seq, args.batch_size)
    scored["DMS_id"] = dms_id
    scored["protein_id"] = protein_id
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_metadata(scored, output_path)
    print(f"Scored {len(scored)} variants for {dms_id} -> {output_path}")


if __name__ == "__main__":
    main()

