"""Statistics for the Phase 1 evolutionary-support gap analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evotail.data.support_score import load_metadata, write_metadata


class SupportGapError(ValueError):
    """Raised for malformed Phase 1 score tables."""


def spearman_correlation(left: pd.Series, right: pd.Series) -> float:
    """Compute Spearman correlation without requiring SciPy."""

    paired = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(paired) < 2 or paired["left"].nunique() < 2 or paired["right"].nunique() < 2:
        return float("nan")
    return float(paired["left"].rank(method="average").corr(paired["right"].rank(method="average")))


def mutation_depth(mutant: object) -> int:
    """Count colon-delimited substitutions, treating WT as zero mutations."""

    value = str(mutant).strip().upper()
    if value in {"", "WT", "NAN"}:
        return 0
    return len([token for token in value.split(":") if token])


def load_score_tables(path: str | Path) -> pd.DataFrame:
    """Load one merged score table or all CSV/TSV files in a score directory."""

    path = Path(path)
    if path.is_file():
        return load_metadata(path)
    if not path.is_dir():
        raise SupportGapError(f"scores path does not exist: {path}")
    files = sorted([*path.glob("*.csv"), *path.glob("*.tsv"), *path.glob("*.parquet")])
    if not files:
        raise SupportGapError(f"no CSV, TSV, or Parquet score files found in {path}")
    return pd.concat([load_metadata(file) for file in files], ignore_index=True)


def merge_support_evidence(scores: pd.DataFrame, support_table: pd.DataFrame) -> pd.DataFrame:
    """Attach one support row per protein and reject ambiguous joins."""

    required_scores = {"DMS_id", "protein_id", "DMS_score", "evo_score"}
    missing = required_scores - set(scores.columns)
    if missing:
        raise SupportGapError(f"score table missing columns: {sorted(missing)}")
    required_support = {"protein_id", "evolutionary_support_score", "support_stratum", "sequence_length"}
    missing = required_support - set(support_table.columns)
    if missing:
        raise SupportGapError(f"support table missing columns: {sorted(missing)}")
    if support_table["protein_id"].duplicated().any():
        raise SupportGapError("support table has duplicate protein_id values")

    support_columns = [column for column in support_table.columns if column != "sequence"]
    merged = scores.merge(
        support_table[support_columns], on="protein_id", how="left", validate="many_to_one"
    )
    if merged["evolutionary_support_score"].isna().any():
        missing_ids = merged.loc[merged["evolutionary_support_score"].isna(), "protein_id"].unique()
        raise SupportGapError(
            f"{len(missing_ids)} proteins in scores have no support row; first: {missing_ids[0]!r}"
        )
    return merged


def summarize_assays(variant_scores: pd.DataFrame, min_variants: int = 20) -> pd.DataFrame:
    """Compute one zero-shot Spearman score per DMS assay."""

    if min_variants < 2:
        raise SupportGapError("min_variants must be at least 2")
    table = variant_scores.copy()
    if "mutant" in table and "mutation_depth" not in table:
        table["mutation_depth"] = table["mutant"].map(mutation_depth)

    summaries: list[dict[str, object]] = []
    for assay_id, group in table.groupby("DMS_id", sort=True):
        valid = group[["DMS_score", "evo_score"]].dropna()
        if len(valid) < min_variants:
            continue
        summary: dict[str, object] = {
            "DMS_id": assay_id,
            "protein_id": group["protein_id"].iloc[0],
            "n_variants": int(len(valid)),
            "spearman": spearman_correlation(valid["DMS_score"], valid["evo_score"]),
            "evolutionary_support_score": float(group["evolutionary_support_score"].iloc[0]),
            "support_stratum": int(group["support_stratum"].iloc[0]),
            "sequence_length": float(group["sequence_length"].iloc[0]),
        }
        if "mutation_depth" in group:
            summary["mean_mutation_depth"] = float(group["mutation_depth"].mean())
        for categorical in ("selection_type", "functional_category", "taxon"):
            if categorical in group:
                summary[categorical] = group[categorical].iloc[0]
        summaries.append(summary)
    if not summaries:
        raise SupportGapError("no assays have enough scored variants for analysis")
    return pd.DataFrame(summaries).sort_values("DMS_id").reset_index(drop=True)


def bootstrap_stratum_means(
    assay_summary: pd.DataFrame, samples: int = 5000, seed: int = 17
) -> pd.DataFrame:
    """Bootstrap assay-level means within each support stratum."""

    if samples < 100:
        raise SupportGapError("bootstrap samples must be at least 100")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for stratum, group in assay_summary.groupby("support_stratum", sort=True):
        values = group["spearman"].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        bootstrap = np.array(
            [rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)]
        )
        rows.append(
            {
                "support_stratum": int(stratum),
                "n_assays": int(len(values)),
                "mean_spearman": float(values.mean()),
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values("support_stratum").reset_index(drop=True)


def fit_support_regression(assay_summary: pd.DataFrame) -> dict[str, object]:
    """Fit transparent OLS with available pre-specified confounders.

    This is an effect-size diagnostic, not a causal estimator. Categorical covariates
    are one-hot encoded only when present in the supplied metadata.
    """

    numeric = ["evolutionary_support_score", "sequence_length", "mean_mutation_depth"]
    present_numeric = [column for column in numeric if column in assay_summary]
    categorical = [column for column in ("selection_type", "functional_category") if column in assay_summary]
    design = assay_summary[["spearman", *present_numeric, *categorical]].dropna().copy()
    if len(design) < len(present_numeric) + 3:
        return {"n_assays": int(len(design)), "warning": "insufficient complete assays for regression"}

    features = design[present_numeric].astype(float)
    if categorical:
        encoded = pd.get_dummies(design[categorical], prefix=categorical, drop_first=True, dtype=float)
        features = pd.concat([features, encoded], axis=1)
    matrix = np.column_stack([np.ones(len(features)), features.to_numpy(dtype=float)])
    response = design["spearman"].to_numpy(dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(matrix, response, rcond=None)
    fitted = matrix @ coefficients
    residual_sum = float(np.square(response - fitted).sum())
    total_sum = float(np.square(response - response.mean()).sum())
    names = ["intercept", *features.columns.tolist()]
    return {
        "n_assays": int(len(design)),
        "r_squared": None if total_sum == 0 else float(1 - residual_sum / total_sum),
        "coefficients": {name: float(value) for name, value in zip(names, coefficients, strict=True)},
    }


def run_support_gap_analysis(
    score_path: str | Path,
    support_path: str | Path,
    output_dir: str | Path,
    min_variants: int = 20,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 17,
    assay_metadata_path: str | Path | None = None,
) -> dict[str, object]:
    """Run and persist Phase 1 assay-level analyses."""

    scores = load_score_tables(score_path)
    if assay_metadata_path is not None:
        metadata = load_metadata(assay_metadata_path)
        if "DMS_id" not in metadata:
            raise SupportGapError("assay metadata needs DMS_id")
        scores = scores.merge(metadata, on="DMS_id", how="left", validate="many_to_one")
    variants = merge_support_evidence(scores, load_metadata(support_path))
    assays = summarize_assays(variants, min_variants=min_variants)
    strata = bootstrap_stratum_means(assays, samples=bootstrap_samples, seed=bootstrap_seed)
    regression = fit_support_regression(assays)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(assays, output_dir / "per_assay_spearman.csv")
    write_metadata(strata, output_dir / "support_stratum_summary.csv")
    stats = {
        "n_variant_rows": int(len(variants)),
        "n_assays": int(len(assays)),
        "support_gap": float(
            strata.sort_values("support_stratum").iloc[-1]["mean_spearman"]
            - strata.sort_values("support_stratum").iloc[0]["mean_spearman"]
        )
        if len(strata) > 1
        else None,
        "regression": regression,
    }
    (output_dir / "support_gap_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats

