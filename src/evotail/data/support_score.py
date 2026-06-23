"""Compute auditable evolutionary-support scores from protein metadata.

The score is deliberately an observation-support score, not a claim that a protein is
an orphan or de novo. Missing evidence is median-imputed and flagged; an observed
zero effective MSA depth remains valid evidence of low support.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("protein_id", "sequence")
CLUSTER_COLUMNS = ("cluster_size_90", "cluster_size_50", "cluster_size_30")
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_AMINO_ACIDS = frozenset("XBZUO")


class SupportScoreError(ValueError):
    """Raised when support-score inputs violate the published data contract."""


@dataclass(frozen=True)
class SupportScoreConfig:
    """Parameters fixed before inspecting downstream performance."""

    strata: int = 4
    alpha: float = 1.0
    minimum_weight: float = 0.25
    maximum_weight: float = 4.0
    allow_ambiguous_amino_acids: bool = False

    def __post_init__(self) -> None:
        if self.strata < 2:
            raise SupportScoreError("strata must be at least 2")
        if self.alpha <= 0:
            raise SupportScoreError("alpha must be positive")
        if self.minimum_weight <= 0 or self.maximum_weight < self.minimum_weight:
            raise SupportScoreError("invalid sampling-weight bounds")


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load CSV, TSV, or Parquet metadata without inferring a scientific schema."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise SupportScoreError(f"Unsupported input extension: {path.suffix}")


def write_metadata(table: pd.DataFrame, path: str | Path) -> None:
    """Write a table based on its explicit file suffix."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        table.to_csv(path, index=False)
    elif suffix in {".tsv", ".txt"}:
        table.to_csv(path, sep="\t", index=False)
    elif suffix in {".parquet", ".pq"}:
        table.to_parquet(path, index=False)
    else:
        raise SupportScoreError(f"Unsupported output extension: {path.suffix}")


def _validate_sequences(
    sequences: pd.Series, allow_ambiguous_amino_acids: bool
) -> pd.Series:
    if sequences.isna().any():
        raise SupportScoreError("sequence contains missing values")
    normalized = sequences.astype(str).str.strip().str.upper()
    allowed = VALID_AMINO_ACIDS | (
        AMBIGUOUS_AMINO_ACIDS if allow_ambiguous_amino_acids else frozenset()
    )
    invalid = normalized.map(lambda seq: not seq or bool(set(seq) - allowed))
    if invalid.any():
        example = normalized.loc[invalid].iloc[0]
        raise SupportScoreError(
            "sequence contains an empty value or unsupported amino acids; "
            f"first invalid sequence: {example!r}"
        )
    return normalized


def _nonnegative_numeric(table: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(table[column], errors="coerce")
    if (values.dropna() < 0).any():
        raise SupportScoreError(f"{column} must be non-negative")
    return values


def _percentile_with_missing(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return rank-percentiles and a boolean missing flag.

    Median imputation is intentionally conservative: an unknown value must not create
    artificial tail membership. The flag remains in the output for later sensitivity
    analyses.
    """

    missing = values.isna()
    if missing.all():
        raise SupportScoreError("a support component is entirely missing")
    filled = values.fillna(values.median())
    if filled.nunique(dropna=True) == 1:
        percentile = pd.Series(0.5, index=filled.index, dtype=float)
    else:
        percentile = filled.rank(method="average", pct=True).astype(float)
    return percentile, missing


def _mean_available(columns: Iterable[pd.Series]) -> pd.Series:
    frame = pd.concat(list(columns), axis=1)
    return frame.mean(axis=1, skipna=True)


def compute_support_table(
    metadata: pd.DataFrame, config: SupportScoreConfig | None = None
) -> pd.DataFrame:
    """Add support evidence, support strata, and bounded tail-priority weights.

    Higher `evolutionary_support_score` means denser observed evolutionary evidence.
    Higher `tail_sampling_weight` means the sequence receives more sampling budget in
    later experiments. This function does not execute sampling or assert causality.
    """

    config = config or SupportScoreConfig()
    missing_required = [column for column in REQUIRED_COLUMNS if column not in metadata]
    if missing_required:
        raise SupportScoreError(f"missing required columns: {missing_required}")
    if metadata["protein_id"].isna().any() or metadata["protein_id"].duplicated().any():
        raise SupportScoreError("protein_id must be present and unique")

    table = metadata.copy()
    table["sequence"] = _validate_sequences(
        table["sequence"], config.allow_ambiguous_amino_acids
    )
    table["sequence_length"] = table["sequence"].str.len()
    table["max_residue_fraction"] = table["sequence"].map(
        lambda seq: max(seq.count(residue) for residue in set(seq)) / len(seq)
    )

    component_percentiles: dict[str, pd.Series] = {}
    missing_flags: dict[str, pd.Series] = {}

    if "nearest_identity" in table:
        identity = pd.to_numeric(table["nearest_identity"], errors="coerce")
        if ((identity.dropna() < 0) | (identity.dropna() > 1)).any():
            raise SupportScoreError("nearest_identity must be in [0, 1]")
        component_percentiles["identity"] , missing_flags["identity"] = _percentile_with_missing(identity)
        table["support_nearest_identity"] = identity

    available_clusters = [column for column in CLUSTER_COLUMNS if column in table]
    if available_clusters:
        transformed_clusters = []
        for column in available_clusters:
            values = _nonnegative_numeric(table, column)
            if (values.dropna() == 0).any():
                raise SupportScoreError(f"{column} must be at least one when observed")
            table[f"support_{column}"] = values
            transformed_clusters.append(np.log1p(values))
        cluster_evidence = _mean_available(transformed_clusters)
        component_percentiles["cluster"] , missing_flags["cluster"] = _percentile_with_missing(cluster_evidence)
        table["support_log_cluster_size"] = cluster_evidence

    if "msa_neff" in table:
        msa_neff = _nonnegative_numeric(table, "msa_neff")
        table["support_msa_neff"] = msa_neff
        component_percentiles["msa"] , missing_flags["msa"] = _percentile_with_missing(np.log1p(msa_neff))

    if "taxon_count" in table:
        taxon_count = _nonnegative_numeric(table, "taxon_count")
        if (taxon_count.dropna() == 0).any():
            raise SupportScoreError("taxon_count must be at least one when observed")
        table["support_taxon_count"] = taxon_count
        component_percentiles["taxon"] , missing_flags["taxon"] = _percentile_with_missing(
            np.log1p(taxon_count)
        )

    if not component_percentiles:
        raise SupportScoreError(
            "provide at least one support-evidence column: nearest_identity, a cluster "
            "size, msa_neff, or taxon_count"
        )

    component_frame = pd.DataFrame(component_percentiles, index=table.index)
    table["support_components_used"] = ",".join(component_frame.columns)
    table["support_missing_component_count"] = pd.DataFrame(missing_flags).sum(axis=1)
    table["evolutionary_support_score"] = component_frame.mean(axis=1)

    score = table["evolutionary_support_score"]
    if score.nunique() == 1:
        strata = pd.Series(0, index=table.index, dtype=int)
    else:
        stable_rank = score.rank(method="first")
        strata = pd.qcut(stable_rank, q=min(config.strata, len(table)), labels=False).astype(int)
    table["support_stratum"] = strata

    raw_weight = np.power(1.0 - score + 1.0 / config.strata, config.alpha)
    clipped_weight = raw_weight.clip(config.minimum_weight, config.maximum_weight)
    table["tail_sampling_weight"] = clipped_weight / clipped_weight.mean()
    return table

