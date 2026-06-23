"""Thin, auditable wrappers around external MMseqs2 clustering commands."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

import pandas as pd


def build_mmseqs_easy_cluster_command(
    input_fasta: str | Path,
    output_prefix: str | Path,
    temporary_dir: str | Path,
    min_seq_identity: float,
    coverage: float = 0.8,
) -> list[str]:
    """Build, but do not run, a deterministic MMseqs2 `easy-cluster` command."""

    if not 0 < min_seq_identity <= 1:
        raise ValueError("min_seq_identity must be in (0, 1]")
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    return [
        "mmseqs",
        "easy-cluster",
        str(input_fasta),
        str(output_prefix),
        str(temporary_dir),
        "--min-seq-id",
        str(min_seq_identity),
        "-c",
        str(coverage),
        "--cov-mode",
        "1",
    ]


def run_mmseqs_easy_cluster(command: Sequence[str]) -> None:
    """Run an already-reviewed MMseqs2 command and fail loudly on any error."""

    subprocess.run(list(command), check=True)


def cluster_sizes_from_tsv(path: str | Path) -> pd.DataFrame:
    """Return one row per member from an MMseqs2 representative/member TSV."""

    memberships = pd.read_csv(path, sep="\t", header=None, names=["representative", "protein_id"])
    if memberships.empty:
        raise ValueError("MMseqs2 cluster TSV is empty")
    sizes = memberships.groupby("representative")["protein_id"].transform("size")
    return memberships.assign(cluster_size=sizes)[["protein_id", "representative", "cluster_size"]]

