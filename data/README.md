# Data contract

Raw data is intentionally not stored in this repository. `configs/data.yaml` is a
record of placeholder paths; the current Python scripts take paths as command-line
arguments, while the Slurm templates take them from exported environment variables.

## Support-table input

`scripts/build_support_table.py` accepts CSV, TSV, or Parquet. Required columns:

| Column | Meaning |
|---|---|
| `protein_id` | Stable protein identifier |
| `sequence` | Uppercase amino-acid sequence |

Optional support-evidence columns are `nearest_identity` (0 to 1),
`cluster_size_90`, `cluster_size_50`, `cluster_size_30`, `msa_neff`, and
`taxon_count`. A numeric zero in `msa_neff` means no detected effective homologous
support; an empty value means unknown and is median-imputed with a recorded flag.

## ProteinGym array manifest

Create a CSV or TSV with these columns:

| Column | Meaning |
|---|---|
| `DMS_id` | Unique ProteinGym assay identifier |
| `assay_path` | Absolute path to that assay's substitutions CSV |
| `target_seq` | Wild-type amino-acid sequence |
| `protein_id` | Identifier used by the support table |

Each assay file must include `mutant` and `DMS_score`. `mutated_sequence` is
optional: when absent, the scorer derives it from `target_seq` and `mutant`. The
scoring script preserves all input columns and adds `evo_score`.

## Leakage audit

For every continual-pretraining corpus, save the exact release date and the
MMseqs2 search results from each evaluation target to the corpus. Phase 1 evaluates
an existing frozen checkpoint, so it measures an association rather than a strict
unseen-protein claim.
