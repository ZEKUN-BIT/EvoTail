# Slurm submission

The provided scripts are templates. Edit account, partition, time, GPU type, module
loading, and virtual-environment activation for the local cluster. Submit Phase 1 in
this order:

```bash
sbatch slurm/01_compute_support.sbatch
sbatch --array=0-$((N_ASSAYS-1)) slurm/02_score_proteingym_array.sbatch
sbatch slurm/03_analyze_support_gap.sbatch
```

`02_score_proteingym_array.sbatch` assigns one ProteinGym assay to each array task.
This is deliberate: assays vary greatly in size and a failed assay can be rerun without
rescoring the complete benchmark.

Before spending GPU time, export the required paths and validate site-specific directives:

```bash
export PROJECT_DIR=/absolute/path/to/EvoTail
export SUPPORT_INPUT=/absolute/path/to/protein_metadata.csv
export SUPPORT_OUTPUT=/absolute/path/to/protein_support.parquet
export PROTEINGYM_MANIFEST=/absolute/path/to/proteingym_manifest.csv
export PROTEINGYM_SCORES_DIR=/scratch/$USER/evotail/proteingym_scores
export PHASE1_OUTPUT_DIR=/scratch/$USER/evotail/phase1
sbatch --test-only slurm/01_compute_support.sbatch
```
