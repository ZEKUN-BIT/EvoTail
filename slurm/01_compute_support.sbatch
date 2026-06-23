#!/usr/bin/env bash
# Edit cluster-specific resource directives before submitting.
#SBATCH --job-name=evotail-support
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?Export PROJECT_DIR before sbatch}"
SUPPORT_INPUT="${SUPPORT_INPUT:?Export SUPPORT_INPUT before sbatch}"
SUPPORT_OUTPUT="${SUPPORT_OUTPUT:?Export SUPPORT_OUTPUT before sbatch}"

cd "$PROJECT_DIR"
mkdir -p slurm/logs "$(dirname "$SUPPORT_OUTPUT")"
# Replace with your site-specific module/conda activation.
# module load cuda/12.1
# source /path/to/venv/bin/activate

python scripts/build_support_table.py \
  --input "$SUPPORT_INPUT" \
  --output "$SUPPORT_OUTPUT" \
  --strata 4 \
  --alpha 1.0 \
  --minimum-weight 0.25 \
  --maximum-weight 4.0

