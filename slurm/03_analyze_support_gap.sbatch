#!/usr/bin/env bash
#SBATCH --job-name=evotail-gap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?Export PROJECT_DIR before sbatch}"
SUPPORT_OUTPUT="${SUPPORT_OUTPUT:?Export SUPPORT_OUTPUT before sbatch}"
PROTEINGYM_SCORES_DIR="${PROTEINGYM_SCORES_DIR:?Export PROTEINGYM_SCORES_DIR before sbatch}"
PHASE1_OUTPUT_DIR="${PHASE1_OUTPUT_DIR:?Export PHASE1_OUTPUT_DIR before sbatch}"

cd "$PROJECT_DIR"
mkdir -p slurm/logs "$PHASE1_OUTPUT_DIR"
# Replace with your site-specific module/conda activation.
# source /path/to/venv/bin/activate

python scripts/analyze_support_gap.py \
  --scores "$PROTEINGYM_SCORES_DIR" \
  --support-table "$SUPPORT_OUTPUT" \
  --output-dir "$PHASE1_OUTPUT_DIR" \
  --min-variants 20 \
  --bootstrap-samples 5000

