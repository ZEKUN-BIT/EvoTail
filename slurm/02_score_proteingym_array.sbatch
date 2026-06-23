#!/usr/bin/env bash
# Submit with: sbatch --array=0-$((N_ASSAYS-1)) slurm/02_score_proteingym_array.sbatch
#SBATCH --job-name=evotail-pgym
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?Export PROJECT_DIR before sbatch}"
PROTEINGYM_MANIFEST="${PROTEINGYM_MANIFEST:?Export PROTEINGYM_MANIFEST before sbatch}"
PROTEINGYM_SCORES_DIR="${PROTEINGYM_SCORES_DIR:?Export PROTEINGYM_SCORES_DIR before sbatch}"
MODEL_NAME="${MODEL_NAME:-facebook/esm2_t30_150M_UR50D}"
HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"

cd "$PROJECT_DIR"
mkdir -p slurm/logs "$PROTEINGYM_SCORES_DIR" "$HF_HOME"
# Replace with your site-specific module/conda activation.
# module load cuda/12.1
# source /path/to/venv/bin/activate

python scripts/score_proteingym_assay.py \
  --manifest "$PROTEINGYM_MANIFEST" \
  --array-index "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$PROTEINGYM_SCORES_DIR" \
  --model "$MODEL_NAME" \
  --cache-dir "$HF_HOME" \
  --device cuda \
  --batch-size 32

