#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH -J swissprot_build
#SBATCH -o swissprot_%j.log
#SBATCH -e swissprot_%j.err

# =============================================================================
# Build Swiss-Prot reference database for vHold GO term transfer
#
# Usage:
#   # Viral scope only (~30 min)
#   sbatch scripts/slurm_build_swissprot.sh viral
#
#   # Full Swiss-Prot (~16 hrs, resubmit to resume)
#   sbatch scripts/slurm_build_swissprot.sh all
#
#   # Both scopes sequentially (viral first, then all)
#   sbatch scripts/slurm_build_swissprot.sh both
#
# The "all" scope exceeds the 8h default GPU limit. The script checkpoints
# every 1000 proteins, so resubmitting the same command resumes automatically.
# =============================================================================

set -euo pipefail

SCOPE="${1:-viral}"
OUTPUT_DIR="${HOME}/.vhold/databases/swissprot"

echo "============================================"
echo "vHold Swiss-Prot Reference Database Builder"
echo "============================================"
echo "Scope:      ${SCOPE}"
echo "Output:     ${OUTPUT_DIR}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started:    $(date)"
echo "============================================"

# Activate vHold environment — adjust path as needed
# Option 1: uv-managed venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Option 2: If using spack modules, uncomment and adjust:
# spack load --sh python cuda

build_scope() {
    local scope="$1"
    echo ""
    echo ">>> Building scope: ${scope}"
    echo ">>> Started: $(date)"

    python scripts/build_swissprot_reference.py \
        --scope "${scope}" \
        --output "${OUTPUT_DIR}" \
        --device cuda \
        --batch-size 32 \
        --checkpoint-interval 1000 \
        --resume

    echo ">>> Finished ${scope}: $(date)"
}

case "${SCOPE}" in
    viral)
        build_scope viral
        ;;
    all)
        build_scope all
        ;;
    both)
        build_scope viral
        build_scope all
        ;;
    *)
        echo "ERROR: Unknown scope '${SCOPE}'. Use: viral, all, or both"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Done: $(date)"
echo "Output: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}/"
echo "============================================"
