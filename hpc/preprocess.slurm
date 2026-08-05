#!/bin/bash
# Convert raw DSEC sequences into the 10-bin voxel .npy tensors the dataloader reads,
# plus the ground-truth and mask tensors.
#
# One array task per sequence: generate_files() handles sequences independently, and the
# sequences differ hugely in length (zurich_city_11_b alone dwarfs thun_00_a).
#
#   sbatch --array=0-17 hpc/preprocess.slurm     # all 18 flow sequences
#   sbatch --array=4    hpc/preprocess.slurm     # thun_00_a only (env smoke test)
#
# This is a pure-Python event-binning loop: single-threaded, and it never touches a GPU.
# Only the 18 sequences that have ground-truth flow are handled; the 7 official test
# sequences are deliberately excluded (no public GT, and we are not doing the submission).

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:0:0
#SBATCH --mem=32G
#SBATCH --job-name=sdf_prep
#SBATCH --output=hpc/logs/prep_%A_%a.out
#SBATCH --account=arnaboll-ai-research
#SBATCH --qos=bbdefault

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p hpc/logs
source hpc/env.sh

# Must match flow_sequences in DSEC_dataset_preprocess.py (order is irrelevant, membership
# is validated by the script itself, which exits non-zero on an unknown name).
SEQUENCES=(
    zurich_city_09_a zurich_city_07_a zurich_city_02_c zurich_city_11_b
    thun_00_a        zurich_city_02_d zurich_city_11_c zurich_city_03_a
    zurich_city_10_a zurich_city_05_b zurich_city_08_a zurich_city_01_a
    zurich_city_10_b zurich_city_02_e zurich_city_05_a zurich_city_06_a
    zurich_city_11_a zurich_city_02_a
)
SEQUENCE="${SEQUENCES[${SLURM_ARRAY_TASK_ID:-0}]}"

echo "Preprocessing ${SEQUENCE} on $(hostname)"

# Fail early and clearly rather than after however long the events take, since the FreeImage
# plugin is only needed once _create_flow_maps starts reading 16-bit PNGs.
python -c "import imageio; imageio.plugins.freeimage.download()" >/dev/null 2>&1 || {
    echo "ERROR: imageio FreeImage plugin unavailable and not downloadable from a compute node."
    echo "Populate the cache on the login node first:"
    echo "  source hpc/env.sh && python -c 'import imageio; imageio.plugins.freeimage.download()'"
    exit 1
}

# The script resolves both its sibling imports (event_representations) and its
# root='../data/Datasets/DSEC' relative to this directory, so it must be run from inside it.
cd "${SLURM_SUBMIT_DIR}/DSEC_dataloader"
python DSEC_dataset_preprocess.py "${SEQUENCE}"

echo "Finished ${SEQUENCE}"
