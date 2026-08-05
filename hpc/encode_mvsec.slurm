#!/bin/bash
# Encode raw MVSEC HDF5 into the per-interval event chunks + ground-truth flow the loader reads.
#
#   sbatch --array=0-3 hpc/encode_mvsec.slurm          # all 4 Table 2 sequences, dt=1
#   DT=4 sbatch --array=0-3 hpc/encode_mvsec.slurm     # dt=4
#   sbatch --array=1 hpc/encode_mvsec.slurm            # indoor_flying1 only
#
# CPU only. Writes data/Datasets/MVSEC/MVSEC_test/<seq>/{event/*.h5, flowgt_dt<N>/*.npy}.

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:0:0
#SBATCH --mem=32G
#SBATCH --job-name=sdf_mvsec_enc
#SBATCH --output=hpc/logs/mvsec_%A_%a.out
#SBATCH --account=arnaboll-ai-research
#SBATCH --qos=bbdefault

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p hpc/logs
source hpc/env.sh

SEQUENCES=(outdoor_day1 indoor_flying1 indoor_flying2 indoor_flying3)
SEQUENCE="${SEQUENCES[${SLURM_ARRAY_TASK_ID:-0}]}"
DT="${DT:-1}"

RAW_DIR="${RAW_DIR:-data/Datasets/MVSEC/raw}"

for suffix in data gt; do
    f="${RAW_DIR}/${SEQUENCE}/${SEQUENCE}_${suffix}.hdf5"
    [ -s "${f}" ] || { echo "ERROR: ${f} missing -- run hpc/download_mvsec.sh first."; exit 1; }
done

echo "Encoding ${SEQUENCE} at dt=${DT} on $(hostname)"

# MVSEC_encoder.py handles its two path arguments ASYMMETRICALLY, which its own defaults get
# wrong (they point at the author's /home/ytian/...):
#   --save-dir is used verbatim, so it must be ABSOLUTE and contain <seq>/<seq>_data.hdf5
#   --out-dir  is passed through osp.join(proc_path, ...), and os.path.join discards
#              everything before an absolute component -- so it must be RELATIVE.
python MDR_dataloader/MVSEC_encoder.py \
    -se "${SEQUENCE}" \
    -dt "${DT}" \
    -sd "$(cd "${RAW_DIR}" && pwd)" \
    -od "data/Datasets/MVSEC/MVSEC_test"

echo "Finished ${SEQUENCE} dt=${DT}"
