#!/bin/bash
# Download the MVSEC HDF5 sequences used in the paper's Table 2.
#
#   bash hpc/download_mvsec.sh                 # all 4 evaluated sequences
#   bash hpc/download_mvsec.sh indoor_flying1  # one sequence
#
# RUN THIS ON A LOGIN NODE (compute nodes have no outbound network).
#
# UNLIKE hpc/download_dsec.sh, THIS SCRIPT IS UNVERIFIED. MVSEC's HDF5 files are distributed
# through a Google Drive folder rather than plain HTTP, so this drives `gdown`, whose folder
# listing breaks whenever Google changes the interstitial page. If it fails, fall back to
# downloading by hand from the folder below and dropping the files into RAW_DIR -- the layout
# is all this script really provides.
#
#   Folder: https://drive.google.com/drive/folders/1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV
#
# Each sequence needs BOTH files:
#   <seq>_data.hdf5   events + image timestamps
#   <seq>_gt.hdf5     ground-truth flow (flow_dist) -- MVSEC_encoder.py needs this
#
# Only these four sequences appear in Table 2. outdoor_day2 / indoor_flying4 are defined in
# the loader but never evaluated.

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

RAW_DIR="${RAW_DIR:-data/Datasets/MVSEC/raw}"
SEQUENCES=("$@")
[ ${#SEQUENCES[@]} -eq 0 ] && SEQUENCES=(outdoor_day1 indoor_flying1 indoor_flying2 indoor_flying3)

if ! command -v gdown >/dev/null; then
    echo "ERROR: gdown not found. Install it into the venv first:"
    echo "  source hpc/env.sh && pip install gdown"
    exit 1
fi

mkdir -p "${RAW_DIR}"

echo "Downloading ${#SEQUENCES[@]} MVSEC sequence(s) into ${RAW_DIR}"
echo

for seq in "${SEQUENCES[@]}"; do
    for suffix in data gt; do
        dst="${RAW_DIR}/${seq}/${seq}_${suffix}.hdf5"
        if [ -s "${dst}" ]; then
            echo "[${seq}] ${suffix}.hdf5 already present, skipping"
            continue
        fi
        mkdir -p "$(dirname "${dst}")"
        echo "[${seq}] fetching ${seq}_${suffix}.hdf5"
        # Resolve by name inside the shared folder. If Google rate-limits or changes the
        # interstitial, this is the step that fails -- see the header for the manual route.
        gdown --fuzzy --folder \
            "https://drive.google.com/drive/folders/1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV" \
            -O "${RAW_DIR}/_folder" --remaining-ok || {
            echo "[${seq}] gdown failed. Download ${seq}_${suffix}.hdf5 by hand into ${dst}"
            exit 1
        }
        found=$(find "${RAW_DIR}/_folder" -name "${seq}_${suffix}.hdf5" | head -1 || true)
        [ -n "${found}" ] || { echo "[${seq}] ${seq}_${suffix}.hdf5 not found in folder"; exit 1; }
        mv "${found}" "${dst}"
    done
    echo "[${seq}] OK"
done

echo
echo "Raw MVSEC in ${RAW_DIR}/<seq>/<seq>_{data,gt}.hdf5"
echo "Next:  sbatch --array=0-3 hpc/encode_mvsec.slurm"
