#!/bin/bash
# Install the DSEC train/validation split lists into the location the dataloader reads.
#
# Run from anywhere -- it locates the repo from its own path:
#   bash hpc/setup_splits.sh
#
# Copies the vendored splits/*_doubleseq.csv (used by the ANN, num_chunks: 2) into
# data/Datasets/DSEC/saved_flow_data/sequence_lists/, then derives the single-chunk
# *_split_seq.csv the SNN needs (num_chunks: 1), which is published nowhere.
#
# Idempotent: re-running refreshes the doubleseq copies and regenerates the derived lists.

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"     # repo root

DEST="${DEST:-data/Datasets/DSEC/saved_flow_data/sequence_lists}"

[ -f splits/train_split_doubleseq.csv ] || {
    echo "ERROR: splits/train_split_doubleseq.csv not found -- is this a full checkout?"
    exit 1
}

mkdir -p "${DEST}"
cp splits/train_split_doubleseq.csv splits/valid_split_doubleseq.csv "${DEST}/"
echo "installed doubleseq splits -> ${DEST}"

# build_seq_split.py resolves --root relative to its own working directory, so give it an
# absolute path rather than relying on being run from DSEC_dataloader/.
python DSEC_dataloader/build_seq_split.py \
    --root "$(cd "$(dirname "${DEST}")" && pwd)" \
    --overwrite

echo
echo "Split lists in ${DEST}:"
for f in train_split_doubleseq train_split_seq valid_split_doubleseq valid_split_seq; do
    n=$(wc -l < "${DEST}/${f}.csv")
    printf '  %-28s %5d rows\n' "${f}.csv" "${n}"
done
echo
echo "Expected: 6000 train / 2152 valid, in both the doubleseq and seq forms."
