#!/bin/bash
# Submit clean-eval GPU jobs per capture, one each for snn and ann.
#
#   bash hpc/submit_clean_sweep.sh <capture_dir> [<capture_dir> ...]
#
# The eval job builds the capture's voxel tensors if they are missing, so raw captures are
# fine. Run ids from hpc/logs/{snn,ann}_runid.txt. Predictions land in
# results/carla_eval/pred/{snn,ann} as <capture_id>_<window>.npy.
# Captures with no events.npy, or already predicted, are skipped; RERUN=1 overrides.

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p hpc/logs

[ "$#" -ge 1 ] || { echo "usage: bash hpc/submit_clean_sweep.sh <capture_dir> [...]" >&2; exit 1; }

SNN_RUNID="${SNN_RUNID:-$(cat hpc/logs/snn_runid.txt 2>/dev/null || true)}"
ANN_RUNID="${ANN_RUNID:-$(cat hpc/logs/ann_runid.txt 2>/dev/null || true)}"
[ -n "${SNN_RUNID}" ] || { echo "ERROR: no SNN run id (hpc/logs/snn_runid.txt)" >&2; exit 1; }
[ -n "${ANN_RUNID}" ] || { echo "ERROR: no ANN run id (hpc/logs/ann_runid.txt)" >&2; exit 1; }

SUBMITTED=0
SKIPPED=0

for CAPTURE in "$@"; do
  SCEN="$(basename "${CAPTURE}")"
  if [ ! -f "${CAPTURE}/events.npy" ]; then
    echo "SKIP ${SCEN}: no events.npy"
    SKIPPED=$((SKIPPED + 2))
    continue
  fi
  # Matches the id carla_eval.slurm builds.
  CAPTURE_ID="carla_${SCEN}"

  for M in snn ann; do
    RUNID="${SNN_RUNID}"; [ "${M}" = "ann" ] && RUNID="${ANN_RUNID}"
    PRED="results/carla_eval/pred/${M}"
    if [ "${RERUN:-0}" = "0" ] && [ -n "$(find "${PRED}" -name "${CAPTURE_ID}_*.npy" \
                                          2>/dev/null | head -1)" ]; then
      echo "SKIP ${SCEN} ${M}: predictions already present"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
    JOB=$(sbatch --parsable hpc/carla_eval.slurm "${M}" "${CAPTURE}" "${RUNID}")
    echo "submitted ${SCEN} ${M}  (${CAPTURE_ID})  job ${JOB}"
    SUBMITTED=$((SUBMITTED + 1))
  done
done

echo
echo "${SUBMITTED} submitted, ${SKIPPED} skipped"
echo "Watch with: squeue --me"
