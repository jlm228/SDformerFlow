#!/bin/bash
# Submit a training run as a chain of Slurm segments, each resuming the previous one's rolling
# per-epoch checkpoint via --dependency=afterok.
#
#   bash hpc/submit_train.sh ann          # ANN, 1 segment   (~20-35 h, fits one 2-day job)
#   bash hpc/submit_train.sh snn          # SNN, 3 segments  (measured ~1h43m/epoch, ~4.3 days)
#   SEGMENTS=4 bash hpc/submit_train.sh snn
#   SMOKE=1   bash hpc/submit_train.sh ann      # 1 segment, 1 epoch, end-to-end check
#   WALLTIME=12:0:0 bash hpc/submit_train.sh ann
#
# To continue an existing run rather than start fresh:
#   PREV_RUNID=<mlflow_run_id> bash hpc/submit_train.sh snn
#
# Chaining exists because GPU walltime here is capped at 2 days (`sacctmgr show qos bbgpu` ->
# MaxWall 2-00:00:00). If a segment is killed by the walltime the next resumes from the last
# per-epoch checkpoint, losing at most one epoch. Over-requesting segments is harmless -- once
# training reaches n_epochs the extra segments exit almost immediately.

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"     # repo root
mkdir -p hpc/logs

MODEL="${1:-}"
case "${MODEL}" in
    ann) SCRIPT=hpc/train_ann.slurm; RUNID_FILE="${RUNID_FILE:-hpc/logs/ann_runid.txt}"; DEF_SEG=1 ;;
    # 3 segments: the smoke test measured ~1h43m/epoch (not the ~35-60min originally guessed),
    # so 60 epochs is ~103h -- 2 segments (96h capacity) is short; 3 (144h) has real margin for
    # per-segment startup and any epoch killed mid-flight by the walltime.
    snn) SCRIPT=hpc/train_snn.slurm; RUNID_FILE="${RUNID_FILE:-hpc/logs/snn_runid.txt}"; DEF_SEG=3 ;;
    *)   echo "usage: bash hpc/submit_train.sh {ann|snn}"; exit 1 ;;
esac

SEGMENTS="${SEGMENTS:-${DEF_SEG}}"
# 2 days is this QoS's hard cap (sacctmgr show qos bbgpu -> MaxWall 2-00:00:00). Shorter
# requests can backfill sooner if the queue is busy, at the cost of more segments.
WALLTIME="${WALLTIME:-2-0:0:0}"

if [ "${SMOKE:-0}" != "0" ]; then
    SEGMENTS=1
    echo "[SMOKE] single segment, 1 epoch"
fi

# A stale run id from an earlier chain would silently resume the wrong run, so make the choice
# explicit rather than guessing.
if [ -z "${PREV_RUNID:-}" ] && [ -f "${RUNID_FILE}" ]; then
    echo "ERROR: ${RUNID_FILE} already exists (run id: $(cat "${RUNID_FILE}"))."
    echo "Continue that run:  PREV_RUNID=$(cat "${RUNID_FILE}") bash hpc/submit_train.sh ${MODEL}"
    echo "Or start fresh:     rm ${RUNID_FILE} && bash hpc/submit_train.sh ${MODEL}"
    exit 1
fi

echo "model=${MODEL} | segments=${SEGMENTS} | walltime=${WALLTIME} | runid_file=${RUNID_FILE}"
echo

PREV_JOB=""
for i in $(seq 1 "${SEGMENTS}"); do
    if [ "${i}" -eq 1 ]; then
        # Segment 1 starts fresh, or from an explicit PREV_RUNID. CHAIN is deliberately unset so
        # it cannot pick up a leftover RUNID_FILE.
        JOB_ID=$(SMOKE="${SMOKE:-0}" PREV_RUNID="${PREV_RUNID:-}" RUNID_FILE="${RUNID_FILE}" \
            CONFIG="${CONFIG:-}" sbatch --parsable --time="${WALLTIME}" --export=ALL "${SCRIPT}")
        echo "segment ${i}/${SEGMENTS}: job ${JOB_ID}"
    else
        JOB_ID=$(SMOKE="${SMOKE:-0}" CHAIN=1 RUNID_FILE="${RUNID_FILE}" \
            CONFIG="${CONFIG:-}" sbatch --parsable --time="${WALLTIME}" \
            --dependency=afterok:"${PREV_JOB}" --export=ALL "${SCRIPT}")
        echo "segment ${i}/${SEGMENTS}: job ${JOB_ID}  (after ${PREV_JOB})"
    fi
    PREV_JOB="${JOB_ID}"
done

echo
echo "Submitted. Watch with:  bash hpc/watch.sh"
echo "Run id appears in ${RUNID_FILE} once segment 1 starts."
