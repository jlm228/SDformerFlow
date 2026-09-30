#!/bin/bash
# Measure where SDA's predicate stops, on a handful of windows.
#
#   sbatch hpc/sda_probe.slurm <capture_dir> <tau_factor> <snn|ann>
#   WINDOWS=8 FD_BATCH=2 sbatch hpc/sda_probe.slurm <capture_dir> 2.5 snn
#
# SDA's cost is quadratic in the round its predicate is met, so a full sweep cannot be sized
# until that round is known. This runs the real attack on a few windows and reports where it
# stopped, plus the wall time the projection needs.
#
# N is capped below the published 500 (see ITERS): a window still running at round 100 is
# already past any affordable sweep, so it is counted censored rather than run to 500.
#
# Run it at two FD_BATCH values. The candidate batch is the other unknown: the swin SNN peaks at
# ~23.4 GB a window, the worst case in the study, so the reference's 64 is far out of reach, and
# whether a batch costs one pass or one pass per candidate changes the projection by an order of
# magnitude. The SNN's BatchNorm runs in train mode, so FD_BATCH=1 is also the correctness-safe
# setting -- results are identical either way, only the throughput differs.
#
# Read the result with:
#   python -m attack_core.sda_rounds <report glob>

#SBATCH --job-name=sdf_sda_probe
#SBATCH --qos=bbgpu
#SBATCH --account=arnaboll-ai-research
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:0:0
#SBATCH --output=hpc/logs/%x_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
mkdir -p hpc/logs

USAGE="usage: sbatch hpc/sda_probe.slurm <capture_dir> <tau_factor> <snn|ann>"
CAPTURE="${1:?${USAGE}}"
TAU_FACTOR="${2:?${USAGE}}"
MODEL="${3:?${USAGE}}"

case "${MODEL}" in
  snn|ann) ;;
  *) echo "model must be snn or ann, got '${MODEL}'" >&2; exit 1 ;;
esac

WINDOWS="${WINDOWS:-5}"
# 1 by default because the SNN's BatchNorm runs in train mode: a batch larger than one changes
# the statistics each candidate sees, so it is only correct through a per-sample wrapper. Raise
# it for the ANN, or once that wrapper is confirmed present.
FD_BATCH="${FD_BATCH:-1}"
OBJECTIVE="${OBJECTIVE:-div}"
SIGN="${SIGN:-suppress}"
# 100, not the published 500. A window that has not stopped by round 100 already costs
# more than any affordable sweep, so spending 500 rounds to confirm it buys nothing --
# and at 500 one window alone can outlast the job, leaving no report to read. Windows
# that stop below 100 are measured exactly; the rest are counted as censored.
ITERS="${ITERS:-100}"
RUNID="${RUNID:-sda_probe}"

CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"
export CARLA_SCRIPTS_ROOT

CAPTURE_ID="carla_$(basename "${CAPTURE}")"
TENSORS="${CAPTURE}/saved_flow_data"
CLEAN="results/carla_eval/pred/${MODEL}"
OUT="results/sda_probe/${MODEL}"
BAND_JSON="${CAPTURE}/attack_band.json"

[ -d "${CLEAN}" ] || { echo "no clean predictions at ${CLEAN}" >&2; exit 1; }
[ -f "${BAND_JSON}" ] || {
  echo "no ${BAND_JSON} -- run attack_core.band first" >&2; exit 1; }
[ -d "${CARLA_SCRIPTS_ROOT}/attack_core" ] || {
  echo "no attack_core under ${CARLA_SCRIPTS_ROOT} -- set CARLA_SCRIPTS_ROOT" >&2; exit 1; }

# Build the voxel tensors if absent, as attack_carla.slurm does. Left in place: a second probe
# at a different FD_BATCH reuses them, and the sweep submitter removes them after the real run.
DONE_MARK="${TENSORS}/sequence_lists/${CAPTURE_ID}_split_seq.csv"
mkdir -p "$(dirname "${TENSORS}")"
(
  flock 9
  if [ -f "${DONE_MARK}" ]; then
    echo "tensors already complete at ${TENSORS} -- skipping conversion"
  else
    echo "converting (no ${DONE_MARK})"
    python carla_eval/carla_to_voxel.py "${CAPTURE}" --out "${TENSORS}" \
        --capture-id "${CAPTURE_ID}"
  fi
) 9>"${TENSORS}.lock"

[ -f "${DONE_MARK}" ] || { echo "conversion did not produce ${DONE_MARK}" >&2; exit 1; }

# Probe from the START of the band: the earliest windows are the hardest to make adversarial
# (the hazard is furthest away and smallest), so they bound the stopping round from above
# rather than flattering it.
BAND_LO=$(python -c "import json;print(json.load(open('${BAND_JSON}'))['band_lo'])")
BAND_HI=$((BAND_LO + WINDOWS - 1))

echo "=== SDA stopping-round probe ==="
echo "capture     ${CAPTURE} (id ${CAPTURE_ID})"
echo "model       ${MODEL}"
echo "windows     ${BAND_LO}..${BAND_HI} (${WINDOWS} of the band)"
echo "objective   ${OBJECTIVE} ${SIGN}"
echo "tau factor  ${TAU_FACTOR}"
echo "N           ${ITERS} (a window reaching this is counted censored)"
echo "fd batch    ${FD_BATCH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

START=$(date +%s)
python carla_eval/attack_carla.py \
    --model "${MODEL}" \
    --runid "${RUNID}" \
    --capture "${CAPTURE}" \
    --tensors "${TENSORS}" \
    --id "${CAPTURE_ID}" \
    --objective "${OBJECTIVE}" \
    --sign "${SIGN}" \
    --attack sda \
    --surrogate pdsg \
    --sda-tau-factors "${TAU_FACTOR}" \
    --sda-iters "${ITERS}" \
    --sda-fd-batch "${FD_BATCH}" \
    --band-lo "${BAND_LO}" \
    --band-hi "${BAND_HI}" \
    --clean-pred "${CLEAN}" \
    --out "${OUT}" \
    --report "${OUT}/reports"
END=$(date +%s)

ELAPSED=$((END - START))
echo
echo "wall time ${ELAPSED}s for ${WINDOWS} windows: $((ELAPSED / WINDOWS))s per window"
echo "at fd_batch ${FD_BATCH}. This is the throughput the projection needs, so record it."
echo
echo "=== stopping round ==="
# Absolute before the cd: attack_core is only importable from the scripts repo, but the
# reports are here.
REPORTS_ABS="$(cd "${OUT}/reports" && pwd)"
( cd "${CARLA_SCRIPTS_ROOT}" && python -m attack_core.sda_rounds \
    "${REPORTS_ABS}"/*.json --iters "${ITERS}" ) || {
  echo "could not summarise; the reports are under ${OUT}/reports" >&2; }
