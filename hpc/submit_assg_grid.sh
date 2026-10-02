#!/bin/bash
# Calibrate ASSG's A for the SDformerFlow SNN: one SA-PGD run per grid value, then pick.
#
#   EPS=<epsilon> CAPTURE_IDS="001 004 028" bash hpc/submit_assg_grid.sh <captures_dir>
#   EPS=0.01 LIMIT=1 DRY_RUN=1 bash hpc/submit_assg_grid.sh <captures_dir>    # see what it would do
#
# The grid is the paper's appendix S3.2: A over [0.82, 0.90] in steps of 0.01, nine values, on
# the Atan base this model uses. Each value is scored on one objective (div_suppress) at one
# epsilon, the middle of the ramp, by mean Delta J over the band windows; see
# attack_core/assg_grid.py. Then A is frozen for every later sweep.
#
# Pick A once every job has finished:
#   cd ../CARLA-hpc-scripts && python -m attack_core.assg_grid \
#       ../SDformerFlow/results/assg_grid/*/A*/snn/reports/*/*.json \
#       --out ../SDformerFlow/results/assg_grid/assg_A_snn.json

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p hpc/logs

USAGE="usage: EPS=<epsilon> bash hpc/submit_assg_grid.sh <captures_dir>"
CAPTURES="${1:?${USAGE}}"
# No default: the epsilon is the middle of the calibrated ramp, which this script cannot know.
EPS="${EPS:?set EPS to the middle of the calibrated epsilon ramp}"
GRID="${GRID:-0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.90}"
ITERS="${ITERS:-100}"
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
GRID_ROOT="${GRID_ROOT:-results/assg_grid}"
MODEL=snn          # the ANN has no spiking neurons, so there is no A to tune

RUNID="${SNN_RUNID:-$(cat hpc/logs/snn_runid.txt 2>/dev/null || true)}"
[ -n "${RUNID}" ] || { echo "ERROR: no SNN run id (hpc/logs/snn_runid.txt)" >&2; exit 1; }
[ -d "results/carla_eval/pred/${MODEL}" ] || {
  echo "ERROR: no clean predictions at results/carla_eval/pred/${MODEL}" >&2; exit 1; }

if [ -n "${CAPTURE_IDS:-}" ]; then
  DIRS=()
  for ID in ${CAPTURE_IDS}; do
    [ -d "${CAPTURES}/${ID}" ] || { echo "no capture ${CAPTURES}/${ID}" >&2; exit 1; }
    DIRS+=("${CAPTURES}/${ID}")
  done
else
  mapfile -t DIRS < <(ls -d "${CAPTURES}"/[0-9][0-9][0-9] 2>/dev/null)
fi
[ "${#DIRS[@]}" -gt 0 ] || { echo "no captures under ${CAPTURES}" >&2; exit 1; }
[ "${LIMIT}" -gt 0 ] && DIRS=("${DIRS[@]:0:${LIMIT}}")

N_A=$(echo ${GRID} | wc -w)
echo "captures   ${#DIRS[@]} under ${CAPTURES}"
echo "grid       A = ${GRID} (${N_A} values)"
echo "run        ${MODEL} | div/suppress | sapgd-assg | eps ${EPS} | ${ITERS} iters"
echo "jobs       up to $(( ${#DIRS[@]} * N_A )) (captures without a band are skipped), under ${GRID_ROOT}"
echo

SB_TIME=""
[ -n "${TIME:-}" ] && SB_TIME="--time=${TIME}"

SUBMITTED=0
SKIPPED=""
for CAP in "${DIRS[@]}"; do
  ID="$(basename "${CAP}")"
  if [ ! -f "${CAP}/attack_band.json" ]; then
    SKIPPED="${SKIPPED} ${ID}"
    continue
  fi
  MANIFEST="hpc/logs/assg_grid_${ID}.txt"
  if [ "${DRY_RUN}" != "0" ]; then
    echo "would submit ${ID}: ${N_A} jobs"
    SUBMITTED=$((SUBMITTED + 1))
    continue
  fi
  echo "${MODEL} ${RUNID} div          suppress  sapgd-assg ${ITERS} ${EPS}" > "${MANIFEST}"

  JOBS=""
  for A in ${GRID}; do
    OUT_ROOT="${GRID_ROOT}/${ID}/A${A}"
    J=$(sbatch --parsable --array=1-1 ${SB_TIME} --job-name="sdf_assg_${ID}_${A}" \
        --export=ALL,ASSG_A="${A}",OUT_ROOT="${OUT_ROOT}",RAND_INIT=1,NORM_SET="${NORM_SET:-}" \
        hpc/attack_carla.slurm "${CAP}" "${MANIFEST}")
    echo "  ${ID} A=${A}: job ${J} -> ${OUT_ROOT}"
    JOBS="${JOBS}:${J}"
  done

  # The nine jobs share one capture's converted tensors, so they are removed only after all of
  # them, as submit_attack_sweep.sh does. KEEP_TENSORS=1 leaves them for the sweep that follows.
  if [ "${KEEP_TENSORS:-0}" != "1" ]; then
    sbatch --parsable --dependency=afterany"${JOBS}" \
        --job-name=sdf_tensors_rm --time=00:15:00 --mem=2G --ntasks=1 \
        --output=hpc/logs/%x_%j.out \
        --wrap="rm -rf '${CAP}/saved_flow_data' '${CAP}/saved_flow_data.lock'" >/dev/null
  fi
  SUBMITTED=$((SUBMITTED + 1))
done

echo
echo "submitted ${SUBMITTED} of ${#DIRS[@]} captures"
[ -n "${SKIPPED}" ] && echo "skipped (no attack_band.json):${SKIPPED}"
echo "when every job has finished, choose A with:"
echo "  cd \${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts} && python -m attack_core.assg_grid \\"
echo "      $(pwd)/${GRID_ROOT}/*/A*/snn/reports/*/*.json --out $(pwd)/${GRID_ROOT}/assg_A_snn.json"
exit 0
