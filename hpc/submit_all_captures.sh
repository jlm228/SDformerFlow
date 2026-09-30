#!/bin/bash
# The whole sweep over every capture, from one command.
#
#   bash hpc/submit_all_captures.sh <captures_dir>
#   ATTACK=sda-pdsg SDA_MODE=target CAPTURE_IDS="001 004 028" \
#       bash hpc/submit_all_captures.sh ../CARLA-hpc-scripts/captures/135_scenarios
#   LIMIT=5 DRY_RUN=1 bash hpc/submit_all_captures.sh <captures_dir>    # see what it would do
#
# submit_attack_sweep.sh handles ONE capture and CLEARS its output dirs on the way in, so calling
# it in a loop directly would leave only the last capture's cells. This clears once, then passes
# KEEP_PREVIOUS=1 to every call.
#
# ONE SWEEP COVERS BOTH MODELS: SDformerFlow (snn) and STTFlowNet-en4 (ann) share this repo's venv
# and its voxel tensors, so the per-capture submission already carries both and only the clean
# checks and the clear need to name them.
#
# SDA needs a target per capture AND per model: the tau factor at which that model's outcome
# tier changes, read from tau_probe_<model>_<sign>.json. A capture the probe never moved for
# some model is SKIPPED. Set SDA_FALLBACK to attack those at a fixed factor instead.

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p hpc/logs

USAGE="usage: bash hpc/submit_all_captures.sh <captures_dir>"
CAPTURES="${1:?${USAGE}}"
CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"

MODELS="${MODELS:-snn ann}"
ATTACK="${ATTACK:-sapgd-assg}"
OPTIMISER="${ATTACK%%-*}"
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
STAGGER="${STAGGER:-2}"

for MODEL in ${MODELS}; do
  [ -d "results/carla_eval/pred/${MODEL}" ] || {
    echo "ERROR: no clean predictions at results/carla_eval/pred/${MODEL};" >&2
    echo "       run hpc/carla_eval.slurm first" >&2; exit 1; }
done

# CAPTURE_IDS picks an explicit set, in the order given, so a run can cover a chosen design
# rather than the first N on disk. Without it, every capture in the directory.
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

# The probe writes tau_probe_<model>_<sign>.json, naming this repo's snn/ann
# sdformerflow/sttflownet_en4.
probe_model() {
  case "$1" in
    snn) echo "sdformerflow" ;;
    ann) echo "sttflownet_en4" ;;
    *)   echo "$1" ;;
  esac
}

TAU_DIR=""
TAU_SIGN="${TAU_SIGN:-suppress}"
if [ "${OPTIMISER}" = "sda" ]; then
  TAU_DIR="${TAU_DIR_IN:-${CARLA_SCRIPTS_ROOT}/results/tau_probe}"
  [ -d "${TAU_DIR}" ] || {
    echo "no tau probe directory at ${TAU_DIR}; set TAU_DIR_IN" >&2; exit 1; }
fi

echo "captures   ${CAPTURES} (${#DIRS[@]} to submit)"
echo "models     ${MODELS}"
echo "attack     ${ATTACK}"
[ "${OPTIMISER}" = "sda" ] && echo "targets    ${TAU_DIR} (sign ${TAU_SIGN})"
echo

# Clear ONCE, before the loop, for the same reason submit_attack_sweep.sh clears: a previous
# sweep with a different ramp leaves cells behind that everything downstream would score too.
if [ "${KEEP_PREVIOUS:-0}" = "0" ] && [ "${DRY_RUN}" = "0" ]; then
  for MODEL in ${MODELS}; do
    echo "clearing results/attack/${MODEL} (KEEP_PREVIOUS=1 to append)"
    rm -rf "results/attack/${MODEL}"
    mkdir -p "results/attack/${MODEL}"
  done
fi

SUBMITTED=0
SKIPPED=""
LOG="hpc/logs/submit_all_$(date +%Y%m%d_%H%M%S).txt"

for CAP in "${DIRS[@]}"; do
  ID="$(basename "${CAP}")"

  if [ ! -f "${CAP}/attack_band.json" ]; then
    SKIPPED="${SKIPPED} ${ID}:no-band"
    continue
  fi

  LEVELS=""
  if [ "${OPTIMISER}" = "sda" ]; then
    # One tau factor per model: each is the misreading that moves THAT model's outcome
    # tier. A capture with no tier change for some model is skipped rather than attacked
    # against a threshold that model's planner does not respond to.
    LEVELS="ok"
    for MM in ${MODELS}; do
      REC="${TAU_DIR}/tau_probe_$(probe_model "${MM}")_${TAU_SIGN}.json"
      [ -f "${REC}" ] || { echo "no tau probe record ${REC}" >&2; exit 1; }
      # Absolute, because the lookup runs after a cd into the scripts repo.
      REC_ABS="$(cd "$(dirname "${REC}")" && pwd)/$(basename "${REC}")"
      # 2>&1, so a failed lookup reports WHY instead of vanishing into the skip list.
      if ! V=$( cd "${CARLA_SCRIPTS_ROOT}" && python -m attack_core.tau_levels \
                  "${REC_ABS}" "${ID}" \
                  ${SDA_FALLBACK:+--fallback "${SDA_FALLBACK}"} 2>&1 ); then
        echo "  ${ID} ${MM}: ${V}" >&2
        SKIPPED="${SKIPPED} ${ID}:no-zeta-${MM}"
        LEVELS=""
        break
      fi
      UP=$(echo "${MM}" | tr a-z A-Z)
      eval "export SDA_LEVELS_${UP}=\"${V}\""
      LEVELS="${LEVELS} ${MM}=${V}"
    done
    [ -n "${LEVELS}" ] || continue
    LEVELS="${LEVELS#ok }"
  fi

  if [ "${DRY_RUN}" != "0" ]; then
    echo "would submit ${ID}${LEVELS:+ at tau ${LEVELS}}"
    SUBMITTED=$((SUBMITTED + 1))
    continue
  fi

  echo "=== ${ID}${LEVELS:+ (tau ${LEVELS})} ===" | tee -a "${LOG}"
  KEEP_PREVIOUS=1 ATTACK="${ATTACK}" \
    bash hpc/submit_attack_sweep.sh "${CAP}" 2>&1 | tee -a "${LOG}"
  SUBMITTED=$((SUBMITTED + 1))

  # The scheduler accepts these faster than it can process them; a short pause keeps the
  # submission log readable and avoids tripping a rate limit partway through.
  sleep "${STAGGER}"
done

echo
echo "submitted ${SUBMITTED} of ${#DIRS[@]} captures"
[ -n "${SKIPPED}" ] && {
  echo "skipped:${SKIPPED}"
  echo "  no-band: run 'python -m attack_core.band --capture <dir>' in ${CARLA_SCRIPTS_ROOT}"
  echo "  no-zeta: the tau probe never moved this capture's outcome tier. Raise the probe's"
  echo "           ceiling, or set SDA_FALLBACK to attack it at a fixed factor anyway"; }
[ "${DRY_RUN}" = "0" ] && echo "log: ${LOG}"
echo "watch with: squeue --me"
exit 0
