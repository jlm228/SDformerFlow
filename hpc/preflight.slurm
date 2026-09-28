#!/bin/bash
# Pre-flight measurement on one window: ATTACK_PLAN.md 2.3 step 1.
#
#   sbatch hpc/preflight.slurm <snn|ann> <runid> <capture_dir> <capture_id> [window]
#
# Reports, for this model:
#   * swap coverage -- which spiking modules a surrogate swap reaches. The ANN has none, and
#     says so: it is the matched control, so only its timings matter;
#   * mean and max |u| per spiking layer, which set the omega search bracket for the A
#     calibration (step 5);
#   * seconds per forward and per forward+backward, which set EPS_CHUNK and, with SDA's
#     candidate counts, its largest feasible rho;
#   * peak GPU memory, which decides the finite-difference batch size -- and on this model
#     whether the per-sample BatchNorm wrapper is needed at all (4.4 item 5).

#SBATCH --job-name=sdf_preflight
#SBATCH --qos=bbgpu
#SBATCH --account=arnaboll-ai-research
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:0
#SBATCH --output=hpc/logs/%x_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
mkdir -p hpc/logs

USAGE="usage: sbatch hpc/preflight.slurm <snn|ann> <runid> <capture_dir> <capture_id> [window]"
MODEL="${1:?${USAGE}}"
RUNID="${2:?${USAGE}}"
CAPTURE="${3:?${USAGE}}"
CAPTURE_ID="${4:?${USAGE}}"
WINDOW="${5:-}"

case "${MODEL}" in
  snn|ann) ;;
  *) echo "${USAGE}" >&2; exit 1 ;;
esac

CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"
export CARLA_SCRIPTS_ROOT

[ -d "${CARLA_SCRIPTS_ROOT}/attack_core" ] || {
  echo "no attack_core under ${CARLA_SCRIPTS_ROOT} -- set CARLA_SCRIPTS_ROOT" >&2; exit 1; }
[ -f "${CAPTURE}/events.npy" ] || {
  echo "no ${CAPTURE}/events.npy -- the pre-flight encodes its window from the raw events" >&2
  exit 1; }
[ -f "${CAPTURE}/windows.csv" ] || {
  echo "no ${CAPTURE}/windows.csv -- needed for the window's start timestamp" >&2; exit 1; }

WINDOW_ARG=""
[ -n "${WINDOW}" ] && WINDOW_ARG="--preflight-window ${WINDOW}"

echo "=== pre-flight: ${MODEL} on ${CAPTURE} (id ${CAPTURE_ID}) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --tensors is required by the parser but unused under --preflight, which reads events.npy.
# --surrogate stays native: the measurement swaps in its own probe whichever surrogate is
# configured, so there is no A to supply and nothing to tune yet.
# shellcheck disable=SC2086
python carla_eval/attack_carla.py --preflight ${WINDOW_ARG} \
    --model "${MODEL}" \
    --runid "${RUNID}" \
    --tensors "results/carla_eval/saved_flow_data" \
    --capture "${CAPTURE}" \
    --id "${CAPTURE_ID}" \
    --objective div \
    --epsilons 0

echo
echo "Next: put mean |u| into the step-5 omega bracket, and the forward+backward time into"
echo "EPS_CHUNK (one SA-PGD task is ~101 passes per epsilon against PGD's 10)."
