#!/bin/bash
# Spike and operation counts for the SNN and the ANN over one CARLA capture.
#
#   sbatch hpc/spike_metrics.slurm <capture_dir> [windows]   # default 4 windows each
#
# Both models in one job: the SNN's counts are read against the ANN's, and they share the
# tensors. Arguments are positional because this cluster drops environment variables.
# Tensors are rebuilt below if carla_eval.slurm has already removed them.

#SBATCH --job-name=sdf_spikes
#SBATCH --qos=bbgpu
#SBATCH --account=arnaboll-ai-research
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:0:0
#SBATCH --output=hpc/logs/%x_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p hpc/logs
source hpc/env.sh

USAGE="usage: sbatch hpc/spike_metrics.slurm <capture_dir> [windows]"
CAPTURE="${1:?${USAGE}}"
WINDOWS="${2:-4}"

CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"
export CARLA_SCRIPTS_ROOT
[ -f "${CARLA_SCRIPTS_ROOT}/snnmetrics/probe.py" ] || {
  echo "ERROR: ${CARLA_SCRIPTS_ROOT}/snnmetrics/probe.py not found."
  echo "       Set CARLA_SCRIPTS_ROOT to your CARLA-hpc-scripts checkout."; exit 1; }
[ -d "${CAPTURE}" ] || { echo "ERROR: capture dir ${CAPTURE} not found."; exit 1; }

CAPTURE_ID="carla_$(basename "${CAPTURE}")"
TENSORS="${CAPTURE}/saved_flow_data"
OUT="results/spike_metrics"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print('GPU:', torch.cuda.get_device_name(0))"
echo

# The metric maths is environment-independent, so its tests run here too.
( cd "${CARLA_SCRIPTS_ROOT}" && python -m snnmetrics.tests.run_all )
echo

# Convert if the tensors are not there, under the same lock and done-marker carla_eval.slurm uses.
DONE_MARK="${TENSORS}/sequence_lists/${CAPTURE_ID}_split_seq.csv"
mkdir -p "$(dirname "${TENSORS}")"
(
  flock 9
  if [ -f "${DONE_MARK}" ]; then
    echo "tensors already complete at ${TENSORS}"
  else
    echo "converting (no ${DONE_MARK})"
    python carla_eval/carla_to_voxel.py "${CAPTURE}" --out "${TENSORS}" --capture-id "${CAPTURE_ID}"
  fi
) 9>"${TENSORS}.lock"
[ -f "${DONE_MARK}" ] || { echo "conversion did not produce ${DONE_MARK}" >&2; exit 1; }
echo

rc=0
for M in snn ann; do
    case "${M}" in
      snn) CONFIG=configs/valid_DSEC_supervised_full.yml; NAME=sdformerflow ;;
      ann) CONFIG=configs/valid_DSEC_ann.yml;             NAME=sttflownet_en4 ;;
    esac
    RUNID="$(cat "hpc/logs/${M}_runid.txt" 2>/dev/null || true)"
    [ -n "${RUNID}" ] || { echo "no run id at hpc/logs/${M}_runid.txt"; exit 1; }

    echo "--- ${M} (${NAME}) ---"
    python -u carla_eval/spike_metrics.py \
        --config "${CONFIG}" \
        --runid "${RUNID}" \
        --path_mlflow "${SDF_MLFLOW_DIR}" \
        --tensors "${TENSORS}" \
        --id "${CAPTURE_ID}" \
        --windows "${WINDOWS}" \
        --out "${OUT}" \
        --name "${NAME}_$(basename "${CAPTURE}")" || { echo "${M} FAILED"; rc=1; }
    echo
done

echo "Done (rc=${rc}). Metrics in ${OUT}:"
ls -la "${OUT}"
exit "${rc}"
