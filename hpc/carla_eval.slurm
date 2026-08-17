#!/bin/bash
# Run a trained SDformerFlow / STTFlowNet over a CARLA capture: convert, predict, render, score.
#
#   sbatch hpc/carla_eval.slurm snn <capture_dir> $(cat hpc/logs/snn_runid.txt)
#   sbatch hpc/carla_eval.slurm ann <capture_dir> $(cat hpc/logs/ann_runid.txt)
#
# <capture_dir> is a capture_dataset.py recording (events.npy, flow_gt/, depth_gt/, seg/,
# windows.csv, meta.json) -- e.g. $BB_DATA/CARLA/50761933.
#
# MODEL, CAPTURE and RUNID are POSITIONAL for the same reason evaluate.slurm's are: this
# cluster's sbatch does not honour an in-script `#SBATCH --export=ALL`, so environment
# variables set at submission time silently do not arrive.
#
# The SNN uses valid_DSEC_supervised_full.yml, not the cropped config: every model must be
# scored over the same pixels, so compare against the reproduced full-resolution number.
#
# Scoring runs out of CARLA-hpc-scripts/score_flow.py rather than in-process, so models in
# environments that cannot be shared still go through one metric implementation. Point
# $CARLA_SCRIPTS_ROOT at that checkout.

#SBATCH --job-name=sdf_carla
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
mkdir -p hpc/logs
source hpc/env.sh

USAGE="usage: sbatch hpc/carla_eval.slurm <snn|ann> <capture_dir> <runid>"
MODEL="${1:?${USAGE}}"
CAPTURE="${2:?${USAGE}}"
RUNID="${3:?${USAGE}}"

CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"
export CARLA_SCRIPTS_ROOT

case "${MODEL}" in
  snn) CONFIG=configs/valid_DSEC_supervised_full.yml ;;
  ann) CONFIG=configs/valid_DSEC_ann.yml ;;
  *)   echo "ERROR: model must be 'snn' or 'ann', got '${MODEL}'"; exit 1 ;;
esac

[ -f "${CONFIG}" ] || { echo "ERROR: ${CONFIG} not found."; exit 1; }
[ -d "${CAPTURE}" ] || { echo "ERROR: capture dir ${CAPTURE} not found."; exit 1; }
[ -f "${CARLA_SCRIPTS_ROOT}/score_flow.py" ] || {
  echo "ERROR: ${CARLA_SCRIPTS_ROOT}/score_flow.py not found."
  echo "       Set CARLA_SCRIPTS_ROOT to your CARLA-hpc-scripts checkout."; exit 1; }

CAPTURE_ID="carla_$(basename "${CAPTURE}")"
TENSORS="${CAPTURE}/saved_flow_data"
OUT="results/carla_eval"
PRED="${OUT}/pred/${MODEL}"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print('GPU:', torch.cuda.get_device_name(0))"

echo "model=${MODEL} | config=${CONFIG} | runid=${RUNID}"
echo "capture=${CAPTURE} | id=${CAPTURE_ID}"
echo

# 1. Convert. Both models read the same 10-bin voxel tensors, so skip if already present and
#    the ann and snn jobs neither race nor duplicate the work.
if [ -d "${TENSORS}/event_tensors" ]; then
  echo "tensors already present at ${TENSORS} -- skipping conversion"
else
  python carla_eval/carla_to_voxel.py "${CAPTURE}" --out "${TENSORS}" --capture-id "${CAPTURE_ID}"
fi
echo

# 2. Representation checks -- cheap, and catch a stale conversion before the GPU work starts.
python carla_eval/test_representation_match.py \
    --carla-tensors "${TENSORS}" --id "${CAPTURE_ID}" \
    --dsec-root "${BB_DATA:-data/Datasets/DSEC}/saved_flow_data" \
    --dsec-raw "${BB_DATA:-data/Datasets/DSEC}"
echo

# 3. Predict.
python carla_eval/predict_carla.py \
    --config "${CONFIG}" \
    --runid "${RUNID}" \
    --path_mlflow "${SDF_MLFLOW_DIR}" \
    --tensors "${TENSORS}" \
    --id "${CAPTURE_ID}" \
    --out "${PRED}" \
    --video "${OUT}/${MODEL}_$(basename "${CAPTURE}").mp4" \
    --rgb "${CAPTURE}/rgb"
echo

# 4. Score, under the same masks and metrics as every other model.
PRIOR="${CARLA_SCRIPTS_ROOT}/../OF_EV_SNN/results/carla_eval/dsec_validity_prior.npy"
PRIOR_ARG=""
[ -f "${PRIOR}" ] && PRIOR_ARG="--prior ${PRIOR}"

python "${CARLA_SCRIPTS_ROOT}/score_flow.py" \
    --tensors "${TENSORS}" \
    --pred "${PRED}" \
    --name "${MODEL}_$(basename "${CAPTURE}")" \
    --capture "${CAPTURE}" \
    --outdir "${OUT}" \
    ${PRIOR_ARG}

echo
echo "Done. Feed the metrics CSV to the gate:"
echo "  python stage2_gate.py   # in the OF_EV_SNN checkout"
