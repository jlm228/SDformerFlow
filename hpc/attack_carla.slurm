#!/bin/bash
# One array task = one (model, objective), swept over every epsilon.
#
#   sbatch --array=1-10 hpc/attack_carla.slurm <capture_dir> <grid_manifest>
#
# Submitted by hpc/submit_attack_sweep.sh, which writes the manifest. Run that, not this.
#
# ONE ARRAY COVERS BOTH MODELS. SDformerFlow (SNN) and STTFlowNet-en4 (ANN) share this repo's
# venv and its converted voxel tensors, so the manifest carries snn|ann per line. That is also
# what makes the pair the study's controlled ablation: byte-identical input, only the neuron
# model differing.
#
# WHY A MANIFEST: this cluster's sbatch does not honour an in-script `#SBATCH --export=ALL`, so
# variables set at submission time silently do not arrive. carla_eval.slurm solves this with
# positional arguments; an array needs one more level, so the index selects a manifest line.
#
# Manifest line format (whitespace separated):
#   <snn|ann> <runid> <objective> <sign> <attack> <iters> <epsilon> [<epsilon> ...]

#SBATCH --job-name=sdf_attack
#SBATCH --qos=bbgpu
#SBATCH --account=arnaboll-ai-research
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# 16 h, not 8: at ~1 s per forward-equivalent the 55M-parameter swin SNN runs ~9-12 h for
# one objective over a ~100-window band at 6 epsilons x 10 PGD steps. The ANN is faster.
#SBATCH --time=16:0:0
#SBATCH --output=hpc/logs/%x_%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
mkdir -p hpc/logs

USAGE="usage: sbatch --array=1-N hpc/attack_carla.slurm <capture_dir> <grid_manifest>"
CAPTURE="${1:?${USAGE}}"
MANIFEST="${2:?${USAGE}}"

CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"
export CARLA_SCRIPTS_ROOT

[ -f "${MANIFEST}" ] || { echo "no manifest at ${MANIFEST}" >&2; exit 1; }
[ -d "${CARLA_SCRIPTS_ROOT}/attack_core" ] || {
  echo "no attack_core under ${CARLA_SCRIPTS_ROOT} -- set CARLA_SCRIPTS_ROOT" >&2; exit 1; }

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST}")"
[ -n "${LINE}" ] || { echo "manifest has no line ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
read -r MODEL RUNID OBJECTIVE SIGN ATTACK ITERS EPSILONS <<< "${LINE}"

CAPTURE_ID="carla_$(basename "${CAPTURE}")"
TENSORS="${CAPTURE}/saved_flow_data"
CLEAN="results/carla_eval/pred/${MODEL}"
OUT="results/attack/${MODEL}"
LABEL="${OBJECTIVE}"
[ "${OBJECTIVE}" = "div" ] && LABEL="div_${SIGN}"

[ -d "${TENSORS}" ] || {
  echo "no ${TENSORS} -- run hpc/carla_eval.slurm ${MODEL} ${CAPTURE} <runid> first" >&2
  exit 1; }
[ -d "${CLEAN}" ] || {
  echo "no clean predictions at ${CLEAN} -- run hpc/carla_eval.slurm first; every attacked" >&2
  echo "output directory is SEEDED from it, so a band-limited attack has no gaps" >&2
  exit 1; }

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${MODEL} / ${LABEL} ==="
echo "capture   ${CAPTURE} (id ${CAPTURE_ID})"
echo "attack    ${ATTACK} (${ITERS} iters)"
echo "epsilons  ${EPSILONS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Unquoted on purpose: --epsilons takes nargs='+'.
# shellcheck disable=SC2086
python carla_eval/attack_carla.py \
    --model "${MODEL}" \
    --runid "${RUNID}" \
    --path_mlflow "${SDF_MLFLOW_DIR}" \
    --tensors "${TENSORS}" \
    --capture "${CAPTURE}" \
    --id "${CAPTURE_ID}" \
    --objective "${OBJECTIVE}" \
    --sign "${SIGN}" \
    --attack "${ATTACK}" \
    --iters "${ITERS}" \
    --epsilons ${EPSILONS} \
    --clean-pred "${CLEAN}" \
    --out "${OUT}" \
    --report "${OUT}/reports" \
    ${DUMP_ADV_TENSORS:+--dump-adv-tensors "${DUMP_ADV_TENSORS}/${MODEL}"}

echo
echo "=== round trip: does the dumped .npy carry the div the objective reported? ==="
RT_FAIL=0
for REPORT in "${OUT}/reports/"*_"${LABEL}"_eps*.json; do
    python carla_eval/attack_carla.py --model "${MODEL}" --runid "${RUNID}" \
        --tensors "${TENSORS}" --capture "${CAPTURE}" --id "${CAPTURE_ID}" \
        --objective "${OBJECTIVE}" --epsilons 0 --clean-pred "${CLEAN}" --out "${OUT}" \
        --round-trip "${REPORT}" || RT_FAIL=1
done
[ "${RT_FAIL}" = "0" ] || { echo "round trip FAILED on at least one epsilon" >&2; exit 1; }

echo
echo "Done: ${MODEL} / ${LABEL}."
