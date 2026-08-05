#!/bin/bash
# Stage 1: train the ANN (STTFlowNet-en4-b2-p4-w10) on the DSEC training split.
#
#   SMOKE=1 sbatch hpc/train_ann.slurm                      # 1 epoch, checks end-to-end
#   sbatch hpc/train_ann.slurm                              # full 100-epoch run
#   PREV_RUNID=<mlflow_run_id> sbatch hpc/train_ann.slurm   # resume an interrupted run
#   CONFIG=configs/train_DSEC_supervised_STT_voxel.yml sbatch hpc/train_ann.slurm   # en3 variant
#
# Default config reproduces the BEST row of the paper's Table 3: STTFlowNet-en4-b2-p4-w10,
# voxel input, trained at full 480x640, 57.51 M params -> EPE 0.81 / 2.50% outliers / AEE 4.33.
# Evaluate afterwards with configs/valid_DSEC_ann.yml.
#
# The repo's original train_DSEC_supervised_STT_voxel.yml is en3-b2-p4-w9 at 288x384 and matches
# no Table 3 row; it is still selectable via CONFIG= above.
#
# ~75k iterations (6000 samples / batch 8 x 100 epochs). At full resolution with 57.5 M params
# this is roughly 20-35 h -- it should fit one job under BlueBEAR's 2-day GPU cap, but not by a
# huge margin. If it is killed by the walltime, resume with PREV_RUNID (the rolling per-epoch
# checkpoint means you lose at most one epoch).

#SBATCH --job-name=sdf_train_ann
#SBATCH --qos=bbgpu
#SBATCH --account=arnaboll-ai-research
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-0:0:0
#SBATCH --output=hpc/logs/%x_%j.out
# 2 days is the hard cap for this QoS (`sacctmgr show qos bbgpu` -> MaxWall 2-00:00:00), so this
# requests the maximum. At ~20-35 h the ANN should finish inside a single segment; the chain in
# hpc/submit_train.sh exists as insurance, not necessity. Override with `sbatch --time=...`,
# which takes precedence over this directive -- submit_train.sh does that via WALLTIME.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p hpc/logs
source hpc/env.sh

CONFIG="${CONFIG:-configs/train_DSEC_supervised_STT_voxel_en4.yml}"
DATA_DIR="data/Datasets/DSEC/saved_flow_data"
RUNID_FILE="${RUNID_FILE:-hpc/logs/ann_runid.txt}"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print('GPU:', torch.cuda.get_device_name(0))"

# Preflight. Each of these otherwise surfaces either as a confusing stack trace hours in, or --
# worse -- as a run that quietly trains on less data than intended.
#
# The ANN configs use num_chunks: 2, so DSECDatasetLite reads *_split_doubleseq.csv. Those CSVs
# are not in this repo; they come from OF_EV_SNN (6000 train / 2152 valid rows).
for split in train_split_doubleseq.csv valid_split_doubleseq.csv; do
    [ -f "${DATA_DIR}/sequence_lists/${split}" ] || {
        echo "ERROR: ${DATA_DIR}/sequence_lists/${split} not found."
        echo "Copy them from OF_EV_SNN/data/dataset/saved_flow_data/sequence_lists/."
        exit 1
    }
done

for d in event_tensors/10bins/left gt_tensors mask_tensors; do
    [ -d "${DATA_DIR}/${d}" ] || {
        echo "ERROR: ${DATA_DIR}/${d} missing -- run hpc/preprocess.slurm first."
        exit 1
    }
done

# A partially failed preprocess array is the likeliest cause of close-but-wrong numbers, so
# count rather than just checking the directory exists.
n_gt=$(find "${DATA_DIR}/gt_tensors" -name '*.npy' | wc -l)
echo "ground-truth tensors: ${n_gt}"
[ "${n_gt}" -gt 0 ] || { echo "ERROR: no gt_tensors -- preprocessing produced no ground truth."; exit 1; }

EXTRA_ARGS=""
if [ "${SMOKE:-0}" != "0" ]; then
    echo "[SMOKE] 1 epoch only"
    EXTRA_ARGS="--smoke_epochs 1"
fi

# Chained segments: this segment reads the previous one's run id from RUNID_FILE and then
# overwrites it with its own, so a pre-submitted --dependency chain needs no run ids up front.
if [ -z "${PREV_RUNID:-}" ] && [ "${CHAIN:-0}" != "0" ] && [ -f "${RUNID_FILE}" ]; then
    PREV_RUNID="$(cat "${RUNID_FILE}")"
    echo "chained resume: picked up ${PREV_RUNID} from ${RUNID_FILE}"
fi

echo "config=${CONFIG} | mlflow=${SDF_MLFLOW_DIR} | resume=${PREV_RUNID:-<none>}"
echo "Expect 'Total parameters' ~57.5 M for en4. ~20.3 M means the en3 architecture."

python train_flow_parallel_supervised.py \
    --config "${CONFIG}" \
    --path_mlflow "${SDF_MLFLOW_DIR}" \
    --runid_file "${RUNID_FILE}" \
    ${PREV_RUNID:+--resume 1 --prev_runid "${PREV_RUNID}"} \
    ${EXTRA_ARGS}

echo
echo "Done. Run id in ${RUNID_FILE}. Evaluate with:"
echo "  CONFIG=configs/valid_DSEC_ann.yml RUNID=\$(cat ${RUNID_FILE}) sbatch hpc/evaluate.slurm"
