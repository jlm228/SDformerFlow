#!/bin/bash
# Stage 2: train the SNN (SDformerFlow / MS_SpikingformerFlowNet_en4) on the DSEC training split.
#
# Do not run this until Stage 1 (hpc/train_ann.slurm) has met its exit criteria -- it shares the
# entire preprocessing/dataloader/MLflow/eval path, at roughly a sixth of the cost.
#
# Usually driven by hpc/submit_train.sh rather than sbatch'd directly:
#   bash hpc/submit_train.sh snn                        # chained segments
#   SMOKE=1 sbatch hpc/train_snn.slurm                  # 1 epoch, end-to-end check
#   PREV_RUNID=<mlflow_run_id> sbatch --export=ALL hpc/train_snn.slurm   # resume one segment by hand
#
# ~360k iterations (6000 samples / batch 1 x 60 epochs), or ~480k for the paper's 80 epochs,
# plus a 30-epoch full-resolution fine-tune. On an A100-80 that is several days, and BlueBEAR
# caps GPU jobs at 2 days, so this MUST be chained. Resume reads the rolling per-epoch
# checkpoint (model_latest / training_state_dict_latest), not the best-loss one.
#
# Single GPU deliberately: the multi-GPU branch in this script calls model.module.init_weights()
# without ever wrapping in DataParallel, so it is broken.

#SBATCH --job-name=sdf_train_snn
#SBATCH --qos=bbgpu
#SBATCH --account=arnaboll-ai-research
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-0:0:0
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --export=ALL
# 2 days is the hard cap for this QoS (`sacctmgr show qos bbgpu` -> MaxWall 2-00:00:00), so this
# requests the maximum. Smoke-test measured ~1h43m/epoch (60 epochs =~ 4.3 days), well over one
# segment, hence the multi-segment chain in hpc/submit_train.sh. Override with `sbatch --time=...`,
# which takes precedence over this directive -- submit_train.sh does that via WALLTIME.
#
# --export=ALL: without it, sbatch on this cluster does not propagate the submitting shell's
# environment into the job, so a direct `PREV_RUNID=... sbatch hpc/train_snn.slurm` would
# silently resume from nothing instead of the given run. submit_train.sh already passes
# --export=ALL itself; this makes a direct sbatch call safe too.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p hpc/logs
source hpc/env.sh

CONFIG="${CONFIG:-configs/train_DSEC_supervised_SDformerFlow_en4.yml}"
DATA_DIR="data/Datasets/DSEC/saved_flow_data"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print('GPU:', torch.cuda.get_device_name(0))"

# spikingjelly JIT-compiles its neuron kernels through CuPy on first use. That failing after
# several days of queueing is the worst way to find out, so check it up front.
python - <<'PY'
import torch
from spikingjelly.activation_based import functional, neuron
n = neuron.LIFNode(step_mode='m').cuda()
functional.set_backend(n, 'cupy', neuron.LIFNode)
n(torch.rand(4, 2, 8, device='cuda'))
print('cupy backend OK')
PY

# The SNN config uses num_chunks: 1, so DSECDatasetLite reads *_split_seq.csv. Only the
# doubleseq variants are published; derive the single-chunk ones with DSEC_dataloader/build_seq_split.py.
for split in train_split_seq.csv valid_split_seq.csv; do
    [ -f "${DATA_DIR}/sequence_lists/${split}" ] || {
        echo "ERROR: ${DATA_DIR}/sequence_lists/${split} not found."
        echo "Derive them from the doubleseq splits:"
        echo "  cd DSEC_dataloader && python build_seq_split.py"
        exit 1
    }
done

for d in event_tensors/10bins/left gt_tensors mask_tensors; do
    [ -d "${DATA_DIR}/${d}" ] || {
        echo "ERROR: ${DATA_DIR}/${d} missing -- run hpc/preprocess.slurm first."
        exit 1
    }
done

EXTRA_ARGS=""
if [ "${SMOKE:-0}" != "0" ]; then
    echo "[SMOKE] 1 epoch only"
    EXTRA_ARGS="--smoke_epochs 1"
fi

# Each segment is a fresh MLflow run that resumes the previous one's weights and optimiser
# state, so the chain is a sequence of run ids. RUNID_FILE carries the id from one segment to
# the next: this segment reads it (if resuming) and then overwrites it with its own id.
RUNID_FILE="${RUNID_FILE:-hpc/logs/snn_runid.txt}"

if [ -z "${PREV_RUNID:-}" ] && [ "${CHAIN:-0}" != "0" ] && [ -f "${RUNID_FILE}" ]; then
    PREV_RUNID="$(cat "${RUNID_FILE}")"
    echo "chained resume: picked up ${PREV_RUNID} from ${RUNID_FILE}"
fi

echo "config=${CONFIG} | mlflow=${SDF_MLFLOW_DIR} | resume=${PREV_RUNID:-<none>}"

python train_flow_parallel_supervised_SNN.py \
    --config "${CONFIG}" \
    --path_mlflow "${SDF_MLFLOW_DIR}" \
    --runid_file "${RUNID_FILE}" \
    ${PREV_RUNID:+--resume 1 --prev_runid "${PREV_RUNID}"} \
    ${EXTRA_ARGS}

echo "Done. Run id recorded in ${RUNID_FILE} -- it is the only handle on the checkpoint."
