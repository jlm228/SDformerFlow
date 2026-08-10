#!/bin/bash
# Evaluate a trained model on the DSEC validation split (5 sequences, 2152 samples) and print
# AEE / AAE / outlier %. GPU job, short.
#
#   sbatch hpc/evaluate.slurm configs/valid_DSEC_ann.yml             $(cat hpc/logs/ann_runid.txt)
#   sbatch hpc/evaluate.slurm configs/valid_DSEC_supervised.yml      $(cat hpc/logs/snn_runid.txt)
#   sbatch hpc/evaluate.slurm configs/valid_DSEC_supervised_full.yml $(cat hpc/logs/snn_runid.txt)
#
# CONFIG and RUNID are POSITIONAL arguments, not environment variables -- deliberately. This
# cluster's sbatch does not propagate the submitting shell's environment into the job even with
# an in-script `#SBATCH --export=ALL` directive (command-line `sbatch --export=ALL ...` does
# work here, as submit_train.sh relies on, but the script-directive form silently does not --
# apparently `--export` has to be resolved before sbatch parses the rest of the script). A
# `CONFIG=... RUNID=... sbatch hpc/evaluate.slurm` invocation would therefore silently submit a
# job with neither variable set. Positional args sidestep environment propagation entirely and
# work on any Slurm installation.
#
# Published targets (EPE / outlier % / AEE):
#   valid_DSEC_ann.yml             STTFlowNet-en4-b2-p4-w10        0.81 / 2.50 / 4.33  (Table 3 best)
#   valid_DSEC_supervised.yml      SDformerFlow-SPE-QK-s10-c2 (C)  0.93 / 3.17 / 6.37  (Table 4 best)
#   valid_DSEC_supervised_full.yml SDformerFlow-SPE-QK-s10-c2 (F)  1.61 / 8.91 / 7.23  (Table 4 best)
#
# C and F are the SAME trained SNN evaluated at two test resolutions -- do not train twice.
# Run both: two published numbers from one model, and if only one matches, the fault is in the
# resolution/remap handling rather than in training.
#
# For the equivalent ANN-vs-SNN comparison use the ANN (0.81) against the SNN's F column (1.61):
# same full-resolution test, capacity matched at 57.5 M vs 54.9 M.
#
# Once both are done:  python hpc/make_results_table.py

#SBATCH --job-name=sdf_eval
#SBATCH --qos=bbgpu
#SBATCH --account=arnaboll-ai-research
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:0:0
#SBATCH --output=hpc/logs/%x_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p hpc/logs
source hpc/env.sh

USAGE="usage: sbatch hpc/evaluate.slurm <config.yml> <runid>"
CONFIG="${1:?${USAGE}}"
RUNID="${2:?${USAGE}}"

[ -f "${CONFIG}" ] || { echo "ERROR: ${CONFIG} not found."; exit 1; }

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print('GPU:', torch.cuda.get_device_name(0))"

echo "config=${CONFIG} | runid=${RUNID} | mlflow=${SDF_MLFLOW_DIR}"
echo

# load_model raises on a run id that does not resolve, rather than silently evaluating a
# randomly initialised network -- so a bad RUNID fails here instead of printing plausible
# nonsense. Results also land under results_inference/<runid>/.
python eval_DSEC_flow_SNN.py \
    --config "${CONFIG}" \
    --path_mlflow "${SDF_MLFLOW_DIR}" \
    --runid "${RUNID}"

echo
echo "Done. Compare against the targets in this script's header, or run:"
echo "  python hpc/make_results_table.py"
