#!/bin/bash
# The whole SDformerFlow + STTFlowNet attack sweep, from one command.
#
#   bash hpc/submit_attack_sweep.sh <capture_dir>
#   SMOKE=1 bash hpc/submit_attack_sweep.sh <capture_dir>     # 2 epsilons, 4 iters
#
# Submits one GPU array covering BOTH models x five objectives (10 tasks), then a CPU job
# chained on --dependency=afterok that scores every cell and renders the figures.
#
# Both models live here and share a venv and the converted voxel tensors, so one submission
# covers them. OF_EV_SNN cannot join: it needs spikingjelly.clock_driven where this repo needs
# activation_based. Run OF_EV_SNN/hpc/submit_attack_sweep.sh for the third model -- two
# submissions cover the study.
#
# Prerequisites, all fatal if missing:
#   * clean runs: sbatch hpc/carla_eval.slurm snn <capture> $(cat hpc/logs/snn_runid.txt)
#                 sbatch hpc/carla_eval.slurm ann <capture> $(cat hpc/logs/ann_runid.txt)
#   * the band:   python -m attack_core.band --capture <capture>   (in CARLA-hpc-scripts)

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p hpc/logs results/attack

USAGE="usage: bash hpc/submit_attack_sweep.sh <capture_dir>"
CAPTURE="${1:?${USAGE}}"
CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"

SNN_RUNID="${SNN_RUNID:-$(cat hpc/logs/snn_runid.txt 2>/dev/null || true)}"
ANN_RUNID="${ANN_RUNID:-$(cat hpc/logs/ann_runid.txt 2>/dev/null || true)}"
[ -n "${SNN_RUNID}" ] || { echo "ERROR: no SNN run id (hpc/logs/snn_runid.txt)"; exit 1; }
[ -n "${ANN_RUNID}" ] || { echo "ERROR: no ANN run id (hpc/logs/ann_runid.txt)"; exit 1; }

[ -f "${CAPTURE}/attack_band.json" ] || {
  echo "ERROR: no ${CAPTURE}/attack_band.json."
  echo "       Compute it once, in an environment with avoidance's dependencies:"
  echo "         cd ${CARLA_SCRIPTS_ROOT} && python -m attack_core.band --capture ${CAPTURE}"
  exit 1; }

for M in snn ann; do
  [ -d "results/carla_eval/pred/${M}" ] || {
    echo "ERROR: no clean predictions at results/carla_eval/pred/${M}."
    echo "       sbatch hpc/carla_eval.slurm ${M} ${CAPTURE} <runid>"; exit 1; }
done

# Both models read BYTE-IDENTICAL voxel tensors, so one epsilon list serves both and an epsilon
# means the same perturbation on each -- which is what makes the SNN/ANN pair a controlled
# ablation rather than a comparison at two different budgets. The values differ from
# OF_EV_SNN's: this representation is signed and runs [-14.20, 10.89] at 10.24% occupancy.
EPSILONS="${EPSILONS:-0.0 0.005 0.01 0.02 0.05 0.1}"
ITERS="${ITERS:-10}"
ATTACK="${ATTACK:-pgd}"
if [ "${SMOKE:-0}" != "0" ]; then
    EPSILONS="0.0 0.02"
    ITERS=4
    echo "[SMOKE] epsilons ${EPSILONS}, ${ITERS} iters"
fi

# Stage 6 check 3: one unbounded epsilon, 10x the largest. If even this fails to force a
# collision, something is masking the gradient -- and it costs one extra epsilon value.
EPS_HUGE="$(python -c "import sys; print('%g' % (10 * max(float(v) for v in sys.argv[1:])))" ${EPSILONS})"
SWEEP_EPS="${EPSILONS} ${EPS_HUGE}"

MANIFEST="hpc/logs/attack_grid_$(basename "${CAPTURE}").txt"
: > "${MANIFEST}"
for M in snn ann; do
  RUNID="${SNN_RUNID}"; [ "${M}" = "ann" ] && RUNID="${ANN_RUNID}"
  {
    echo "${M} ${RUNID} random_sign  none      ${ATTACK} ${ITERS} ${SWEEP_EPS}"
    echo "${M} ${RUNID} epe_global   none      ${ATTACK} ${ITERS} ${SWEEP_EPS}"
    echo "${M} ${RUNID} epe_masked   none      ${ATTACK} ${ITERS} ${SWEEP_EPS}"
    echo "${M} ${RUNID} div          suppress  ${ATTACK} ${ITERS} ${SWEEP_EPS}"
    echo "${M} ${RUNID} div          inflate   ${ATTACK} ${ITERS} ${SWEEP_EPS}"
    # FGSM against the same objective, for the one-step-beats-iterative comparison. Skipped
    # when ATTACK is already fgsm, or this row would duplicate the one above and two array
    # tasks would write the same output directory.
    if [ "${ATTACK}" != "fgsm" ]; then
      echo "${M} ${RUNID} div          suppress  fgsm 1 ${SWEEP_EPS}"
    fi
  } >> "${MANIFEST}"
done
N=$(wc -l < "${MANIFEST}")

echo "manifest  ${MANIFEST} (${N} tasks: 2 models x 5 objectives)"
sed 's/^/    /' "${MANIFEST}"
echo

# The transfer check needs the perturbed INPUT tensors, not just the predictions. They are
# byte-identical between these two models, so an ANN-derived perturbation feeds the SNN with no
# re-encoding -- that is Stage 6 check 2, and it is only defined within this pair.
export DUMP_ADV_TENSORS="${DUMP_ADV_TENSORS:-results/attack/adv_tensors}"

ARRAY_ID=$(sbatch --parsable --array=1-"${N}" \
    hpc/attack_carla.slurm "${CAPTURE}" "${MANIFEST}")
echo "attack array : job ${ARRAY_ID} (1-${N})"

# afterany, not afterok: one failed cell must not block scoring and figures for the
# rest. sweep.py reports what is missing.
SCORE_ID=$(sbatch --parsable --dependency=afterany:"${ARRAY_ID}" \
    hpc/score_attack.slurm "${CAPTURE}")
echo "score+figures: job ${SCORE_ID} (after ${ARRAY_ID})"
echo
echo "Watch with: squeue --me"
echo "A failed cell reruns alone: sbatch --array=<index> hpc/attack_carla.slurm ${CAPTURE} ${MANIFEST}"
