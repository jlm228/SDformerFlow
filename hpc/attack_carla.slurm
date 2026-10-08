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
#
# PACK=k runs k manifest lines side by side on the one GPU: task i covers lines
# (i-1)*k+1 .. i*k. At batch 1 the SNN leaves the GPU mostly idle (37% SM busy), and two
# processes give 1.27x the throughput (hpc/pack_probe.slurm, A100 80 GB). Three do not fit: each
# process peaks at ~26 GB. Each line still runs, dumps and round-trips exactly as it does alone;
# it is ~1.6x slower in wall time, so size --time for that. PACK defaults to 1, so a direct
# `sbatch --array=<line>` reruns one line, as before. submit_attack_sweep.sh sets PACK and
# scales --cpus-per-task and --mem with it.

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

CAPTURE_ID="carla_$(basename "${CAPTURE}")"
TENSORS="${CAPTURE}/saved_flow_data"

# Build the tensors if they are not there, exactly as carla_eval.slurm does. Conversion writes
# ~1.1 GB per capture, which submit_attack_sweep.sh removes once the whole array has finished.
DONE_MARK="${TENSORS}/sequence_lists/${CAPTURE_ID}_split_seq.csv"
mkdir -p "$(dirname "${TENSORS}")"
(
  flock 9
  if [ -f "${DONE_MARK}" ]; then
    echo "tensors already complete at ${TENSORS} -- skipping conversion"
  else
    echo "converting (no ${DONE_MARK})"
    python carla_eval/carla_to_voxel.py "${CAPTURE}" --out "${TENSORS}" --capture-id "${CAPTURE_ID}"
  fi
) 9>"${TENSORS}.lock"

[ -f "${DONE_MARK}" ] || { echo "conversion did not produce ${DONE_MARK}" >&2; exit 1; }

PACK="${PACK:-1}"
FIRST=$(( (SLURM_ARRAY_TASK_ID - 1) * PACK + 1 ))
LAST=$(( SLURM_ARRAY_TASK_ID * PACK ))
TOTAL="$(wc -l < "${MANIFEST}")"
[ "${FIRST}" -le "${TOTAL}" ] || { echo "manifest has no line ${FIRST}" >&2; exit 1; }
[ "${LAST}" -le "${TOTAL}" ] || LAST="${TOTAL}"

# One manifest line: its attack, then its round trip. Runs in a subshell, so an `exit` ends
# this line only and the other packed line carries on.
run_line() {
  LINE_NO="$1"
  LINE="$(sed -n "${LINE_NO}p" "${MANIFEST}")"
  [ -n "${LINE}" ] || { echo "manifest has no line ${LINE_NO}" >&2; exit 1; }
  read -r MODEL RUNID OBJECTIVE SIGN ATTACK ITERS EPSILONS <<< "${LINE}"

  # The manifest's <attack> is "<optimiser>-<surrogate>" for everything new (sapgd-assg,
  # sapgd-native), and the bare "pgd"/"fgsm" for the runs already on disk. Split it here so the
  # manifest stays one column and the labels keep matching the dump directories.
  OPTIMISER="${ATTACK%%-*}"
  SURROGATE="${ATTACK#*-}"
  [ "${SURROGATE}" = "${ATTACK}" ] && SURROGATE="native"

  # The paper's setting for SA-PGD is N = 100; a manifest that leaves ITERS at the PGD default
  # would quietly run a tenth of the attack.
  if [ "${OPTIMISER}" = "sapgd" ] && [ "${ITERS}" -lt 100 ]; then
    echo "warning: ${ATTACK} with --iters ${ITERS}; the published setting is 100" >&2
  fi

  # ASSG's A is tuned per model and frozen. Passed through the environment so the manifest
  # format does not change. --norm-set defaults to floating: both paths re-normalise over the
  # perturbed input, so the function attacked is the one scored.
  ASSG_ARGS=""
  case "${SURROGATE}" in
    assg|assgs) ASSG_ARGS="--assg-A ${ASSG_A:?set ASSG_A for surrogate ${SURROGATE}}" ;;
  esac

  CLEAN="results/carla_eval/pred/${MODEL}"
  # OUT_ROOT lets a calibration run (hpc/submit_assg_grid.sh) keep its cells out of the sweep's
  # results/attack, which the sweep submitters clear and score.
  OUT="${OUT_ROOT:-results/attack}/${MODEL}"
  # One reports directory per capture. Report names carry model, label and epsilon but not the
  # capture, so captures sharing one directory overwrite each other's reports, and the round trip
  # below would check this capture's dumps against another capture's report.
  REPORTS="${OUT}/reports/${CAPTURE_ID}"
  LABEL="${OBJECTIVE}"
  [ "${OBJECTIVE}" = "div" ] && LABEL="div_${SIGN}"
  # The attack is part of the label: one manifest runs the same objective under both PGD
  # and FGSM, which would otherwise write the same output files.
  LABEL="${LABEL}_${ATTACK}"

  # RAND_INIT=0 starts PGD at the clean input. Separates a bad random start from a bad
  # gradient when PGD underperforms FGSM.
  NRI=""
  if [ "${RAND_INIT:-1}" = "0" ]; then
    NRI="--no-rand-init"
    LABEL="${LABEL}_nri"
  fi

  # SDA has no epsilon: the manifest's last column carries tau factors (target mode, the core
  # matrix) or event budgets from `calibrate --events` (budget mode). SDA_MODE picks which, and
  # ITERS is SDA's N rather than a step count.
  SDA_ARGS=""
  if [ "${OPTIMISER}" = "sda" ]; then
    case "${SDA_MODE:-target}" in
      target) SDA_ARGS="--sda-tau-factors ${EPSILONS}" ;;
      budget) SDA_ARGS="--sda-budgets ${EPSILONS}" ;;
      *) echo "SDA_MODE must be target or budget, got '${SDA_MODE}'" >&2; exit 1 ;;
    esac
    SDA_ARGS="${SDA_ARGS} --sda-iters ${ITERS}"
    SDA_ARGS="${SDA_ARGS} --sda-directions ${SDA_DIRECTIONS:-inject_only}"
    SDA_ARGS="${SDA_ARGS} --sda-rank ${SDA_RANK:-grad}"
    [ -n "${SDA_K_INIT:-}" ] && SDA_ARGS="${SDA_ARGS} --sda-k-init ${SDA_K_INIT}"
    [ -n "${SDA_FD_BATCH:-}" ] && SDA_ARGS="${SDA_ARGS} --sda-fd-batch ${SDA_FD_BATCH}"
    [ -n "${SDA_EPE_MARGIN:-}" ] && SDA_ARGS="${SDA_ARGS} --sda-epe-margin ${SDA_EPE_MARGIN}"
    LEVEL_ARGS="${SDA_ARGS}"
  else
    LEVEL_ARGS="--iters ${ITERS} ${NRI} --epsilons ${EPSILONS}"
  fi

  # Spike and operation counts on each band window's scoring pass, per epsilon, on by default
  # (SPIKE_METRICS=0 turns them off; SDA does not support them). Kept out of results/attack,
  # which score_attack.slurm walks for prediction dumps.
  SPIKE_ARGS=""
  if [ "${SPIKE_METRICS:-1}" != "0" ] && [ "${OPTIMISER}" != "sda" ]; then
    SPIKE_ARGS="--spike-metrics ${OUT_ROOT:-results/attack}_spike_metrics/${MODEL}/${CAPTURE_ID}"
  fi


  [ -d "${CLEAN}" ] || {
    echo "no clean predictions at ${CLEAN} -- run hpc/carla_eval.slurm first; every attacked" >&2
    echo "output directory is SEEDED from it, so a band-limited attack has no gaps" >&2
    exit 1; }

  echo "=== task ${SLURM_ARRAY_TASK_ID}, manifest line ${LINE_NO}: ${MODEL} / ${LABEL} ==="
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
      --attack "${OPTIMISER}" \
      --surrogate "${SURROGATE}" ${ASSG_ARGS} \
      --norm-set "${NORM_SET:-floating}" \
      ${LEVEL_ARGS} \
      --clean-pred "${CLEAN}" \
      --out "${OUT}" \
      --report "${REPORTS}" ${SPIKE_ARGS} \
      ${DUMP_ADV_TENSORS:+--dump-adv-tensors "${DUMP_ADV_TENSORS}/${MODEL}"}

  echo
  echo "=== round trip: does the dumped .npy carry the div the objective reported? ==="
  RT_FAIL=0
  for REPORT in "${REPORTS}/"*_"${LABEL}"_eps*.json; do
      python carla_eval/attack_carla.py --model "${MODEL}" --runid "${RUNID}" \
          --tensors "${TENSORS}" --capture "${CAPTURE}" --id "${CAPTURE_ID}" \
          --objective "${OBJECTIVE}" --epsilons 0 --clean-pred "${CLEAN}" --out "${OUT}" \
          --round-trip "${REPORT}" || RT_FAIL=1
  done
  [ "${RT_FAIL}" = "0" ] || { echo "round trip FAILED on at least one epsilon" >&2; exit 1; }

  echo
  echo "Done: ${MODEL} / ${LABEL}."
}

if [ "${FIRST}" -eq "${LAST}" ]; then
  ( run_line "${FIRST}" )
  exit $?
fi

# Packed: each line's output carries its manifest line number, since both share this log.
echo "=== task ${SLURM_ARRAY_TASK_ID}: manifest lines ${FIRST}-${LAST}, packed on one GPU ==="
PIDS=()
for N in $(seq "${FIRST}" "${LAST}"); do
  # The group, not the bare pipeline, goes to the background: then $! is the group, whose status
  # under pipefail is the line's, rather than sed's, which always succeeds.
  { ( run_line "${N}" ) 2>&1 | sed -u "s/^/[L${N}] /"; } &
  PIDS+=($!)
done
FAILED=""
for I in "${!PIDS[@]}"; do
  wait "${PIDS[${I}]}" || FAILED="${FAILED} $((FIRST + I))"
done
if [ -n "${FAILED}" ]; then
  echo "FAILED manifest line(s):${FAILED} -- rerun one alone with" >&2
  echo "  sbatch --array=<line> hpc/attack_carla.slurm ${CAPTURE} ${MANIFEST}" >&2
  exit 1
fi
echo "Done: manifest lines ${FIRST}-${LAST}."
