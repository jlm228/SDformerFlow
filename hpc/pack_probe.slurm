#!/bin/bash
# How many SA-PGD attack processes one GPU can usefully run at once.
#
#   sbatch hpc/pack_probe.slurm <capture_dir> [band_lo]
#   PACKS="1 2 3" WINDOWS=4 ITERS=50 ATTACK=sapgd-pdsg sbatch hpc/pack_probe.slurm <capture_dir>
#   sbatch --gres=gpu:a100_40:1 hpc/pack_probe.slurm <capture_dir>     # the same on a 40 GB card
#
# The SNN cannot batch (window_partition_v2 couples the copies in a batch, see attack_carla.py),
# so at batch 1 the GPU may sit idle between the torch backend's small per-time-step kernels.
# Running P independent attacks side by side fills those gaps if there are any. Each phase runs
# P copies of the SAME short attack (WINDOWS windows x ITERS steps x 1 epsilon) concurrently and
# times the attack itself, from the band line to the reports line, so model loading and seeding
# do not count. If MPS is available, the largest P runs a second time under it.
#
# Read the summary at the end: aggregate passes/s against P = 1 is the speed-up packing gives.
# Nothing is written under results/ or captures/: the converted tensors (when the capture has
# none), every dump and every report go to a per-job scratch directory removed at the end. The
# capture needs its attack_band.json, and the SNN's clean predictions must exist.

#SBATCH --job-name=sdf_pack_probe
#SBATCH --qos=bbgpu
#SBATCH --account=arnaboll-ai-research
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Enough CPU and host memory for three attack processes, each sized as attack_carla.slurm sizes
# one: 4 cores, and a share of the host memory.
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1:0:0
#SBATCH --output=hpc/logs/%x_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
mkdir -p hpc/logs

USAGE="usage: sbatch hpc/pack_probe.slurm <capture_dir> [band_lo]"
CAPTURE="${1:?${USAGE}}"
BAND_LO="${2:-}"

PACKS="${PACKS:-1 2 3}"
WINDOWS="${WINDOWS:-4}"
ITERS="${ITERS:-50}"
ATTACK="${ATTACK:-sapgd-pdsg}"
EPSILON="${EPSILON:-0.02}"

CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"
export CARLA_SCRIPTS_ROOT
[ -d "${CARLA_SCRIPTS_ROOT}/attack_core" ] || {
  echo "no attack_core under ${CARLA_SCRIPTS_ROOT} -- set CARLA_SCRIPTS_ROOT" >&2; exit 1; }

RUNID="${SNN_RUNID:-$(cat hpc/logs/snn_runid.txt 2>/dev/null || true)}"
[ -n "${RUNID}" ] || { echo "no SNN run id (hpc/logs/snn_runid.txt or SNN_RUNID)" >&2; exit 1; }

OPTIMISER="${ATTACK%%-*}"
SURROGATE="${ATTACK#*-}"
[ "${SURROGATE}" = "${ATTACK}" ] && SURROGATE="native"
ASSG_ARGS=""
case "${SURROGATE}" in
  assg|assgs) ASSG_ARGS="--assg-A ${ASSG_A:?set ASSG_A for surrogate ${SURROGATE}}" ;;
esac

CAPTURE_ID="carla_$(basename "${CAPTURE}")"
CLEAN="results/carla_eval/pred/snn"
[ -d "${CLEAN}" ] || { echo "no clean predictions at ${CLEAN}" >&2; exit 1; }

SCRATCH="${TMPDIR:-/tmp}/sdf_pack_probe_${SLURM_JOB_ID:-$$}"
mkdir -p "${SCRATCH}"
trap 'rm -rf "${SCRATCH}"' EXIT

# The sweep converts each capture on demand and removes the tensors afterwards, so they are
# usually not there. Reuse them if a running sweep left them; otherwise convert into scratch,
# so this probe writes nothing under captures/ and its copy goes when the job ends.
TENSORS="${CAPTURE}/saved_flow_data"
if [ ! -f "${TENSORS}/sequence_lists/${CAPTURE_ID}_split_seq.csv" ]; then
  TENSORS="${SCRATCH}/saved_flow_data"
  echo "converting ${CAPTURE} into scratch (not timed)"
  python carla_eval/carla_to_voxel.py "${CAPTURE}" --out "${TENSORS}" --capture-id "${CAPTURE_ID}"
  [ -f "${TENSORS}/sequence_lists/${CAPTURE_ID}_split_seq.csv" ] || {
    echo "conversion did not produce ${TENSORS}/sequence_lists/${CAPTURE_ID}_split_seq.csv" >&2
    exit 1; }
fi

# Start of the band unless given: the windows the real sweep attacks, so the input is typical.
if [ -z "${BAND_LO}" ]; then
  BAND_LO="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['band_lo'])" \
             "${CAPTURE}/attack_band.json")"
fi
BAND_HI=$((BAND_LO + WINDOWS - 1))

# Passes one process makes inside the timed span: the start plus ITERS steps per window, and
# the final re-score. The re-score is forward-only but counted whole; it is the same in every
# phase, so the ratios are unaffected.
PASSES_PER_PROC=$((WINDOWS * (ITERS + 2)))

echo "=== pack probe: ${ATTACK} on ${CAPTURE_ID}, windows [${BAND_LO}, ${BAND_HI}], ${ITERS} iters, eps ${EPSILON} ==="
nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv,noheader
echo "packs: ${PACKS}"
echo

run_one() {
  # One attack, every output line stamped with epoch seconds.
  local tag="$1"
  OMP_NUM_THREADS=4 python -u carla_eval/attack_carla.py \
      --model snn --runid "${RUNID}" --path_mlflow "${SDF_MLFLOW_DIR}" \
      --tensors "${TENSORS}" --capture "${CAPTURE}" --id "${CAPTURE_ID}" \
      --objective div --sign suppress \
      --attack "${OPTIMISER}" --surrogate "${SURROGATE}" ${ASSG_ARGS} \
      --norm-set "${NORM_SET:-floating}" \
      --iters "${ITERS}" --epsilons "${EPSILON}" \
      --band-lo "${BAND_LO}" --band-hi "${BAND_HI}" \
      --clean-pred "${CLEAN}" --out "${SCRATCH}/${tag}/out" --report "${SCRATCH}/${tag}/rep" \
      2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "${line}"; done \
      > "${SCRATCH}/${tag}.log"
}

attack_span() {
  # Seconds from the band line (printed just before the sweep starts) to the reports line.
  awk '/\| band \[/ && !s {s=$1} /^[0-9.]+ reports:/ {e=$1} END {if (s && e) printf "%.1f", e-s; else print "NA"}' "$1"
}

declare -A RESULT
phase() {
  local p="$1" name="$2"
  echo "--- ${name}: ${p} process(es) ---"
  nvidia-smi dmon -s um -d 2 > "${SCRATCH}/${name}.dmon" 2>/dev/null &
  local dmon=$!
  local pids=() k
  for k in $(seq 1 "${p}"); do
    run_one "${name}_${k}" &
    pids+=($!)
  done
  local fail=0
  for k in "${pids[@]}"; do wait "${k}" || fail=1; done
  kill "${dmon}" 2>/dev/null || true; wait "${dmon}" 2>/dev/null || true

  local spans=() worst=0 s
  for k in $(seq 1 "${p}"); do
    s="$(attack_span "${SCRATCH}/${name}_${k}.log")"
    spans+=("${s}")
    if [ "${s}" = "NA" ]; then
      fail=1
      echo "  process ${k} did not finish; its last lines:"
      tail -n 15 "${SCRATCH}/${name}_${k}.log" | sed 's/^/    /'
    else
      worst="$(awk -v a="${worst}" -v b="${s}" 'BEGIN {print (b > a) ? b : a}')"
    fi
  done
  # Mean SM and memory-bandwidth utilisation over the phase, skipping dmon's header lines.
  local util
  util="$(awk '!/^#/ && NF >= 3 {sm += $2; mem += $3; n++} END {if (n) printf "sm %.0f%%  mem %.0f%%", sm/n, mem/n; else print "n/a"}' "${SCRATCH}/${name}.dmon")"
  echo "  attack spans (s): ${spans[*]}   gpu: ${util}"
  if [ "${fail}" = "0" ]; then
    local rate
    rate="$(awk -v n="$((p * PASSES_PER_PROC))" -v t="${worst}" 'BEGIN {printf "%.2f", n / t}')"
    RESULT["${name}"]="${p} ${worst} ${rate} ${util}"
    echo "  aggregate ${rate} passes/s"
  else
    RESULT["${name}"]="${p} FAILED"
  fi
  # Peak memory per process is not visible from here; an OOM shows up as a failed phase.
  echo
}

ORDER=()
for P in ${PACKS}; do
  phase "${P}" "p${P}"
  ORDER+=("p${P}")
done

# MPS lets kernels from different processes run at the same time rather than time-slice. It
# needs no root when the GPU is in Default compute mode; if it will not start, skip it.
MAXP="$(echo ${PACKS} | tr ' ' '\n' | sort -n | tail -n 1)"
if [ "${MAXP}" -gt 1 ] && command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  export CUDA_MPS_PIPE_DIRECTORY="${SCRATCH}/mps_pipe" CUDA_MPS_LOG_DIRECTORY="${SCRATCH}/mps_log"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
  if nvidia-cuda-mps-control -d; then
    phase "${MAXP}" "mps${MAXP}"
    ORDER+=("mps${MAXP}")
    echo quit | nvidia-cuda-mps-control || true
  else
    echo "MPS did not start; skipped"
  fi
else
  echo "MPS not available (or only one pack size); skipped"
fi

echo
echo "=== summary: ${ATTACK}, ${WINDOWS} windows x ${ITERS} iters, ${PASSES_PER_PROC} passes per process ==="
BASE="$(echo "${RESULT[p1]:-}" | awk '{print $3}')"
printf "%-8s %4s %10s %12s %9s  %s\n" phase P "span (s)" "passes/s" "vs P=1" "gpu"
for name in "${ORDER[@]}"; do
  read -r p span rate rest <<< "${RESULT[${name}]}"
  if [ "${span}" = "FAILED" ]; then
    printf "%-8s %4s %10s\n" "${name}" "${p}" "FAILED"
    continue
  fi
  speed="-"
  [ -n "${BASE}" ] && speed="$(awk -v r="${rate}" -v b="${BASE}" 'BEGIN {printf "%.2fx", r / b}')"
  printf "%-8s %4s %10s %12s %9s  %s\n" "${name}" "${p}" "${span}" "${rate}" "${speed}" "${rest}"
done
echo
echo "'vs P=1' is the throughput gain per GPU from packing; multiply GPU-hour estimates by its"
echo "inverse. A FAILED phase at the top P is most likely out of memory."
