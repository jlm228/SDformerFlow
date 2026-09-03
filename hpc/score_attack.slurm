#!/bin/bash
# Score every attacked cell for BOTH models through the frozen planner, then render figures.
#
#   sbatch hpc/score_attack.slurm <capture_dir>
#
# Chained behind the attack array by submit_attack_sweep.sh. CPU only -- scoring is seconds per
# case, so this must not sit in the GPU queue.
#
# The avoidance package is model-free (numpy/pandas/cv2, no torch), so it scores this repo's two
# models and OF_EV_SNN's one through a single implementation, out of the CARLA-hpc-scripts
# checkout. Results land in the same directory OF_EV_SNN's scoring job writes to, so the grid
# figure spans all three.

#SBATCH --job-name=sdf_score_attack
#SBATCH --account=arnaboll-ai-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:0:0
#SBATCH --output=hpc/logs/%x_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
mkdir -p hpc/logs

USAGE="usage: sbatch hpc/score_attack.slurm <capture_dir>"
CAPTURE="${1:?${USAGE}}"
CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"

CAP_ABS="$(cd "${CAPTURE}" && pwd)"
SCEN="$(basename "${CAP_ABS}")"
CAPTURE_ID="carla_${SCEN}"
# Prefer the prefix the dumps actually carry, in case the capture was converted under
# another one; run_case would otherwise assume "carla_<basename>".
_PROBE="$(find "$(pwd)/results/carla_eval/pred" -name "*_[0-9][0-9][0-9][0-9].npy" 2>/dev/null | head -1)"
[ -n "${_PROBE}" ] && CAPTURE_ID="$(basename "${_PROBE}" .npy | sed "s/_[0-9]\{4\}$//")"
REPO="$(pwd)"   # already cd'd to SLURM_SUBMIT_DIR; $0 is Slurm's spooled copy

cd "${CARLA_SCRIPTS_ROOT}"
RES="results/attack-${SCEN}"
mkdir -p "${RES}/reports"

# The controls belong in the SAME directory as every attacked cell: viz/bev.py refuses to run
# without no_action (it supplies the recorded path), sweep.py refuses without gt_oracle (it is
# what deviation is measured against). Both are idempotent, and OF_EV_SNN's scoring job may
# have written them already.
for COND in no_action gt_oracle; do
    if [ ! -f "${RES}/${SCEN}_none_${COND}.json" ]; then
        echo "=== control: ${COND} ==="
        python -m avoidance.eval.run_case --capture "${CAP_ABS}" --model none \
            --condition "${COND}" --out "${RES}"
    fi
done

# SDformerFlow's dumps are prefixed carla_<capture>, unlike OF_EV_SNN's bare capture name, so
# --capture-id is needed on every case here.
for M in snn ann; do
    NAME="sdformerflow"; [ "${M}" = "ann" ] && NAME="sttflownet_en4"
    CLEAN="${REPO}/results/carla_eval/pred/${M}"
    ATTACK_ROOT="${REPO}/results/attack/${M}"

    if [ -d "${CLEAN}" ] && [ ! -f "${RES}/${SCEN}_${NAME}_clean.json" ]; then
        echo "=== clean anchor: ${NAME} ==="
        python -m avoidance.eval.run_case --capture "${CAP_ABS}" --model "${NAME}" \
            --condition clean --pred "${CLEAN}" --capture-id "${CAPTURE_ID}" --out "${RES}"
    fi

    echo
    echo "=== attacked cells: ${NAME} ==="
    for EPS_DIR in "${ATTACK_ROOT}"/*/eps*/; do
        [ -d "${EPS_DIR}" ] || continue
        EPS="$(basename "${EPS_DIR}")"; EPS="${EPS#eps}"
        OBJECTIVE="$(basename "$(dirname "${EPS_DIR}")")"
        # eps 0 is the clean case and objective-independent; scoring it once per objective
        # would put five identical rows in sweep.csv.
        if [ "$(python -c "print(float('${EPS}') == 0)")" = "True" ]; then continue; fi
        CASE="${NAME}__${OBJECTIVE}__eps${EPS}"
        echo "  ${CASE}"
        python -m avoidance.eval.run_case --capture "${CAP_ABS}" --model "${CASE}" \
            --condition clean --pred "${EPS_DIR}" --capture-id "${CAPTURE_ID}" --out "${RES}"
    done

    cp -f "${ATTACK_ROOT}/reports/"*.json "${RES}/reports/" 2>/dev/null || true
done

echo
bash avoidance/viz/make_figures.sh "${RES}" "${CAP_ABS}" "${REPO}/results/attack"
