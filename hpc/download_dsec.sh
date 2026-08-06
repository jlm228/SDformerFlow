#!/bin/bash
# Download and unpack the DSEC data SDformerFlow needs, into the layout the preprocessing
# script expects.
#
#   bash hpc/download_dsec.sh                      # all 18 flow sequences (~38 GB)
#   bash hpc/download_dsec.sh thun_00_a            # one sequence (~0.3 GB, good first test)
#   CHECK=1 bash hpc/download_dsec.sh              # report sizes, download nothing
#   JOBS=4 bash hpc/download_dsec.sh               # 4 concurrent sequences
#   KEEP_ZIPS=1 bash hpc/download_dsec.sh          # keep archives after extracting
#
# RUN THIS ON A LOGIN NODE. BlueBEAR compute nodes have no outbound network, which is the same
# reason the imageio FreeImage cache has to be primed up front.
#
# Only the 18 sequences with public ground-truth flow are fetched. The 7 official test
# sequences are excluded: they have no public GT, and the DSEC leaderboard is out of scope.
# Only the LEFT event camera is used by this model, so the right camera is never downloaded.
#
# Resumable and idempotent: partial archives are continued, and any sequence that already has
# an .extracted marker is skipped, so re-running after an interruption is safe.

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"     # repo root

BASE_URL="https://download.ifi.uzh.ch/rpg/DSEC/train"
DATA_ROOT="${DATA_ROOT:-data/Datasets/DSEC}"
ZIP_DIR="${ZIP_DIR:-${DATA_ROOT}/_zips}"
JOBS="${JOBS:-1}"
KEEP_ZIPS="${KEEP_ZIPS:-0}"
CHECK="${CHECK:-0}"

ALL_SEQUENCES=(
    zurich_city_09_a zurich_city_07_a zurich_city_02_c zurich_city_11_b
    thun_00_a        zurich_city_02_d zurich_city_11_c zurich_city_03_a
    zurich_city_10_a zurich_city_05_b zurich_city_08_a zurich_city_01_a
    zurich_city_10_b zurich_city_02_e zurich_city_05_a zurich_city_06_a
    zurich_city_11_a zurich_city_02_a
)

SEQUENCES=("$@")
[ ${#SEQUENCES[@]} -eq 0 ] && SEQUENCES=("${ALL_SEQUENCES[@]}")

command -v unzip >/dev/null || { echo "ERROR: unzip not found."; exit 1; }

# --- one sequence ------------------------------------------------------------------------
# Exported and driven by xargs so JOBS>1 fetches several sequences at once.
fetch_sequence() {
    set -euo pipefail
    seq="$1"

    events_dir="${DATA_ROOT}/train_events/${seq}/events/left"
    flow_dir="${DATA_ROOT}/train_optical_flow/${seq}/flow/forward"
    ts_dst="${DATA_ROOT}/train_optical_flow/${seq}/flow/forward_timestamps.txt"
    marker="${DATA_ROOT}/train_optical_flow/${seq}/.extracted"

    if [ -f "${marker}" ]; then
        echo "[${seq}] already extracted, skipping"
        return 0
    fi

    mkdir -p "${events_dir}" "${flow_dir}" "${ZIP_DIR}"

    ev_zip="${ZIP_DIR}/${seq}_events_left.zip"
    fl_zip="${ZIP_DIR}/${seq}_optical_flow_forward_event.zip"

    # -C - resumes a partial file; --retry rides out transient drops on a long transfer.
    for pair in "${ev_zip}|${seq}_events_left.zip" "${fl_zip}|${seq}_optical_flow_forward_event.zip"; do
        dst="${pair%%|*}"; name="${pair##*|}"
        echo "[${seq}] downloading ${name}"
        curl -fSL --retry 5 --retry-delay 10 -C - -o "${dst}" "${BASE_URL}/${seq}/${name}"
    done

    echo "[${seq}] downloading forward_timestamps.txt"
    curl -fSL --retry 5 --retry-delay 10 \
        -o "${ts_dst}" "${BASE_URL}/${seq}/${seq}_optical_flow_forward_timestamps.txt"

    # Both archives are flat, so they extract straight into their target directories.
    echo "[${seq}] extracting"
    unzip -o -q "${ev_zip}" -d "${events_dir}"
    unzip -o -q "${fl_zip}" -d "${flow_dir}"

    for f in events.h5 rectify_map.h5; do
        [ -f "${events_dir}/${f}" ] || { echo "[${seq}] ERROR: missing ${f}"; return 1; }
    done

    # A PNG/timestamp mismatch silently misaligns ground truth with events, because
    # _create_flow_maps numbers the sorted PNGs sequentially rather than matching timestamps.
    # Catching it here is far cheaper than discovering it as "close but wrong" metrics later.
    n_png=$(find "${flow_dir}" -name '*.png' | wc -l)
    n_ts=$(grep -cv '^\s*\(#.*\)\?$' "${ts_dst}")
    if [ "${n_png}" -ne "${n_ts}" ]; then
        echo "[${seq}] ERROR: ${n_png} flow PNGs vs ${n_ts} timestamp rows"
        return 1
    fi

    touch "${marker}"
    [ "${KEEP_ZIPS}" = "0" ] && rm -f "${ev_zip}" "${fl_zip}"

    echo "[${seq}] OK: ${n_png} flow maps, events.h5 + rectify_map.h5"
}
export -f fetch_sequence
export BASE_URL DATA_ROOT ZIP_DIR KEEP_ZIPS

# --- local state report ------------------------------------------------------------------
# What actually landed on disk, for after an interrupted run. Re-running the script then
# finishes the job: completed sequences are skipped, partial archives resume.
if [ "${VERIFY:-0}" != "0" ]; then
    echo "Verifying ${#SEQUENCES[@]} sequence(s) under ${DATA_ROOT}"
    echo
    done_n=0; part_n=0; miss_n=0
    for seq in "${SEQUENCES[@]}"; do
        events="${DATA_ROOT}/train_events/${seq}/events/left"
        flow="${DATA_ROOT}/train_optical_flow/${seq}/flow"
        marker="${DATA_ROOT}/train_optical_flow/${seq}/.extracted"

        if [ -f "${marker}" ] && [ -f "${events}/events.h5" ] && [ -f "${events}/rectify_map.h5" ]; then
            n_png=$(find "${flow}/forward" -name '*.png' 2>/dev/null | wc -l)
            n_ts=$(grep -cv '^\s*\(#.*\)\?$' "${flow}/forward_timestamps.txt" 2>/dev/null || echo 0)
            if [ "${n_png}" -eq "${n_ts}" ] && [ "${n_png}" -gt 0 ]; then
                printf '  %-20s OK        %4d flow maps\n' "${seq}" "${n_png}"
                done_n=$((done_n+1))
            else
                printf '  %-20s MISMATCH  %d PNGs vs %d timestamps\n' "${seq}" "${n_png}" "${n_ts}"
                part_n=$((part_n+1))
            fi
        elif [ -e "${events}/events.h5" ] || [ -e "${ZIP_DIR}/${seq}_events_left.zip" ]; then
            printf '  %-20s PARTIAL   (re-run to resume)\n' "${seq}"
            part_n=$((part_n+1))
        else
            printf '  %-20s missing\n' "${seq}"
            miss_n=$((miss_n+1))
        fi
    done
    echo
    echo "complete: ${done_n}   partial: ${part_n}   missing: ${miss_n}   (of ${#SEQUENCES[@]})"
    if [ "${done_n}" -eq "${#SEQUENCES[@]}" ]; then
        echo "All sequences present. Next:  sbatch --array=0-17 hpc/preprocess.slurm"
    else
        echo "Finish with:  JOBS=4 bash hpc/download_dsec.sh"
    fi
    exit 0
fi

# --- remote size report -------------------------------------------------------------------
if [ "${CHECK}" != "0" ]; then
    echo "Sequences: ${#SEQUENCES[@]}   target: ${DATA_ROOT}"
    printf '%s\n' "${SEQUENCES[@]}" | while read -r seq; do
        sz=$(curl -sIL --max-time 30 "${BASE_URL}/${seq}/${seq}_events_left.zip" \
             | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
        printf '  %-20s %s bytes\n' "${seq}" "${sz:-?}"
    done
    echo "(CHECK=1: nothing downloaded)"
    exit 0
fi

echo "Downloading ${#SEQUENCES[@]} sequence(s) into ${DATA_ROOT} with JOBS=${JOBS}"
echo "Expect ~38 GB for the full set. Run from a login node."
echo

printf '%s\n' "${SEQUENCES[@]}" | xargs -P "${JOBS}" -I{} bash -c 'fetch_sequence "$@"' _ {}

echo
echo "Done. Layout:"
echo "  ${DATA_ROOT}/train_events/<seq>/events/left/{events.h5,rectify_map.h5}"
echo "  ${DATA_ROOT}/train_optical_flow/<seq>/flow/{forward/*.png,forward_timestamps.txt}"
echo
echo "Next:  sbatch --array=0-17 hpc/preprocess.slurm"
