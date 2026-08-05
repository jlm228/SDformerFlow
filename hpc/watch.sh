#!/bin/bash
# Follow a running job from the terminal.
#
#   bash hpc/watch.sh                # newest log in hpc/logs/, follow it
#   bash hpc/watch.sh 49123456       # a specific job id
#   bash hpc/watch.sh queue          # just show the queue and exit
#
# Slurm writes job output to the shared filesystem as it goes, so tailing the log is the
# normal way to watch progress -- there is no need to ssh to the compute node. The training
# scripts print a tqdm bar per epoch plus the epoch loss, and tqdm's carriage returns render
# as very long lines in a file, so the bar is easier to read with `less -R` after the fact.

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

LOGDIR="hpc/logs"

show_queue() {
    echo "=== queue ==="
    squeue --me -o "%.10i %.14j %.8T %.10M %.10l %.6D %R" || true
    echo
}

if [ "${1:-}" = "queue" ]; then
    show_queue
    exit 0
fi

if [ -n "${1:-}" ]; then
    LOG=$(ls -1 "${LOGDIR}"/*"${1}"*.out 2>/dev/null | head -1 || true)
    [ -n "${LOG}" ] || { echo "No log in ${LOGDIR} matching '${1}'"; exit 1; }
else
    LOG=$(ls -1t "${LOGDIR}"/*.out 2>/dev/null | head -1 || true)
    [ -n "${LOG}" ] || { echo "No logs in ${LOGDIR} yet."; exit 1; }
fi

show_queue
echo "=== following ${LOG} ==="
echo "(Ctrl-C stops watching; it does NOT stop the job -- use scancel for that)"
echo

# --retry so this works when started before the file exists, e.g. while still queued.
tail -n 40 -F "${LOG}"
