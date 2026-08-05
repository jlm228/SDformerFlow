# Load the modules and virtualenv this project needs. Source it, don't execute it:
#
#   source hpc/env.sh
#
# Every new shell (login, srun session, batch job) starts without these, so this runs
# at the top of an interactive session and of each Slurm script.

# Paths are derived from THIS file's own location, so moving the whole project
# (repo + sibling venvs/) needs no edits here. Layout assumed:
#     <parent>/SDformerFlow/      <- the repo (this file is at SDformerFlow/hpc/env.sh)
#     <parent>/venvs/sdformerflow <- the virtualenv
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # <repo>/hpc
REPO_DIR="$(dirname "${_ENV_DIR}")"                        # <repo>
PARENT_DIR="$(dirname "${REPO_DIR}")"                      # <parent> (holds repo + venvs)

# venv sits beside the repo; override with `export SDFORMERFLOW_VENV=/path/to/venv`.
VENV_DIR="${SDFORMERFLOW_VENV:-${PARENT_DIR}/venvs/sdformerflow}"

module purge
module load bluebear
module load bear-apps/2023a/live
module load Python/3.11.3-GCCcore-12.3.0

source "${VENV_DIR}/bin/activate"

# imageio downloads the FreeImage library on first use, which compute nodes cannot do,
# so it must point at a cache populated on the login node. The DSEC ground-truth
# preprocessing reads 16-bit flow PNGs with format='PNG-FI', which needs that plugin --
# and DSEC_dataset_preprocess.py has the download() call commented out. Populate once:
#     source hpc/env.sh && python -c "import imageio; imageio.plugins.freeimage.download()"
export IMAGEIO_USERDIR="${HOME}/.imageio"

# The training scripts call mlflow.set_tracking_uri(args.path_mlflow) with a default of
# "", which resolves relative to the working directory. Jobs pass --path_mlflow "${SDF_MLFLOW_DIR}"
# explicitly so runs land on /rds rather than wherever the job happened to start.
# Artifacts are large (a pickled whole-model checkpoint is ~200 MB), so keep this off /home.
export SDF_MLFLOW_DIR="${SDF_MLFLOW_DIR:-${REPO_DIR}/mlruns}"

# Slurm redirects stdout to a file rather than a tty, which makes Python block-buffer it: the
# per-epoch print()s would then lag behind by kilobytes and appear out of order relative to the
# tqdm bar (tqdm writes to stderr and flushes itself, so it stays live either way). Unbuffering
# keeps `tail -f` on the job log an accurate view of where the run actually is.
export PYTHONUNBUFFERED=1

# spikingjelly JIT-compiles its CUDA kernels through CuPy at runtime (set_backend(..., "cupy")).
# The cupy-cuda12x wheel bundles the toolkit, so no CUDA module is loaded here; if the JIT
# fails on a compute node, that is the first thing to revisit.
