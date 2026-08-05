# `hpc/` — BlueBEAR job scripts

Reproducing the paper's DSEC validation results for a capacity-matched ANN/SNN pair.
Environment setup lives in [`env.sh`](env.sh) (modules + venv + imageio cache + MLflow dir);
every job sources it. Logs go to `hpc/logs/`.

## What we are reproducing

Both models train on the same DSEC train split (13 sequences, 6000 samples) and are scored on
the same validation split (5 sequences, 2152 samples — `zurich_city_11_b` 964, `03_a` 439,
`02_d` 361, `08_a` 348, `thun_00_a` 40). The evaluator pools all samples and divides by count,
so every figure is a **sample-weighted** mean in which `zurich_city_11_b` alone is 45%.

| Model | Config | Eval config | Target EPE / Out% / AEE |
|---|---|---|---|
| `STTFlowNet-en4-b2-p4-w10` | `train_DSEC_supervised_STT_voxel_en4.yml` | `valid_DSEC_ann.yml` | **0.81 / 2.50 / 4.33** (Table 3 best) |
| `SDformerFlow-SPE-QK-s10-c2` | `train_DSEC_supervised_SDformerFlow_en4.yml` | `valid_DSEC_supervised.yml` | **0.93 / 3.17 / 6.37** (Table 4 best, cropped) |
| *same trained SNN* | — | `valid_DSEC_supervised_full.yml` | **1.61 / 8.91 / 7.23** (Table 4 best, full) |

Both are **en4**, 57.51 M vs 54.92 M params — capacity-matched, which is what makes the
comparison fair. Compare the ANN (0.81) against the SNN's **full-resolution** figure (1.61), not
the cropped one: test cropping alone moves SNN EPE by ~73%.

> **C and F are one training run evaluated twice**, not two models. Train the SNN once.

The repo's own `train_DSEC_supervised_STT_voxel.yml` (en3-b2-p4-w9 at 288×384) matches **no**
Table 3 row. It stays selectable via `CONFIG=` but is not the target.

## One-time setup (login node)

```bash
python -m venv ../venvs/sdformerflow          # beside the repo, per env.sh
source hpc/env.sh
pip install -r requirements.txt               # version notes below
python -c "import imageio; imageio.plugins.freeimage.download()"   # compute nodes have no network
```

**Version notes.** `requirements.txt` is internally inconsistent — the upstream README claims
Python 3.7.3 + CUDA 11.8, but it pins `cupy-cuda12x`, which needs Python ≥3.8 and CUDA 12.x. Use
the Python 3.11.3 module from `env.sh`, and:

- **Pin `torch<2.6`.** From 2.6 `torch.load` defaults to `weights_only=True`, which breaks
  `utils.load_model` — it loads a pickled whole `nn.Module`, not a state_dict.
- **Pin `mlflow<2.0`**, or verify the artifact layout; `utils.py` hardcodes the 1.x path
  `model/data/model.pth`.
- Keep `spikingjelly==0.0.0.0.14` and `timm==0.6.13` exactly. Record `pip freeze`.

Then install the split CSVs. They are vendored in [`splits/`](../splits/) (see that README for
provenance — upstream does not ship them despite depending on them):

```bash
bash hpc/setup_splits.sh      # runs from anywhere; prints 6000/2152 for all four lists
```

The ANN (`num_chunks: 2`) reads the `*_doubleseq.csv` files directly. The SNN (`num_chunks: 1`)
reads `*_split_seq.csv`, which is published nowhere; `build_seq_split.py` derives it by taking
column 2 of each doubleseq CSV (the target frame, whose ground truth is the label), preserving
the authors' exact partition rather than inventing one.

## 0. Download DSEC (login node, ~15–60 min)

```bash
CHECK=1 bash hpc/download_dsec.sh       # report sizes, download nothing
bash hpc/download_dsec.sh thun_00_a     # smallest sequence first (~0.3 GB)
JOBS=4 bash hpc/download_dsec.sh        # all 18 sequences, 38.1 GB
```

Resumable and idempotent — a sequence with an `.extracted` marker is skipped, so re-running
after an interruption is safe. Validates flow-PNG count against timestamp rows, because a
mismatch silently misaligns ground truth with events.

## 1. Preprocess (CPU array, ~1–3 h wall)

```bash
sbatch --array=4    hpc/preprocess.slurm     # thun_00_a only, smoke test
sbatch --array=0-17 hpc/preprocess.slurm     # all 18 sequences in parallel
```

Produces `event_tensors/10bins/left/<seq>/`, `gt_tensors/` and `mask_tensors/`. Budget ~120 GB
on top of the raw downloads. Only the 18 sequences with public ground truth are processed; the
7 official test sequences are excluded (no public GT, leaderboard out of scope).

## 2. Train the ANN (GPU, ~20–35 h)

```bash
SMOKE=1 bash hpc/submit_train.sh ann    # 1 segment, 1 epoch, end-to-end check
bash hpc/submit_train.sh ann            # 1 segment (fits the 2-day cap)
```

Trains at **full 480×640** — that is what the Table 3 row reports, ~2.8× the pixels of the
288×384 crop.

**Cheapest sanity check:** the script prints `Total parameters` at startup. Expect **~57.5 M**;
~20.3 M means you are still on the en3 architecture — stop immediately.

## 3. Evaluate the ANN

```bash
CONFIG=configs/valid_DSEC_ann.yml RUNID=$(cat hpc/logs/ann_runid.txt) sbatch hpc/evaluate.slurm
```

No crop and no remap: this model trains and tests at 480×640, so its swin position-bias tables
are already correctly sized.

## 4. Train the SNN (GPU, ~1.5–2.5 days)

```bash
SMOKE=1 bash hpc/submit_train.sh snn    # 1 segment, 1 epoch
bash hpc/submit_train.sh snn            # 2 chained segments
```

Expect `Total parameters` ~**54.9 M**. `train_snn.slurm` also JIT-compiles a spikingjelly CuPy
kernel before training starts — that failing after days of queueing is the worst way to find out.

Single GPU deliberately: this script's multi-GPU branch calls `model.module.init_weights()`
without ever wrapping in `DataParallel`, so it is broken.

## 5. Evaluate the SNN — both resolutions, one model

```bash
CONFIG=configs/valid_DSEC_supervised.yml      RUNID=$(cat hpc/logs/snn_runid.txt) sbatch hpc/evaluate.slurm  # C
CONFIG=configs/valid_DSEC_supervised_full.yml RUNID=$(cat hpc/logs/snn_runid.txt) sbatch hpc/evaluate.slurm  # F
```

Run both — two published numbers from one trained model at no extra training cost. If only one
matches, the fault is localised to resolution handling rather than to training.

The SNN trains at 288×384, so testing at full 480×640 needs the swin relative-position-bias
tables interpolated to a larger window: 480/288 = 1.667 and 9 × 1.667 = 15, hence
`window_size: [2,15,15]` with `pretrained_window_size: [2,9,9]` and `remap: "v1"` — the authors'
own values, shipped commented out in `valid_DSEC_supervised.yml`. Without the remap you still
get numbers, from position biases sized for the wrong window.

## Checkpointing and resume

Upstream saved **only when training loss improved**, with no periodic save, and reset `best_loss`
to `1e6` on every restart — so a resume near the end silently overwrote the best model with a
worse one, and a plateau could lose many epochs to the walltime. Now:

| artifact | written | read by |
|---|---|---|
| `model/`, `training_state_dict/` | on best-loss improvement | evaluation |
| `model_latest/`, `training_state_dict_latest/` | **every epoch** | `--resume` |

`best_loss` is persisted in the state dict and restored on resume, and `--resume` reads
`model_latest` so weights and optimiser state come from the same epoch. **Worst case you lose one
epoch** (~12–20 min ANN, ~35–60 min SNN); the ~200 MB write is negligible against that.

Chaining is what makes the walltime cap irrelevant. Each segment is a fresh MLflow run resuming
the previous one's `model_latest`; the run id is handed along through `hpc/logs/{ann,snn}_runid.txt`
(written by `--runid_file`, read by the next segment when `CHAIN=1`), joined by
`--dependency=afterok`. Over-requesting segments is harmless: once training reaches `n_epochs`
the extras exit immediately.

Starting a fresh chain while a `*_runid.txt` exists is refused — a stale id would silently resume
the wrong run. Either pass `PREV_RUNID=...` to continue, or delete the file.

Scripts request `--time=2-0:0:0`, this QoS's hard cap (`sacctmgr show qos bbgpu` → `MaxWall
2-00:00:00`). At that ceiling the ANN (~20–35 h) fits a single segment and the SNN (~1.5–2.5
days) needs at most two, which is what `submit_train.sh` defaults to. Shorter requests can
backfill sooner on a busy queue at the cost of more segments — `WALLTIME=12:0:0 bash
hpc/submit_train.sh snn`.

## Monitoring

```bash
bash hpc/watch.sh          # newest log, live tail + queue
bash hpc/watch.sh 49123456 # a specific job
bash hpc/watch.sh queue    # queue only
```

Slurm streams output to the shared filesystem, so tailing the log is the normal way to follow a
run — no need to ssh to the compute node. Ctrl-C stops watching, not the job. `tqdm` gives one
progress bar per epoch (750 iterations for the ANN, 6000 for the SNN); `env.sh` sets
`PYTHONUNBUFFERED=1` so the per-epoch `print`s do not lag behind it.

## Notes and gotchas

- **MLflow is the checkpoint store.** `load_model` resolves weights by run id, not path, so
  `SDF_MLFLOW_DIR` must stay put — moving it orphans every run id. Jobs pass it explicitly
  because the script default (`""`) resolves relative to the working directory.
- `load_model` now **raises** on a run id that does not resolve. It previously fell through and
  evaluated a randomly-initialised network while printing plausible metrics. An empty
  `--prev_runid` is still the normal train-from-scratch path.
- `eval_DSEC_flow_SNN.py` is SNN-named but is the DSEC evaluator for **both** models: it imports
  the ANN classes, and its spikingjelly setup is skipped when `model.spiking_neuron` is null,
  which is how the ANN configs are written. That path previously raised `TypeError`.
- The upstream README's SNN training command names a config that does not exist
  (`train_DSEC_supervised_MS_Spikingformer4.yml`); the real one is
  `train_DSEC_supervised_SDformerFlow_en4.yml`.
- Exact numeric reproduction is not achievable: `parser.py` seeded only `torch.manual_seed(0)`
  upstream. CUDA/numpy/`random` seeding and `cudnn.deterministic` are now added
  (`loader.deterministic: true` to opt in), but some spikingjelly/CuPy kernels have no
  deterministic implementation.
