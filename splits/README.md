# DSEC train / validation split

The two CSVs here define the split every result in this reproduction is measured against.

## Why they live in this repo

SDformerFlow's paper states only that it *"adopt[s] a similar data split strategy as in [38],
dividing the training sequences into training and validation sets"*, and its README adds *"We
follow the same data splits as in OF_EV_SNN"* — but **the split files are not in the upstream
repo**. The dataloader simply expects them to already exist at
`data/Datasets/DSEC/saved_flow_data/sequence_lists/`.

They are published by [OF_EV_SNN](https://github.com/J-Cuadrado/OF_EV_SNN) (reference [38]), at
`data/dataset/saved_flow_data/sequence_lists/`. Vendored here — 424 KB of text — so that
reproducing this work does not require cloning an unrelated repository.

## The split

| | sequences | samples |
|---|---|---|
| train | 13 | 6000 |
| validation | 5 | 2152 |

**Validation:** `zurich_city_11_b` (964), `zurich_city_03_a` (439), `zurich_city_02_d` (361),
`zurich_city_08_a` (348), `thun_00_a` (40).

**Train:** `zurich_city_11_c` (818), `02_c` (794), `10_a` (751), `06_a` (641), `09_a` (637),
`05_a` (628), `02_e` (469), `05_b` (392), `10_b` (382), `11_a` (230), `07_a` (99), `01_a` (96),
`02_a` (63).

13 + 5 = the 18 sequences with public ground-truth flow, which is exactly the list
`DSEC_dataset_preprocess.py` processes.

The evaluator pools all samples and divides by the count, so reported metrics are
**sample-weighted**: `zurich_city_11_b` alone accounts for 45% of every validation figure, and
`thun_00_a` for 1.9%.

## Format, and the derived single-chunk lists

Each row of a `*_doubleseq.csv` is `frame1.npy,frame2.npy` — a consecutive pair. `DSECDatasetLite`
selects its split file from `data.num_chunks`:

- `num_chunks: 2` (the **ANN** configs) reads `*_split_doubleseq.csv` — these files.
- `num_chunks: 1` (the **SNN** configs) reads `*_split_seq.csv` — **not published anywhere**.

Derive the single-chunk lists with `DSEC_dataloader/build_seq_split.py`, which takes column 2 of
each doubleseq CSV. Column 2 is the *target* frame, whose ground truth is the label loaded in the
two-chunk case, so this preserves the authors' exact partition rather than inventing a new one.

## Caveat

Because the paper never names its validation sequences, this reproduction rests on OF_EV_SNN's
CSVs being the same split the SDformerFlow authors actually used. The README asserts it; it is not
independently verifiable. Given `zurich_city_11_b` dominates the average, a different assignment
would move the numbers noticeably — so if results land close-but-off, suspect this before
suspecting the training.
