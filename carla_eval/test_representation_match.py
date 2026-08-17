"""Representation checks for SDformerFlow and STTFlowNet.

    python carla_eval/test_representation_match.py --carla-tensors <saved_flow_data> --id <id>
    python carla_eval/test_representation_match.py --dsec-root data/Datasets/DSEC/saved_flow_data \
        --dsec-raw data/Datasets/DSEC --dsec-seq thun_00_a

Three checks; each runs only if the data it needs is present, and the exit code is non-zero if
any check that ran failed.

  parity    events_to_voxel, fed raw DSEC events, reproduces the precomputed DSEC voxel tensors
            bit for bit. Needs raw DSEC (events.h5 + rectify_map.h5) and its preprocessed
            tensors.

  matched   the one-chunk model (10 bins) and the two-chunk model (20 bins) get byte-identical
            tensors for the same window: the second chunk of the two-chunk input must equal the
            one-chunk input, with matching labels and masks. Needs only the CARLA tensors.

  shape     CARLA voxels match DSEC's in shape, dtype and value range. Needs both sets.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite  # noqa: E402
from carla_eval.carla_to_voxel import events_to_voxel  # noqa: E402

NUM_BINS = 10
HEIGHT, WIDTH = 480, 640


def _config(path, num_chunks, num_frames=NUM_BINS):
    """The minimum DSECDatasetLite reads, built inline so a config edit cannot change it."""
    return {
        "data": {"path": path, "preprocessed": True, "num_frames": num_frames,
                 "num_chunks": num_chunks},
        "model": {"encoding": "voxel"},
        "loader": {"resolution": [HEIGHT, WIDTH], "polarity": True},
    }


# --------------------------------------------------------------------------------------------


def check_parity(dsec_root, dsec_raw, sequence, window_idxs):
    """events_to_voxel on raw DSEC events == the tensors DSEC preprocessing wrote."""
    import h5py
    from DSEC_dataloader.event_representations import EventSlicer, rectify_events

    events_dir = os.path.join(dsec_raw, "train_events", sequence, "events", "left")
    ts_path = os.path.join(dsec_raw, "train_optical_flow", sequence, "flow",
                           "forward_timestamps.txt")
    for p in (events_dir, ts_path):
        if not os.path.exists(p):
            print("  SKIP parity: %s not found" % p)
            return None

    timestamps = np.loadtxt(ts_path, delimiter=",", dtype="int64")
    if timestamps.ndim == 2 and timestamps.shape[1] > 2:
        timestamps = timestamps[:, :2]

    with h5py.File(os.path.join(events_dir, "rectify_map.h5"), "r") as f:
        rectmap = f["rectify_map"][()]

    ok = True
    with h5py.File(os.path.join(events_dir, "events.h5"), "r") as f:
        slicer = EventSlicer(f)
        for idx in window_idxs:
            t_beg, t_end = int(timestamps[idx - 1][0]), int(timestamps[idx - 1][1])
            ev = slicer.get_events(t_beg, t_end)

            xy = rectify_events(ev["x"], ev["y"], rectmap)
            got = events_to_voxel(xy[:, 0], xy[:, 1], ev["t"], ev["p"],
                                  NUM_BINS, HEIGHT, WIDTH).numpy()

            expected_path = os.path.join(
                dsec_root, "event_tensors", "%dbins" % NUM_BINS, "left", sequence,
                "%s_%04d.npy" % (sequence, idx))
            if not os.path.exists(expected_path):
                print("  SKIP parity window %d: %s not found" % (idx, expected_path))
                continue
            expected = np.load(expected_path)

            same = got.shape == expected.shape and np.array_equal(got, expected)
            print("    window %4d: n_events=%-9d max|diff|=%.3e -> %s"
                  % (idx, len(ev["t"]),
                     float(np.abs(got - expected).max()) if got.shape == expected.shape
                     else float("nan"),
                     "PASS" if same else "FAIL"))
            ok &= same
    return ok


def check_matched(carla_tensors, capture_id):
    """The one-chunk input == the second chunk of the two-chunk input, byte for byte."""
    snn = DSECDatasetLite(_config(carla_tensors, num_chunks=1), file_list=capture_id,
                          stereo=False, transform=None)
    ann = DSECDatasetLite(_config(carla_tensors, num_chunks=2), file_list=capture_id,
                          stereo=False, transform=None)

    if len(snn) != len(ann):
        print("    FAIL: %d one-chunk samples vs %d two-chunk -- the split lists disagree on "
              "the target set" % (len(snn), len(ann)))
        return False

    ok = True
    for i in range(len(snn)):
        snn_chunk, snn_mask, snn_label = snn[i]
        ann_chunk, ann_mask, ann_label = ann[i]

        if ann_chunk.shape[0] != 2 * NUM_BINS or snn_chunk.shape[0] != NUM_BINS:
            print("    FAIL sample %d: bins ANN=%d SNN=%d, expected %d and %d"
                  % (i, ann_chunk.shape[0], snn_chunk.shape[0], 2 * NUM_BINS, NUM_BINS))
            return False

        # The two-chunk input is (previous, target); the target half must match.
        same_input = torch.equal(ann_chunk[NUM_BINS:], snn_chunk)
        same_label = torch.equal(ann_label, snn_label)
        same_mask = torch.equal(ann_mask, snn_mask)
        if not (same_input and same_label and same_mask):
            print("    FAIL sample %d: input=%s label=%s mask=%s"
                  % (i, same_input, same_label, same_mask))
            ok = False

    if ok:
        print("    %d/%d samples: ANN[%d:] == SNN, and labels and masks identical -> PASS"
              % (len(snn), len(snn), NUM_BINS))
    return ok


def check_shape(carla_tensors, capture_id, dsec_root, sequence):
    """Shape, dtype and value range of CARLA voxels against DSEC's."""
    carla_files = sorted(glob.glob(os.path.join(
        carla_tensors, "event_tensors", "%dbins" % NUM_BINS, "left", capture_id, "*.npy")))
    dsec_files = sorted(glob.glob(os.path.join(
        dsec_root, "event_tensors", "%dbins" % NUM_BINS, "left", sequence, "*.npy")))
    if not carla_files or not dsec_files:
        print("  SKIP shape: need both CARLA and DSEC voxel tensors")
        return None

    def summarise(files, n=20):
        arrs = [np.load(f) for f in files[:n]]
        return (arrs[0].shape, arrs[0].dtype,
                float(min(a.min() for a in arrs)), float(max(a.max() for a in arrs)),
                float(np.mean([np.abs(a).mean() for a in arrs])))

    c_shape, c_dtype, c_lo, c_hi, c_absmean = summarise(carla_files)
    d_shape, d_dtype, d_lo, d_hi, d_absmean = summarise(dsec_files)

    print("    CARLA %-8s shape=%s dtype=%s range=[%.1f, %.1f] mean|v|=%.4f"
          % (capture_id[:8], c_shape, c_dtype, c_lo, c_hi, c_absmean))
    print("    DSEC  %-8s shape=%s dtype=%s range=[%.1f, %.1f] mean|v|=%.4f"
          % (sequence[:8], d_shape, d_dtype, d_lo, d_hi, d_absmean))

    ok = c_shape == d_shape and c_dtype == d_dtype
    print("    shape/dtype match -> %s" % ("PASS" if ok else "FAIL"))
    # Range is reported, not asserted: a signed event count legitimately differs in scale
    # between a simulator and a real sequence.
    ratio = c_absmean / d_absmean if d_absmean else float("inf")
    print("    mean|v| CARLA/DSEC = %.2fx (reported, not asserted)" % ratio)
    return ok


# --------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carla-tensors", default=None, help="saved_flow_data from carla_to_voxel")
    ap.add_argument("--id", default=None, help="capture id / sequence prefix of those tensors")
    ap.add_argument("--dsec-root", default="data/Datasets/DSEC/saved_flow_data")
    ap.add_argument("--dsec-raw", default="data/Datasets/DSEC")
    ap.add_argument("--dsec-seq", default="thun_00_a")
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 2, 20])
    args = ap.parse_args()

    results = {}

    print("=== parity: events_to_voxel vs DSEC preprocessing ===")
    if os.path.isdir(args.dsec_raw):
        results["parity"] = check_parity(args.dsec_root, args.dsec_raw, args.dsec_seq,
                                         args.windows)
    else:
        print("  SKIP: --dsec-raw %s not found" % args.dsec_raw)

    print("\n=== matched: SDformerFlow vs STTFlowNet input equality ===")
    if args.carla_tensors and args.id:
        results["matched"] = check_matched(args.carla_tensors, args.id)
    else:
        print("  SKIP: pass --carla-tensors and --id")

    print("\n=== shape: CARLA voxels vs DSEC voxels ===")
    if args.carla_tensors and args.id:
        results["shape"] = check_shape(args.carla_tensors, args.id, args.dsec_root,
                                       args.dsec_seq)
    else:
        print("  SKIP: pass --carla-tensors and --id")

    ran = {k: v for k, v in results.items() if v is not None}
    print("\n" + "=" * 60)
    if not ran:
        print("No check had the data it needs. Nothing was verified.")
        raise SystemExit(2)
    for k, v in ran.items():
        print("%-10s %s" % (k, "PASS" if v else "FAIL"))
    skipped = [k for k in ("parity", "matched", "shape") if k not in ran]
    if skipped:
        print("skipped:   %s" % ", ".join(skipped))
    raise SystemExit(0 if all(ran.values()) else 1)


if __name__ == "__main__":
    main()
