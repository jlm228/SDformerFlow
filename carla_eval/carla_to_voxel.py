"""Convert a CARLA capture into the `saved_flow_data` layout SDformerFlow's loader reads.

    python carla_eval/carla_to_voxel.py <capture_dir> --out data/Datasets/CARLA/<id>/saved_flow_data

Events are voxelised by `VoxelGrid.convert_CHW`, the same call `DSEC_dataset_preprocess.py`
makes, so the representation matches DSEC's by construction. Ground truth comes from
`inspect_capture.labels_for_window`, shared with the OF_EV_SNN converter, so both produce
identical labels and masks and differ only in the event tensor.

Layout written (mirrors DSEC's, so DSECDatasetLite needs no changes):

    <out>/event_tensors/10bins/left/<id>/<id>_0001.npy   float32 (10, 480, 640) signed voxel
    <out>/gt_tensors/<id>_0001.npy                       float32 (2, 480, 640) px per window
    <out>/mask_tensors/<id>_0001.npy                     float32 (480, 640) validity
    <out>/ped_mask_tensors/<id>_0001.npy                 float32 (480, 640) pedestrian
    <out>/sequence_lists/<id>_split_doubleseq.csv        (prev, target) -- ANN, num_chunks 2
    <out>/sequence_lists/<id>_split_seq.csv              (target)       -- SNN, num_chunks 1
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from DSEC_dataloader.build_seq_split import build_seq_split  # noqa: E402
from DSEC_dataloader.event_representations import VoxelGrid  # noqa: E402

DEFAULT_CARLA_SCRIPTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "CARLA-hpc-scripts")


def _import_inspect_capture(path=None):
    """Import inspect_capture from the sibling repo.

    Located by $CARLA_SCRIPTS_ROOT, falling back to ../../CARLA-hpc-scripts.
    """
    root = path or os.environ.get("CARLA_SCRIPTS_ROOT") or DEFAULT_CARLA_SCRIPTS
    root = os.path.abspath(root)
    if not os.path.isfile(os.path.join(root, "inspect_capture.py")):
        raise SystemExit(
            "inspect_capture.py not found under %s.\n"
            "Point $CARLA_SCRIPTS_ROOT at your CARLA-hpc-scripts checkout, or pass "
            "--carla-scripts." % root)
    sys.path.insert(0, root)
    import inspect_capture  # noqa: E402
    return inspect_capture


def events_to_voxel(x, y, t, pol, num_bins, height, width):
    """One window of raw events -> signed voxel grid, as DSEC preprocessing builds it.

    Time is normalised to [0, 1] by the window's own first and last event -- not by the
    nominal window bounds -- and convert_CHW then rescales to [0, num_bins - 1]. Polarity goes
    in as 0/1 and becomes +/-1 inside convert_CHW.
    """
    if len(t) < 2 or t[-1] == t[0]:
        # Cannot normalise: t[-1] - t[0] == 0 would give NaN everywhere.
        return torch.zeros((num_bins, height, width), dtype=torch.float32)

    t_norm = (t - t[0]).astype("float32")
    t_norm = t_norm / t_norm[-1]

    grid = VoxelGrid((num_bins, height, width))
    return grid.convert_CHW({
        "p": torch.from_numpy(np.asarray(pol).astype("float32")),
        "t": torch.from_numpy(t_norm),
        "x": torch.from_numpy(np.asarray(x).astype("float32")),
        "y": torch.from_numpy(np.asarray(y).astype("float32")),
    })


def convert(capture_dir, out_dir, capture_id=None, num_bins=10, flow_y_flip=True,
            gt_offset=1, depth_max=500.0, carla_scripts=None):
    ic = _import_inspect_capture(carla_scripts)

    events, windows, meta = ic.load_capture(capture_dir)
    capture_id = capture_id or os.path.basename(os.path.normpath(capture_dir))
    height, width = meta["height"], meta["width"]
    window_us = int(round(meta["window_s"] * 1e6))

    ev_dir = os.path.join(out_dir, "event_tensors", "%sbins" % str(num_bins).zfill(2),
                          "left", capture_id)
    gt_dir = os.path.join(out_dir, "gt_tensors")
    mask_dir = os.path.join(out_dir, "mask_tensors")
    ped_dir = os.path.join(out_dir, "ped_mask_tensors")
    seq_dir = os.path.join(out_dir, "sequence_lists")
    for d in (ev_dir, gt_dir, mask_dir, ped_dir, seq_dir):
        os.makedirs(d, exist_ok=True)

    # searchsorted rather than a boolean mask per window, which would be a full pass over tens
    # of millions of events each time. It needs sorted input, so sort if the array is not.
    t_all = events["t"]
    if len(t_all) > 1 and not np.all(np.diff(t_all) >= 0):
        order = np.argsort(t_all, kind="stable")
        events, t_all = events[order], t_all[order]
    t_starts = windows["t_start_us"].to_numpy()

    written, empty = [], 0
    for i in range(len(windows)):
        t_start = t_starts[i]
        if t_start != t_start:  # NaN: window recorded zero events
            continue
        t_start = int(t_start)
        t_end = t_start + window_us

        gt = ic.labels_for_window(capture_dir, i, height, width, flow_y_flip=flow_y_flip,
                                  gt_offset=gt_offset, depth_max=depth_max)
        if gt is None:
            continue
        flow_2hw, mask, ped = gt

        lo, hi = np.searchsorted(t_all, (t_start, t_end), side="left")
        ev = events[lo:hi]
        if len(ev) < 2:
            empty += 1

        chunk = events_to_voxel(ev["x"], ev["y"], ev["t"], ev["pol"], num_bins, height, width)

        fname = "%s_%04d.npy" % (capture_id, i + 1)
        np.save(os.path.join(ev_dir, fname), chunk)
        np.save(os.path.join(gt_dir, fname), flow_2hw)
        np.save(os.path.join(mask_dir, fname), mask)
        np.save(os.path.join(ped_dir, fname), ped)
        written.append(fname)

    double_csv = os.path.join(seq_dir, "%s_split_doubleseq.csv" % capture_id)
    with open(double_csv, "w", newline="") as fh:
        csv.writer(fh).writerows([(written[i - 1], written[i])
                                  for i in range(1, len(written))])

    # Derived from column 2 of the doubleseq list by the same helper the DSEC splits use, so
    # the one- and two-chunk models get the identical target set. The first window is excluded
    # because the two-chunk model has no predecessor for it.
    build_seq_split(seq_dir, capture_id, overwrite=True)

    print("wrote %d voxel/gt/mask/ped tensor sets -> %s" % (len(written), out_dir))
    print("  num_bins: %d   flow y-flip: %s   gt_offset: %+d   depth_max: %.0f m"
          % (num_bins, flow_y_flip, gt_offset, depth_max))
    if empty:
        print("  WARNING: %d window(s) had <2 events and were written as zero voxels" % empty)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture_dir")
    ap.add_argument("--out", required=True, help="saved_flow_data directory to write")
    ap.add_argument("--capture-id", default=None)
    ap.add_argument("--num-bins", type=int, default=10,
                    help="must equal data.num_frames in the model config (default 10)")
    ap.add_argument("--no-flow-y-flip", action="store_true",
                    help="do NOT negate the GT flow y-channel")
    ap.add_argument("--gt-offset", type=int, default=1)
    ap.add_argument("--depth-max", type=float, default=500.0)
    ap.add_argument("--carla-scripts", default=None,
                    help="path to the CARLA-hpc-scripts checkout "
                         "(default: $CARLA_SCRIPTS_ROOT, then ../../CARLA-hpc-scripts)")
    args = ap.parse_args()

    convert(args.capture_dir, args.out, capture_id=args.capture_id, num_bins=args.num_bins,
            flow_y_flip=not args.no_flow_y_flip, gt_offset=args.gt_offset,
            depth_max=args.depth_max, carla_scripts=args.carla_scripts)


if __name__ == "__main__":
    main()
