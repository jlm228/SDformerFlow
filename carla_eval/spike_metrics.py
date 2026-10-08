"""Spike and operation counts for SDformerFlow (SNN) or STTFlowNet-en4 (ANN).

    python carla_eval/spike_metrics.py --config configs/valid_DSEC_supervised_full.yml \
        --runid $(cat hpc/logs/snn_runid.txt) --path_mlflow $SDF_MLFLOW_DIR \
        --tensors <capture>/saved_flow_data --id carla_<job> --windows 4 \
        --out results/spike_metrics --name sdformerflow

The metrics themselves live in CARLA-hpc-scripts/snnmetrics, which cannot import spikingjelly;
this file is the adapter that does. Set $CARLA_SCRIPTS_ROOT to that checkout.

The model is built as predict_carla.py builds it, so the network measured is the one that was
scored. Run the ANN too, with valid_DSEC_ann.yml: its counts are what the SNN's are read
against.
"""
import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite       # noqa: E402
from configs.parser import YAMLParser                               # noqa: E402

from carla_eval.flow_model import SwinFlowAdapter                   # noqa: E402
from carla_eval.predict_carla import build_config                   # noqa: E402

CARLA_SCRIPTS_ROOT = os.environ.get("CARLA_SCRIPTS_ROOT", "../CARLA-hpc-scripts")
sys.path.insert(0, os.path.abspath(CARLA_SCRIPTS_ROOT))

from snnmetrics.probe import SpikeProbe                             # noqa: E402
from snnmetrics.cost import (footprint_bytes, connection_sparsity,  # noqa: E402
                             wall_stats, write_csvs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/valid_DSEC_supervised_full.yml")
    ap.add_argument("--runid", required=True)
    ap.add_argument("--path_mlflow", default="")
    ap.add_argument("--tensors", required=True, help="saved_flow_data from carla_to_voxel.py")
    ap.add_argument("--id", required=True, help="capture id / split-list prefix")
    ap.add_argument("--windows", type=int, default=4,
                    help="how many windows to measure; sparsity converges fast, so a few "
                         "suffice for the network properties")
    ap.add_argument("--out", default=os.path.join("results", "spike_metrics"))
    ap.add_argument("--name", default=None, help="label for the output files")
    ap.add_argument("--channel-dim", type=int, default=None,
                    help="override the inferred channel axis for the per-channel counts; the "
                         "swin blocks are token-shaped, so check n_channels in the per-layer "
                         "CSV against the architecture on the first run")
    ap.add_argument("--lenient", action="store_true",
                    help="record rather than refuse a finite-threshold layer whose output is "
                         "not {0, 1} (a graded or gated neuron model)")
    args = ap.parse_args()

    config_parser = YAMLParser(args.config)
    config = build_config(config_parser, args.runid, args.path_mlflow, args.tensors)
    device = config_parser.device

    if config["loader"].get("crop"):
        # Same refusal as predict_carla.py: operation counts scale with the measured pixels,
        # so a cropped run is not comparable with the scored full-resolution one.
        raise SystemExit(
            "loader.crop is %s. Spike metrics must be recorded at the resolution the model "
            "was scored at -- use valid_DSEC_supervised_full.yml (SNN) or valid_DSEC_ann.yml "
            "(ANN)." % (config["loader"]["crop"],))

    dataset = DSECDatasetLite(config, file_list=args.id, stereo=False, transform=None,
                              scale_factor=config["test"]["scale_factor"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    n = min(args.windows, len(loader))
    print("%d windows available, measuring %d" % (len(dataset), n), flush=True)

    model = SwinFlowAdapter(config, args.runid, device)
    name = args.name or ("sdformerflow" if model.spiking else "sttflownet_en4")

    static = {"footprint_bytes": footprint_bytes(model.net),
              "connection_sparsity": connection_sparsity(model.net),
              "spiking": bool(model.spiking),
              "num_steps": config["data"].get("num_frames"),
              "num_chunks": config["data"].get("num_chunks"),
              "resolution": "x".join(str(d) for d in config["loader"]["resolution"]),
              "device": str(device),
              "config": args.config, "runid": args.runid, "capture_id": args.id}
    print("footprint: %.2f MB | connection sparsity: %.4g | spiking: %s | T: %s"
          % (static["footprint_bytes"] / 1e6, static["connection_sparsity"],
             static["spiking"], static["num_steps"]), flush=True)

    wall = []
    with SpikeProbe(model.net, strict_binary=not args.lenient,
                    channel_dim=args.channel_dim) as probe:
        for idx, batch in enumerate(tqdm(loader, total=n, desc=name)):
            if idx >= n:
                break
            chunk = batch[0]
            # Per-window reset: without it the membranestate carries over and the measured 
            # firing rate drifts across windows.
            model.reset_state()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.forward(chunk)
            if device.type == "cuda":
                torch.cuda.synchronize()
            wall.append(time.perf_counter() - t0)
            probe.mark_window()

    static.update(wall_stats(wall))

    if probe.nonbinary:
        print("\nWARNING: %d finite-threshold layer(s) emitted values outside {0, 1}; their "
              "'firing rate' is a non-zero fraction, not a spike rate:" % len(probe.nonbinary))
        for layer, (lo, hi) in sorted(probe.nonbinary.items()):
            print("    %-48s [%g, %g]" % (layer, lo, hi))

    os.makedirs(args.out, exist_ok=True)
    records_path = probe.dump(os.path.join(args.out, "%s_spikes.json" % name), meta=static)

    overall = write_csvs(name, json.load(open(records_path)), args.out, extra=static)
    print("\n=== %s ===" % name)
    for k, v in overall.items():
        print("  %-28s %s" % (k, "%.6g" % v if isinstance(v, float) else v))
    print("\nrecords: %s" % records_path)


if __name__ == "__main__":
    main()
