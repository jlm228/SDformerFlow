"""Clean SDformerFlow (snn) and STTFlowNet (ann) inference over many captures, keeping compact
predictions.

    python carla_eval/infer_batch.py --captures <list.txt> --out <dir> --scratch <dir> \
        [--shard I --shards N] [--models snn ann]
    python carla_eval/infer_batch.py --check <capture_dir> --out <dir> --scratch <dir>

The capture loop, window selection, scoring and compaction are CARLA-hpc-scripts'
flow/batch_infer.py. This file supplies the converter (carla_to_voxel.convert, shared by both
models) and the prediction loop of predict_carla.py: one sample at a time, state reset before
each, with each model built as predict_carla.py builds it. Run ids are read from
hpc/logs/<model>_runid.txt. --check runs flow/batch_infer.check on one capture instead.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite  # noqa: E402
from configs.parser import YAMLParser  # noqa: E402

from carla_eval.carla_to_voxel import convert as carla_to_voxel  # noqa: E402
from carla_eval.flow_model import SwinFlowAdapter  # noqa: E402
from carla_eval.predict_carla import build_config  # noqa: E402

CONFIGS = {"snn": "configs/valid_DSEC_supervised_full.yml",
           "ann": "configs/valid_DSEC_ann.yml"}
DEFAULT_CARLA_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                     "CARLA-hpc-scripts")


def import_batch_infer(path=None):
    """flow.batch_infer from the CARLA-hpc-scripts checkout."""
    root = os.path.abspath(path or os.environ.get("CARLA_SCRIPTS_ROOT") or DEFAULT_CARLA_SCRIPTS)
    if not os.path.isfile(os.path.join(root, "flow", "batch_infer.py")):
        raise SystemExit("flow/batch_infer.py not found under %s; point $CARLA_SCRIPTS_ROOT at "
                         "your CARLA-hpc-scripts checkout." % root)
    sys.path.insert(0, root)
    from flow import batch_infer
    return batch_infer


def convert(capture_dir, tensors_dir, capture_id, windows):
    carla_to_voxel(capture_dir, tensors_dir, capture_id=capture_id, only_windows=windows)


def load_model(name, path_mlflow):
    """(config, adapter) for snn or ann, as predict_carla.py builds them."""
    with open(os.path.join("hpc", "logs", "%s_runid.txt" % name)) as fh:
        runid = fh.read().strip()
    parser = YAMLParser(CONFIGS[name])
    config = build_config(parser, runid, path_mlflow, tensors="")
    if config["loader"].get("crop"):
        raise SystemExit("loader.crop is %s; CARLA evaluation runs at full resolution."
                         % (config["loader"]["crop"],))
    print("%s: runid %s, config %s" % (name, runid, CONFIGS[name]), flush=True)
    return config, SwinFlowAdapter(config, runid, parser.device, verbose=False)


def make_predict(models):
    def predict(name, tensors_dir, capture_id, pred_dir):
        """predict_carla.py's loop, writing (2, H, W) float32 per target."""
        config, model = models[name]
        config["data"]["path"] = tensors_dir
        dataset = DSECDatasetLite(config, file_list=capture_id, stereo=False, transform=None,
                                  scale_factor=config["test"]["scale_factor"])
        files = (dataset.files.iloc[:, 1] if config["data"]["num_chunks"] == 2
                 else dataset.files.iloc[:, 0]).tolist()
        os.makedirs(pred_dir, exist_ok=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                             drop_last=False)
        for idx, (chunk, _mask, _label) in enumerate(loader):
            model.reset_state()
            pred = model.forward(chunk)
            np.save(os.path.join(pred_dir, files[idx]),
                    pred[0].detach().cpu().numpy().astype(np.float32))
    return predict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--captures", help="file listing capture directories, one per line")
    g.add_argument("--check", help="one capture directory to check end to end")
    ap.add_argument("--out", required=True, help="output root (pred/, epe/, done/, failed/)")
    ap.add_argument("--scratch", required=True, help="working space, emptied per capture")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--models", nargs="+", default=["snn", "ann"], choices=sorted(CONFIGS))
    ap.add_argument("--path_mlflow", default=os.environ.get("SDF_MLFLOW_DIR", ""))
    ap.add_argument("--prior", default=os.path.join(
        "..", "OF_EV_SNN", "results", "carla_eval", "dsec_validity_prior.npy"))
    ap.add_argument("--carla-scripts", default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.prior):
        raise SystemExit("not found: %s" % args.prior)
    bi = import_batch_infer(args.carla_scripts)
    models = {m: load_model(m, args.path_mlflow) for m in args.models}
    predict = make_predict(models)

    if args.check:
        ok = bi.check(args.check, args.models, convert, predict, args.out, args.scratch,
                      prior=args.prior)
        raise SystemExit(0 if ok else 1)
    caps = bi.shard(bi.read_capture_list(args.captures), args.shard, args.shards)
    failures = bi.run(caps, args.models, convert, predict, args.out, args.scratch,
                      prior=args.prior)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
