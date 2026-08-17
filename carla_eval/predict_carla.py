"""Run SDformerFlow or STTFlowNet over a converted CARLA capture; dump predictions and a video.

    python carla_eval/predict_carla.py --config configs/valid_DSEC_supervised_full.yml \
        --runid $(cat hpc/logs/snn_runid.txt) --path_mlflow $SDF_MLFLOW_DIR \
        --tensors data/Datasets/CARLA/50761933/saved_flow_data --id carla_50761933 \
        --out results/carla_eval/pred/sdformerflow --video results/carla_eval/sdformerflow.mp4

Metrics are not computed here; score_flow.py scores the dumped predictions so every model goes
through one metric implementation.

The model is built as eval_DSEC_flow_SNN.py builds it: same config merge from the MLflow run's
logged params, same input transform, same BatchNorm mode, same per-sample state reset. Only the
dataset root and split list differ.

Pass valid_DSEC_supervised_full.yml for the SNN -- it evaluates at 480x640 with the swin
position biases interpolated (remap: v1). STTFlowNet trains at full resolution already and
takes valid_DSEC_ann.yml unchanged.
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mlflow  # noqa: E402

from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite  # noqa: E402
from configs.parser import YAMLParser  # noqa: E402

from carla_eval.flow_model import SwinFlowAdapter  # noqa: E402


def build_config(config_parser, runid, path_mlflow, tensors, resolution=None):
    """Config as the DSEC evaluator resolves it, repointed at the CARLA tensors.

    The MLflow run's logged params come first and the YAML overrides them; that ordering is
    what lets the YAML switch the test resolution.
    """
    mlflow.set_tracking_uri(path_mlflow)
    run = mlflow.get_run(runid)
    config = config_parser.merge_configs(run.data.params)

    config["data"]["path"] = tensors
    if resolution:
        config["loader"]["resolution"] = list(resolution)
    config["loader"]["batch_size"] = 1
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/valid_DSEC_supervised_full.yml")
    ap.add_argument("--runid", required=True)
    ap.add_argument("--path_mlflow", default="")
    ap.add_argument("--tensors", required=True, help="saved_flow_data from carla_to_voxel.py")
    ap.add_argument("--id", required=True, help="capture id / split-list prefix")
    ap.add_argument("--out", required=True, help="directory for dumped (2,H,W) predictions")
    ap.add_argument("--video", default=None, help="render a GT-vs-prediction video here")
    ap.add_argument("--rgb", default=None, help="capture rgb/ dir, for the video's camera panel")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    config_parser = YAMLParser(args.config)
    config = build_config(config_parser, args.runid, args.path_mlflow, args.tensors)
    device = config_parser.device

    if config["loader"].get("crop"):
        # Every model must be scored over the same pixels, so no cropping here.
        raise SystemExit(
            "loader.crop is %s. CARLA evaluation must run at full resolution -- use "
            "valid_DSEC_supervised_full.yml (SNN) or valid_DSEC_ann.yml (ANN)."
            % (config["loader"]["crop"],))

    dataset = DSECDatasetLite(config, file_list=args.id, stereo=False, transform=None,
                              scale_factor=config["test"]["scale_factor"])
    print("%d windows" % len(dataset))

    model = SwinFlowAdapter(config, args.runid, device)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    files = (dataset.files.iloc[:, 1] if config["data"]["num_chunks"] == 2
             else dataset.files.iloc[:, 0]).tolist()

    os.makedirs(args.out, exist_ok=True)
    preds, labels, masks = [], [], []

    n = len(loader) if args.max_samples is None else min(len(loader), args.max_samples)
    for idx, (chunk, mask, label) in enumerate(tqdm(loader, total=n, desc=args.id,
                                                    miniters=max(n // 100, 1))):
        if args.max_samples is not None and idx >= args.max_samples:
            break
        model.reset_state()
        pred = model.forward(chunk)

        arr = pred[0].detach().cpu().numpy().astype(np.float32)
        np.save(os.path.join(args.out, files[idx]), arr)

        if args.video:
            preds.append(arr)
            labels.append(label[0].numpy())
            masks.append(mask[0].numpy())

    print("wrote %d predictions -> %s" % (min(n, len(files)), args.out))

    if args.video:
        from carla_eval.visualise import write_flow_video
        write_flow_video(np.array(labels), np.array(preds), np.array(masks), args.video,
                         rgb_dir=args.rgb, files=files[:len(preds)])
        print("wrote %s" % args.video)


if __name__ == "__main__":
    main()
