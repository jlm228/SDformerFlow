"""Forward one model over another model's perturbed input tensors.

    python carla_eval/transfer_attack.py --model snn --runid <id> \\
        --capture <capture> --tensors <capture>/saved_flow_data --id carla_<capture> \\
        --adv-tensors results/attack/adv_tensors/ann/div_inflate_pgd \\
        --clean-pred results/carla_eval/pred/snn \\
        --out results/attack/snn --report results/attack/snn/reports

Both swin models read byte-identical voxels, so an ANN-derived perturbation feeds the SNN with
no re-encoding. The SNN's own gradients never enter, so this attacks it without its surrogate.

Output matches an attack run: a seeded prediction dump per budget and a report sweep.py joins
on, labelled `<objective>_transfer` so it forms its own curve.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carla_eval.attack_carla import (CONFIGS, build_capture_loader,  # noqa: E402
                                     import_attack_core)
from carla_eval.predict_carla import build_config                    # noqa: E402
from carla_eval.flow_model import SwinFlowAdapter                    # noqa: E402
from utils.yaml_parser import YAMLParser                             # noqa: E402

WINDOW = re.compile(r"_(\d{4})\.npy$")


def eps_dirs(adv_root):
    """[(epsilon, dir)] under an adv-tensor root written as <label>/eps<value>/."""
    out = []
    for d in sorted(glob.glob(os.path.join(adv_root, "eps*"))):
        if os.path.isdir(d):
            try:
                out.append((float(os.path.basename(d)[3:]), d))
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("snn", "ann"), required=True,
                    help="the model to FORWARD; the perturbations come from the other one")
    ap.add_argument("--runid", required=True)
    ap.add_argument("--path_mlflow", default=os.environ.get("SDF_MLFLOW_DIR", "mlruns"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--tensors", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--adv-tensors", required=True,
                    help="<label>/ directory of the source model's perturbed inputs")
    ap.add_argument("--clean-pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--carla-scripts", default=os.environ.get("CARLA_SCRIPTS_ROOT",
                                                              "../CARLA-hpc-scripts"))
    args = ap.parse_args()

    _core, band_mod, runner = import_attack_core(args.carla_scripts)
    from attack_core.reference import divergence_numpy

    label = "%s_transfer" % os.path.basename(os.path.normpath(args.adv_tensors))
    config_parser = YAMLParser(args.config or CONFIGS[args.model])
    config = build_config(config_parser, args.runid, args.path_mlflow, args.tensors)
    device = config_parser.device
    load_window, _n = build_capture_loader(config, args.id, device)
    model = SwinFlowAdapter(config, args.runid, device)
    ped_dir = os.path.join(args.tensors, "ped_mask_tensors")

    model_name = {"snn": "sdformerflow", "ann": "sttflownet_en4"}[args.model]
    reports = {}
    for eps, d in eps_dirs(args.adv_tensors):
        out_dir = os.path.join(args.out, label, runner.eps_tag(eps))
        n_seeded = band_mod.seed_from_clean(args.clean_pred, out_dir, args.id)
        rep = {"windows": [], "model": model_name, "objective": label, "sign": None,
               "label": label, "attack": "transfer", "epsilon": float(eps), "iters": 0,
               "alpha": None, "seed": None, "rand_init": False, "capture_id": args.id,
               "band_lo": None, "band_hi": None, "support_mode": "all",
               "clip_min": None, "clip_max": None, "pred_dir": out_dir,
               "n_seeded": n_seeded, "source": os.path.abspath(args.adv_tensors)}

        files = sorted(glob.glob(os.path.join(d, "*.npy")))
        print("\n%s eps %g: %d perturbed windows" % (label, eps, len(files)))
        for path in files:
            m = WINDOW.search(os.path.basename(path))
            if m is None:
                continue
            i = int(m.group(1)) - 1
            x = torch.from_numpy(np.load(path)).unsqueeze(0).to(device=device,
                                                                dtype=torch.float32)
            with torch.no_grad():
                model.reset_state()
                flow = model.forward(x)
            adv_np = flow.detach().float().cpu().numpy()[0]
            np.save(os.path.join(out_dir, "%s_%04d.npy" % (args.id, i + 1)), adv_np)

            haz = np.load(os.path.join(ped_dir, os.path.basename(path)))
            clean_np = np.load(os.path.join(args.clean_pred,
                                            "%s_%04d.npy" % (args.id, i + 1)))
            d_adv = divergence_numpy(adv_np, haz > 0)
            d_clean = divergence_numpy(clean_np.astype(np.float32), haz > 0)
            rep["windows"].append({
                "window": i, "div_clean": d_clean, "div_adv": d_adv,
                "div_ratio": (d_adv / d_clean) if d_clean else None,
                "epe_masked_clean": None, "epe_masked_adv": None,
                "epe_global_clean": None, "epe_global_adv": None,
                "history": [], "loss_first": None, "loss_last": None,
                "constraint": {"passed": True, "max_abs_delta": None}})
        reports[eps] = rep

    runner.write_reports(reports, args.report, label)
    print("\nforwarded %s over %s" % (model_name, args.adv_tensors))


if __name__ == "__main__":
    main()
