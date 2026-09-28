"""Adversarially perturb SDformerFlow or STTFlowNet over a CARLA capture; dump the flow.

    python carla_eval/attack_carla.py --model snn --config configs/valid_DSEC_supervised_full.yml \
        --runid $(cat hpc/logs/snn_runid.txt) --path_mlflow $SDF_MLFLOW_DIR \
        --tensors <capture>/saved_flow_data --capture <capture> --id carla_<capture> \
        --objective div --sign suppress --epsilons 0.0 0.05 0.1 \
        --clean-pred results/carla_eval/pred/snn --out results/attack/snn

The objective and the optimisation loop live in CARLA-hpc-scripts/attack_core, shared with
OF_EV_SNN, so "the same attack across three models" is true by construction. This file is only
the model-specific half.

The model is built exactly as predict_carla.py builds it, so the attacked run perturbs the
network the clean run evaluated, not a differently-configured one.

Epsilon is applied to the RAW SIGNED VOXEL, before prepare_chunk. That is the representation
SDformerFlow's SNN and STTFlowNet consume BYTE-IDENTICALLY, so an epsilon here is
genuinely matched across the neuron-model ablation.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite            # noqa: E402
from configs.parser import YAMLParser                                    # noqa: E402
from utils.input_prep import forward_model                               # noqa: E402

from carla_eval.flow_model import SwinFlowAdapter                        # noqa: E402
from carla_eval.predict_carla import build_config                        # noqa: E402

DEFAULT_CARLA_SCRIPTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "CARLA-hpc-scripts")

CONFIGS = {"snn": "configs/valid_DSEC_supervised_full.yml",
           "ann": "configs/valid_DSEC_ann.yml"}
MODEL_NAMES = {"snn": "sdformerflow", "ann": "sttflownet_en4"}


def import_attack_core(path=None):
    """Import attack_core from the CARLA-hpc-scripts checkout, as carla_to_voxel.py does."""
    root = os.path.abspath(path or os.environ.get("CARLA_SCRIPTS_ROOT")
                           or DEFAULT_CARLA_SCRIPTS)
    if not os.path.isdir(os.path.join(root, "attack_core")):
        raise SystemExit(
            "attack_core not found under %s.\n"
            "Point $CARLA_SCRIPTS_ROOT at your CARLA-hpc-scripts checkout, or pass "
            "--carla-scripts." % root)
    sys.path.insert(0, root)
    import attack_core                                                   # noqa: E402
    from attack_core import band as band_mod, runner                     # noqa: E402
    from attack_core import preflight, surrogates                     # noqa: E402
    return attack_core, band_mod, runner, preflight, surrogates


def mod_loss_function(pred, label, mask):
    """Masked endpoint error.

    Duplicated from OF_EV_SNN/eval/vector_loss_functions.py rather than imported: the two
    repos cannot share an interpreter (incompatible spikingjelly versions), which is the same
    reason score_flow.py scores every model from dumped predictions. Identical by inspection
    and pinned by attack_core's own numpy reference.
    """
    n_pixels = torch.sum(mask)
    err = torch.sqrt((pred[:, 0] - label[:, 0]) ** 2 + (pred[:, 1] - label[:, 1]) ** 2)
    return torch.sum(err * mask) / n_pixels


def build_capture_loader(config, capture_id, device):
    """(load_window, n_windows) over a converted capture.

    Reads exactly what predict_carla.py reads, so the attacked run walks the windows the clean
    run walked. `ped_mask_tensors` carries the hazard mask, written by carla_to_voxel.py from
    inspect_capture.labels_for_window -- the same decode both repos share, so no cross-repo
    bridge is needed at attack time.
    """
    dataset = DSECDatasetLite(config, file_list=capture_id, stereo=False, transform=None,
                              scale_factor=config["test"]["scale_factor"])
    files = (dataset.files.iloc[:, 1] if config["data"]["num_chunks"] == 2
             else dataset.files.iloc[:, 0]).tolist()
    ped_dir = os.path.join(config["data"]["path"], "ped_mask_tensors")

    # Tensor filenames are 1-based over windows.csv rows, which are 0-based.
    by_window = {int(re.search(r"_(\d{4})\.npy$", f).group(1)) - 1: k
                 for k, f in enumerate(files)}

    def load_window(i):
        k = by_window.get(i)
        if k is None:
            return None
        chunk, valid, label = dataset[k]
        x = torch.as_tensor(chunk).unsqueeze(0).to(device=device, dtype=torch.float32)
        gt = torch.as_tensor(label).unsqueeze(0).to(device=device, dtype=torch.float32)
        valid = torch.as_tensor(valid).unsqueeze(0).unsqueeze(0).to(
            device=device, dtype=torch.float32)
        haz = torch.from_numpy(np.load(os.path.join(ped_dir, files[k]))).unsqueeze(0).unsqueeze(
            0).to(device=device, dtype=torch.float32)
        return x, gt, valid, haz

    return load_window, len(files)


def _surrogate_kwargs(args):
    """Constructor arguments for the chosen surrogate."""
    if args.surrogate in ("assg", "assgs"):
        if args.assg_A is None:
            raise SystemExit(
                "--surrogate %s needs --assg-A. It is tuned per model and then frozen, so "
                "there is no default." % args.surrogate)
        return {"A": args.assg_A, "gamma": args.assg_gamma,
                "beta1": args.assg_betas[0], "beta2": args.assg_betas[1]}
    if args.surrogate == "pdsg":
        return {"mode": args.pdsg_mode, "channel_dim": args.pdsg_channel_dim}
    return {}


def set_torch_backend(net):
    """Force the torch backend on every spikingjelly module that offers one.

    cupy kernels carry their own surrogate, so a swapped `surrogate_function` would be ignored
    in the backward while the forward still worked -- a silent no-op.
    """
    changed = 0
    for module in net.modules():
        if getattr(module, "backend", None) not in (None, "torch"):
            module.backend = "torch"
            changed += 1
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["snn", "ann"])
    ap.add_argument("--config", default=None, help="default: the config for --model")
    ap.add_argument("--runid", required=True, help="MLflow run id; hpc/logs/*_runid.txt")
    ap.add_argument("--path_mlflow", default="")
    ap.add_argument("--tensors", required=True, help="saved_flow_data from carla_to_voxel.py")
    ap.add_argument("--capture", required=True, help="the raw capture dir, for the band JSON")
    ap.add_argument("--id", required=True, help="capture id / split-list prefix")
    ap.add_argument("--objective", required=True,
                    choices=["random_sign", "epe_global", "epe_masked", "div"])
    ap.add_argument("--sign", default="suppress", choices=["suppress", "inflate", "none"],
                    help="div only: suppress reads tau LONG, inflate reads it SHORT. 'none' is "
                         "the placeholder the sweep manifest carries for objectives that have "
                         "no direction, and is ignored unless --objective is div")
    ap.add_argument("--attack", default="pgd", choices=["fgsm", "pgd", "sapgd"])
    ap.add_argument("--surrogate", default="native",
                    choices=["native", "pdsg", "assg", "assgs"],
                    help="gradient substitute during the attack. assg is the Atan base, this "
                         "SNN's own family (native ATan, alpha=2). Ignored for --model ann, "
                         "which has no spiking neurons")
    ap.add_argument("--assg-A", type=float, default=None,
                    help="ASSG sharpness setting; tuned per model and then frozen, so it has "
                         "no default. Required for assg/assgs")
    ap.add_argument("--assg-gamma", type=float, default=1.5)
    ap.add_argument("--assg-betas", type=float, nargs=2, default=(0.9, 0.9),
                    metavar=("BETA1", "BETA2"))
    ap.add_argument("--pdsg-mode", default="channel", choices=["channel", "layer"])
    ap.add_argument("--pdsg-channel-dim", type=int, default=1)
    ap.add_argument("--norm-set", default="floating", choices=["floating", "pinned"],
                    help="floating re-normalises over the perturbed input on BOTH paths, so "
                         "the function attacked is the one scored. pinned keeps the clean "
                         "normalisation set, which is what the PGD dumps already on disk used")
    ap.add_argument("--rhos", type=float, nargs="+", default=None,
                    help="the rho each epsilon was calibrated from, recorded in the reports")
    ap.add_argument("--scene-mass", type=float, default=None,
                    help="events per window, so realised rho can be reported")
    ap.add_argument("--preflight", action="store_true",
                    help="measure swap coverage, mean |u| per spiking layer, timing and peak "
                         "memory on one window, then exit without attacking")
    ap.add_argument("--epsilons", type=float, nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=None, help="default: epsilon / 4")
    ap.add_argument("--no-rand-init", action="store_true",
                    help="start PGD at the clean input, not a random point in the ball")
    ap.add_argument("--seed", type=int, default=2305)
    ap.add_argument("--support", default="all", choices=["all", "nonzero"],
                    help="nonzero restricts the perturbation to cells that already carry "
                         "events, which is also more event-consistent")
    ap.add_argument("--band-lo", type=int, default=None)
    ap.add_argument("--band-hi", type=int, default=None)
    ap.add_argument("--band-json", default=None,
                    help="default: <capture>/attack_band.json, from attack_core.band")
    ap.add_argument("--clean-pred", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None, help="default: <out>/reports")
    ap.add_argument("--dump-adv-tensors", default=None,
                    help="also write the perturbed INPUT tensors. The SNN and ANN consume "
                         "byte-identical voxels, so these transfer cleanly between them -- "
                         "that is Stage 6's transfer check")
    ap.add_argument("--carla-scripts", default=None)
    ap.add_argument("--round-trip", default=None, metavar="REPORT_JSON")
    args = ap.parse_args()

    # --preflight only loads the model and measures it; it writes no dumps, so it should not
    # demand the paths a real run needs.
    if not args.preflight:
        for flag, value in (("--clean-pred", args.clean_pred), ("--out", args.out)):
            if value is None:
                ap.error("%s is required unless --preflight" % flag)

    # The manifest carries "none" for objectives with no direction; the objective builder
    # only accepts a real sign, and ignores it for everything but div.
    if args.sign == "none":
        args.sign = "suppress"

    _core, band_mod, runner, preflight, surrogates = import_attack_core(args.carla_scripts)

    if args.round_trip:
        from attack_core.reference import round_trip
        with open(args.round_trip) as fh:
            rep = json.load(fh)
        ok, rows = round_trip(args.round_trip, rep["pred_dir"], rep["capture_id"],
                              mask_dir=os.path.join(args.tensors, "ped_mask_tensors"))
        worst = max((r.get("rel_delta", 0.0) for r in rows), default=0.0)
        print("round trip: %d windows | worst relative div error %.3e | %s"
              % (len(rows), worst, "PASS" if ok else "FAIL"))
        for r in rows:
            if not r.get("passed", True):
                print("  window %s: reported %.6g, recomputed %.6g"
                      % (r["window"], r.get("div_reported"), r.get("div_recomputed")))
        raise SystemExit(0 if ok else 1)

    config_path = args.config or CONFIGS[args.model]
    config_parser = YAMLParser(config_path)
    config = build_config(config_parser, args.runid, args.path_mlflow, args.tensors)
    device = config_parser.device

    if config["loader"].get("crop"):
        # Same guard predict_carla.py applies: every model is scored over the same pixels.
        raise SystemExit("loader.crop is %s. CARLA evaluation must run at full resolution."
                         % (config["loader"]["crop"],))

    load_window, n_windows = build_capture_loader(config, args.id, device)

    if args.band_lo is not None and args.band_hi is not None:
        lo, hi = args.band_lo, args.band_hi
    else:
        path = args.band_json or os.path.join(args.capture, "attack_band.json")
        if not os.path.exists(path):
            raise SystemExit(
                "no band at %s. Compute it once, in an environment with avoidance's "
                "dependencies:\n  python -m attack_core.band --capture %s"
                % (path, args.capture))
        lo, hi, _meta = band_mod.read(path)

    model = SwinFlowAdapter(config, args.runid, device)

    def forward_grad_factory(x_clean):
        """The forward the attack differentiates, with its minmax normalisation set.

        `floating` (the default) lets the set be recomputed from the perturbed input, exactly as
        `prepare_chunk` does on the scored path, so one function is attacked and scored.
        `pinned` fixes the set to this window's clean tensor: a perturbation that lifts a cell
        off zero then cannot join the non-zero set, move lo/hi and rescale voxels it never
        touched. test_prepare_chunk_equivalence.py measures that at 7.9e-4 on untouched cells
        for 1e-3 on one voxel. The PGD dumps already on disk were made pinned.
        """
        if args.norm_set == "floating":
            return model.forward_grad
        nz = model.support(x_clean)
        return lambda x: model.forward_grad(x, nz=nz)

    def forward_eval(x):
        """Prediction with state reset first.

        `SwinFlowAdapter.forward` leaves the reset to its caller; every window must start from
        a clean membrane state or the prediction depends on evaluation order.
        """
        model.reset_state()
        return model.forward(x)

    # num_chunks 2 (the ANN) puts the PREVIOUS window in the first num_bins channels and the
    # target window in the last, so consecutive samples share a window and the perturbation has
    # to be carried. num_chunks 1 (the SNN) is one window per sample, so nothing is shared.
    bin_layout = None
    if config["data"]["num_chunks"] == 2:
        nb = model.num_bins
        bin_layout = runner.BinLayout(axis=1, own=slice(nb, 2 * nb),
                                      inherit_from=slice(nb, 2 * nb), inherit_to=slice(0, nb))

    handle = None
    if model.spiking:
        # cupy kernels embed their own surrogate, so the swap would not reach the backward.
        set_torch_backend(model.net)
        factory = surrogates.build_surrogate_factory(args.surrogate, **_surrogate_kwargs(args))
        if factory is not None:
            handle = surrogates.swap_surrogates(model.net, factory=factory)
            cov = handle.coverage
            print("surrogate %s on %d of %d spiking modules"
                  % (args.surrogate, cov["n_swapped"], cov["n_candidates"]))
    elif args.surrogate != "native":
        raise SystemExit("--surrogate %s on an ANN: there are no spiking neurons to swap"
                         % args.surrogate)

    if args.preflight:
        try:
            first = load_window(lo)
            if first is None:
                raise SystemExit("window %d is not in this capture" % lo)
            preflight.report(model.net, forward_eval, first[0],
                             native_alpha=2.0 if model.spiking else None,
                             device=str(device))
        finally:
            if handle is not None:
                surrogates.restore_surrogates(handle)
        raise SystemExit(0)

    g = torch.Generator(device="cpu")

    def random_sign_fn(x, eps, seed):
        g.manual_seed(int(seed))
        sign = (torch.randint(0, 2, x.shape, generator=g, dtype=torch.float32) * 2 - 1)
        return x + eps * sign.to(x.device, x.dtype)

    label = runner.attack_label(args.attack, args.surrogate)
    print("%s (%s) | objective %s%s | attack %s | norm-set %s | band [%d, %d] of %d windows"
          % (MODEL_NAMES[args.model], args.model, args.objective,
             "/" + args.sign if args.objective == "div" else "", label, args.norm_set,
             lo, hi, n_windows))
    print("epsilons: %s" % " ".join("%g" % e for e in args.epsilons))

    try:
        reports, _dirs = runner.run_sweep(
            band=(lo, hi), load_window=load_window,
            forward_grad_factory=forward_grad_factory, forward_eval=forward_eval,
            epe_fn=mod_loss_function,
            objective=args.objective, sign=args.sign, attack=label,
            epsilons=args.epsilons, iters=args.iters, alpha=args.alpha, seed=args.seed,
            clean_pred_dir=args.clean_pred, out_root=args.out, capture_id=args.id,
            model_name=MODEL_NAMES[args.model],
            # The voxel is SIGNED -- a negative cell is an OFF event, not an invalid count --
            # so unlike OF_EV_SNN's count tensor there is no non-negativity clamp here.
            clip_min=None, clip_max=None, support_mode=args.support,
            dump_adv_tensors=args.dump_adv_tensors,
            bin_layout=bin_layout, surrogate_ctx=handle,
            rhos=args.rhos, scene_mass=args.scene_mass,
            rand_init=not args.no_rand_init, random_sign_fn=random_sign_fn)
    finally:
        # Must run even when the attack raises, or a failed window leaves the adaptive
        # surrogate installed for whatever runs next in this process.
        if handle is not None:
            surrogates.restore_surrogates(handle)

    paths = runner.write_reports(reports, args.report or os.path.join(args.out, "reports"),
                                 reports[args.epsilons[0]]["label"])
    print("\nreports:")
    for eps in args.epsilons:
        print("  %s" % paths[eps])


if __name__ == "__main__":
    main()
