"""Adversarially perturb SDformerFlow or STTFlowNet over a CARLA capture; dump the flow.

    python carla_eval/attack_carla.py --model snn --config configs/valid_DSEC_supervised_full.yml \
        --runid $(cat hpc/logs/snn_runid.txt) --path_mlflow $SDF_MLFLOW_DIR \
        --tensors <capture>/saved_flow_data --capture <capture> --id carla_<capture> \
        --objective div --sign suppress --epsilons 0.0 0.05 0.1 \
        --clean-pred results/carla_eval/pred/snn --out results/attack/snn

The objective and the optimisation loop live in CARLA-hpc-scripts/attack_core, shared with
OF_EV_SNN. This file is only the model-specific half.

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

# `preds_out` is an nn.ModuleList of IFNode(v_threshold=inf) built in Spiking_STSwinNet's
# __init__ and never called in any forward. An adaptive surrogate there would give NaN (u is
# -inf), and the swap refuses to attach to it, so it is excluded by name. Nothing is lost:
# the module never runs, so it contributes no gradient either way.
SKIP_MODULES = ("preds_out",)


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


def preflight_input(capture_dir, model, window):
    """One model input for `window`, voxelised from events.npy in memory.

    The pre-flight runs before anything has been voxelised, and the pipeline deletes the
    saved_flow_data tensors once inference is done. This rebuilds the exact tensor the attack
    would be handed: `events_to_voxel` over each window's own [t_start, t_start + window_s),
    as carla_to_voxel.py writes it, then concatenated along the channel axis for num_chunks 2
    the way the (prev, target) split list pairs them.

    It deliberately does NOT go through `input_from_events`, which splits one span at its
    midpoint and so assumes consecutive windows are adjacent.
    """
    from groundtruth.inspect_capture import load_capture
    from carla_eval.carla_to_voxel import events_to_voxel

    events, windows, meta = load_capture(capture_dir, mmap=True)
    t_starts = windows["t_start_us"].to_numpy()
    chunks = model.num_chunks
    first = window - (chunks - 1)
    if first < 0 or window >= len(t_starts):
        raise SystemExit(
            "--preflight-window %d is out of range: this model needs %d consecutive windows, "
            "so it must be in [%d, %d]" % (window, chunks, chunks - 1, len(t_starts) - 1))

    window_us = int(round(float(meta["window_s"]) * 1e6))
    height, width = model.config["loader"]["resolution"]
    stride = (int(t_starts[window]) - int(t_starts[window - 1])
              if window > 0 and not np.isnan(t_starts[window - 1]) else None)
    print("capture geometry: window %d us, row stride %s us%s"
          % (window_us, stride,
             " (windows overlap)" if stride is not None and stride < window_us else ""))

    voxels = []
    t_all = events["t"]
    for w in range(first, window + 1):
        if np.isnan(t_starts[w]):
            raise SystemExit("window %d recorded no events; pick another "
                             "--preflight-window" % w)
        t0 = int(t_starts[w])
        lo, hi = np.searchsorted(t_all, (t0, t0 + window_us), side="left")
        ev = events[lo:hi]
        voxels.append(torch.as_tensor(
            events_to_voxel(ev["x"], ev["y"], ev["t"], ev["pol"],
                            model.num_bins, height, width)))

    return torch.cat(voxels, dim=0).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["snn", "ann"])
    ap.add_argument("--config", default=None, help="default: the config for --model")
    ap.add_argument("--runid", required=True, help="MLflow run id; hpc/logs/*_runid.txt")
    ap.add_argument("--path_mlflow", default="")
    ap.add_argument("--tensors", required=True, help="saved_flow_data from carla_to_voxel.py")
    ap.add_argument("--capture", required=True, help="the raw capture dir, for the band JSON")
    ap.add_argument("--id", default=None, help="capture id / split-list prefix; "
                                               "not needed with --preflight")
    ap.add_argument("--objective", required=True,
                    choices=["random_sign", "epe_global", "epe_masked", "div"])
    ap.add_argument("--sign", default="suppress", choices=["suppress", "inflate", "none"],
                    help="div only: suppress reads tau LONG, inflate reads it SHORT. 'none' is "
                         "the placeholder the sweep manifest carries for objectives that have "
                         "no direction, and is ignored unless --objective is div")
    ap.add_argument("--attack", default="pgd", choices=["fgsm", "pgd", "sapgd", "sda"])
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
    ap.add_argument("--sda-tau-factors", type=float, nargs="+", default=None,
                    help="target mode: how far the planner must be made to misread tau. "
                         "suppress drives div to div_clean/f, inflate to f*div_clean")
    ap.add_argument("--sda-budgets", type=float, nargs="+", default=None,
                    help="budget mode: events per window, from calibrate --events")
    ap.add_argument("--sda-epe-margin", type=float, default=None,
                    help="target mode with --objective epe_masked: extra masked EPE, in pixels")
    ap.add_argument("--sda-directions", default="inject_only",
                    choices=["inject_only", "inject_remove"])
    ap.add_argument("--sda-k-init", type=int, default=10,
                    help="candidates tested per round grow as (n+1)*k_init; the search width, "
                         "not the target")
    ap.add_argument("--sda-iters", type=int, default=500)
    ap.add_argument("--sda-fd-batch", type=int, default=1)
    ap.add_argument("--sda-rank", default="grad", choices=["grad", "random"],
                    help="random is the gradient-free control at matched event mass")
    ap.add_argument("--rhos", type=float, nargs="+", default=None,
                    help="the rho each epsilon was calibrated from, recorded in the reports")
    ap.add_argument("--scene-mass", type=float, default=None,
                    help="events per window, so realised rho can be reported")
    ap.add_argument("--preflight-window", type=int, default=None,
                    help="window to measure on; default: the band's first")
    ap.add_argument("--preflight", action="store_true",
                    help="measure swap coverage, mean |u| per spiking layer, timing and peak "
                         "memory on one window, then exit without attacking")
    ap.add_argument("--epsilons", type=float, nargs="+", default=None)
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
    if args.attack == "sda":
        # SDA has no epsilon: its levels are tau factors or event budgets.
        if (args.sda_tau_factors is None) == (args.sda_budgets is None):
            ap.error("--attack sda needs exactly one of --sda-tau-factors "
                     "(target mode) or --sda-budgets (budget mode)")
    elif not args.preflight and args.epsilons is None:
        ap.error("--epsilons is required unless --attack sda or --preflight")

    if not args.preflight:
        for flag, value in (("--clean-pred", args.clean_pred), ("--out", args.out),
                            ("--id", args.id)):
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

    # The pre-flight encodes its own window from events.npy: the saved_flow_data
    # tensors are written for inference and deleted afterwards, and the pre-flight is
    # meant to run before any of that.
    if args.preflight:
        load_window, n_windows = None, 0
    else:
        load_window, n_windows = build_capture_loader(config, args.id, device)

    if args.band_lo is not None and args.band_hi is not None:
        lo, hi = args.band_lo, args.band_hi
    else:
        path = args.band_json or os.path.join(args.capture, "attack_band.json")
        if os.path.exists(path):
            lo, hi, _meta = band_mod.read(path)
        elif args.preflight:
            # Measuring needs a window, not the band.
            lo, hi = max(config['data']['num_chunks'] - 1, 0), 0
        else:
            raise SystemExit(
                "no band at %s. Compute it once, in an environment with avoidance's "
                "dependencies:\n  python -m attack_core.band --capture %s"
                % (path, args.capture))

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
            handle = surrogates.swap_surrogates(model.net, factory=factory, skip=SKIP_MODULES)
            cov = handle.coverage
            print("surrogate %s on %d of %d spiking modules"
                  % (args.surrogate, cov["n_swapped"], cov["n_candidates"]))
    elif args.surrogate != "native":
        raise SystemExit("--surrogate %s on an ANN: there are no spiking neurons to swap"
                         % args.surrogate)

    if args.preflight:
        try:
            w = (args.preflight_window if args.preflight_window is not None
                 else max(lo, model.num_chunks - 1))
            x = preflight_input(args.capture, model, w).to(device)
            print("preflight on window %d, input %s" % (w, tuple(x.shape)))
            # forward_grad, not forward_eval: the latter is wrapped in no_grad, so the
            # forward+backward timing -- the number that sets EPS_CHUNK -- would be skipped.
            # base="atan": this SNN's native family. A is scale-free there, so no bracket is
            # derived -- E|u| is reported only for the M0 = 1 bias.
            preflight.report(model.net, model.forward_grad, x,
                             loss_fn=lambda f: (f ** 2).mean(),
                             native_alpha=2.0 if model.spiking else None,
                             base="atan" if model.spiking else None,
                             skip=SKIP_MODULES, device=str(device))
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
        if args.attack == "sda":
            reports, _dirs = runner.run_sweep_sda(
                band=(lo, hi), load_window=load_window,
                forward_grad_factory=forward_grad_factory, forward_eval=forward_eval,
                epe_fn=mod_loss_function,
                objective=args.objective, sign=args.sign, attack=label, seed=args.seed,
                clean_pred_dir=args.clean_pred, out_root=args.out, capture_id=args.id,
                model_name=MODEL_NAMES[args.model],
                tau_factors=args.sda_tau_factors, budgets=args.sda_budgets,
                epe_margin=args.sda_epe_margin,
                domain='voxel', directions=args.sda_directions,
                support_mode=args.support, k_init=args.sda_k_init, iters=args.sda_iters,
                fd_batch=args.sda_fd_batch, rank=args.sda_rank,
                bin_layout=bin_layout, surrogate_ctx=handle, scene_mass=args.scene_mass,
                dump_adv_tensors=args.dump_adv_tensors, verbose=True)
        else:
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
