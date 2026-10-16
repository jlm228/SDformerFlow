"""Compare SA-PGD surrogate gradients on the SNN: attack strength, TTC error, planner outcome.

    python carla_eval/plot_surrogate_comparison.py
    python carla_eval/plot_surrogate_comparison.py --scenarios 021 023 --surrogates assg pdsg

Writes two figures:

    <out>                  one column per objective; rows: the median best loss J, the TTC
                           error across the attacked band, and the share of scenarios in each
                           outcome tier, all against the epsilon budget
    <out>_convergence.png  the median loss against SA-PGD iteration, every budget

Colour is the budget, line style (and bar hatching) the surrogate.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.lines import Line2D                                   # noqa: E402
from matplotlib.patches import Patch                                  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator              # noqa: E402

# Line style, marker, bar hatch and name per surrogate.
SURROGATES = {"assg": ("-", "o", "", "ASSG"), "pdsg": ("--", "s", "//", "PDSG"),
              "native": (":", "^", "..", "native ATan")}
EPS_COLOURS = ["#0d0887", "#cc4778", "#f89540", "#7e03a8"]
# Title and what J measures, with units.
OBJECTIVES = {"div_suppress": ("Suppress divergence", "J = -divergence (1/window)"),
              "div_inflate": ("Inflate divergence", "J = divergence (1/window)"),
              "epe_masked": ("Masked EPE", "J = EPE on hazard (px)")}
TIERS = [(0, "avoided, no braking", "#1a9850"), (1, "avoided with braking", "#91cf60"),
         (2, "emergency braking", "#fc8d59"), (3, "collided", "#d73027")]
# TTC error is drawn up to this, in %. A divergence that flipped sign reads tau as infinite.
TTC_CAP = 300.0


def style(sg):
    return SURROGATES.get(sg, ("-.", "D", "xx", sg))


def windows_of(attack_root, scenario, objective, surrogate, eps_tag):
    """Per-window entries of one attack report, or [] if it was not run."""
    pattern = os.path.join(attack_root, "snn", "reports", "carla_%s" % scenario,
                           "*_%s_sapgd-%s_eps%s.json" % (objective, surrogate, eps_tag))
    found = glob.glob(pattern)
    return json.load(open(found[0]))["windows"] if found else []


def sweep_rows(results_root, scenarios, model):
    """{(objective label, epsilon): [sweep.csv row per scenario]}; clean rows are ('clean', 0)."""
    out = {}
    for s in scenarios:
        path = os.path.join(results_root, "attack-%s" % s, "sweep.csv")
        if not os.path.exists(path):
            print("no %s, skipping scenario %s" % (path, s))
            continue
        for r in csv.DictReader(open(path)):
            if r["model"] == model:
                out.setdefault((r["objective"], round(float(r["epsilon"]), 5)), []).append(r)
    return out


def ttc_error_pct(row):
    """Band-median TTC error, %. tau = 2/div, so tau_adv/tau_clean = 1/median div ratio."""
    try:
        ratio = float(row["median_div_ratio_band"])
    except (KeyError, ValueError):
        return None
    return np.inf if ratio <= 0.0 else 100.0 * (1.0 / ratio - 1.0)


def log_budget_axis(ax, percents):
    ax.set_xscale("log")
    ax.set_xticks(percents)
    ax.set_xticklabels(["%g" % p for p in percents])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("% of scene events injected")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["016", "018", "021", "023", "026", "028"])
    ap.add_argument("--objectives", nargs="+", default=["div_suppress", "div_inflate", "epe_masked"])
    ap.add_argument("--surrogates", nargs="+", default=["assg", "pdsg"])
    ap.add_argument("--epsilons", nargs="+", default=["000.00175", "000.00699", "000.03495"],
                    help="epsilon directory tags")
    ap.add_argument("--percents", nargs="+", type=float, default=[0.25, 1.0, 5.0],
                    help="the share of scene events each epsilon injects, for the x-axis")
    ap.add_argument("--attack-root", default="results/attack")
    ap.add_argument("--results-root", default="../CARLA-hpc-scripts/results")
    ap.add_argument("--model", default="sdformerflow")
    ap.add_argument("--out", default="assg_vs_pdsg.png")
    args = ap.parse_args()

    eps = list(zip(args.epsilons, args.percents))
    rows = sweep_rows(args.results_root, args.scenarios, args.model)
    n_obj = len(args.objectives)
    names = " vs ".join(style(s)[3] for s in args.surrogates)
    scen = " ".join(args.scenarios)

    # -------------------------------------------------------------------- main figure
    fig, ax = plt.subplots(3, n_obj, figsize=(4.6 * n_obj, 10.5), squeeze=False)
    for j, obj in enumerate(args.objectives):
        title, j_label = OBJECTIVES.get(obj, (obj, "J"))
        for sg in args.surrogates:
            ls, marker, _hatch, _name = style(sg)
            label = "%s_sapgd-%s" % (obj, sg)

            # Strength: median, over every window of every scenario, of the best J reached.
            xs, ys = [], []
            for k, (tag, pct) in enumerate(eps):
                best = [max(w["loss_history"]) for s in args.scenarios
                        for w in windows_of(args.attack_root, s, obj, sg, tag) if w["loss_history"]]
                if best:
                    xs.append(pct)
                    ys.append(np.median(best))
                    ax[0, j].plot(pct, ys[-1], marker, color=EPS_COLOURS[k % 4], mec="0.2",
                                  ms=7, zorder=3)
            ax[0, j].plot(xs, ys, ls=ls, color="0.25", lw=1.4, zorder=2)

            # TTC error: each scenario a dot, the median across scenarios a line.
            xs, ys = [], []
            for k, (tag, pct) in enumerate(eps):
                errs = [e for e in (ttc_error_pct(r)
                                    for r in rows.get((label, round(float(tag), 5)), []))
                        if e is not None]
                if not errs:
                    continue
                ax[1, j].scatter([pct] * len(errs), np.minimum(errs, TTC_CAP), s=14,
                                 color=EPS_COLOURS[k % 4], marker=marker, alpha=0.45, zorder=2)
                xs.append(pct)
                ys.append(min(float(np.median(errs)), TTC_CAP))
            ax[1, j].plot(xs, ys, ls=ls, color="0.25", lw=1.6, zorder=3)

        # Outcome: share of scenarios per tier, a bar per surrogate, clean first for reference.
        groups = [("clean", None)] + [("%g %%" % pct, tag) for tag, pct in eps]
        width = 0.8 / len(args.surrogates)
        for g, (_gname, tag) in enumerate(groups):
            for m, sg in enumerate([None] if tag is None else args.surrogates):
                key = ("clean", 0.0) if tag is None else \
                    ("%s_sapgd-%s" % (obj, sg), round(float(tag), 5))
                tiers = [int(r["outcome_tier"]) for r in rows.get(key, [])
                         if r["outcome_tier"] != ""]
                if not tiers:
                    continue
                x = g if tag is None else g - 0.4 + width * (m + 0.5)
                bottom = 0.0
                for t, _tname, col in TIERS:
                    share = tiers.count(t) / float(len(tiers))
                    ax[2, j].bar(x, share, width=0.5 if tag is None else width, bottom=bottom,
                                 color=col, edgecolor="0.25", lw=0.6,
                                 hatch="" if sg is None else style(sg)[2])
                    bottom += share
        ax[2, j].set_xticks(range(len(groups)))
        ax[2, j].set_xticklabels([g[0] for g in groups])
        ax[2, j].set_ylim(0, 1)
        ax[2, j].set_xlabel("budget (% of scene events injected)")

        ax[0, j].set_title(title, fontweight="bold")
        ax[0, j].set_ylabel("median best " + j_label)
        ax[1, j].set_ylabel("TTC error across band (%)")
        ax[2, j].set_ylabel("share of scenarios")
        log_budget_axis(ax[0, j], args.percents)
        log_budget_axis(ax[1, j], args.percents)
        ax[1, j].axhline(0.0, color="0.5", lw=0.8)
        ax[1, j].axhline(TTC_CAP, color="0.7", lw=0.8, ls=":")
        ax[1, j].text(args.percents[0], TTC_CAP, " capped (tau read as infinite)",
                      fontsize=7, color="0.4", va="bottom")
        ax[1, j].set_ylim(-100, TTC_CAP * 1.12)

    handles = [Line2D([], [], color="0.25", ls=style(sg)[0], marker=style(sg)[1], lw=1.6,
                      label=style(sg)[3]) for sg in args.surrogates]
    handles += [Line2D([], [], color=EPS_COLOURS[k % 4], marker="o", ls="none", ms=7,
                       label="%g %%" % pct) for k, (_t, pct) in enumerate(eps)]
    handles += [Patch(facecolor=col, edgecolor="0.25", label=name) for _t, name, col in TIERS]
    handles += [Patch(facecolor="white", edgecolor="0.25", hatch=style(sg)[2],
                      label="%s (bar)" % style(sg)[3]) for sg in args.surrogates if style(sg)[2]]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False,
               fontsize=8)
    fig.suptitle("%s: %s, scenarios %s" % (args.model, names, scen))
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(args.out, dpi=150)
    print("wrote %s" % args.out)

    # -------------------------------------------------------------------- convergence figure
    fig, ax = plt.subplots(1, n_obj, figsize=(4.6 * n_obj, 3.8), squeeze=False)
    for j, obj in enumerate(args.objectives):
        title, j_label = OBJECTIVES.get(obj, (obj, "J"))
        for sg in args.surrogates:
            for k, (tag, _pct) in enumerate(eps):
                hist = [np.array(w["loss_history"]) for s in args.scenarios
                        for w in windows_of(args.attack_root, s, obj, sg, tag)
                        if len(w["loss_history"]) > 1]
                if hist:
                    n = min(len(h) for h in hist)
                    ax[0, j].plot(np.median([h[:n] for h in hist], 0), ls=style(sg)[0],
                                  color=EPS_COLOURS[k % 4])
        ax[0, j].set_title(title, fontweight="bold")
        ax[0, j].set_xlabel("SA-PGD iteration")
        ax[0, j].set_ylabel("median " + j_label)
    handles = [Line2D([], [], color="0.25", ls=style(sg)[0], lw=1.6, label=style(sg)[3])
               for sg in args.surrogates]
    handles += [Line2D([], [], color=EPS_COLOURS[k % 4], lw=2.5, label="%g %%" % pct)
                for k, (_t, pct) in enumerate(eps)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle("%s: SA-PGD convergence, %s, scenarios %s" % (args.model, names, scen))
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
    conv = os.path.splitext(args.out)[0] + "_convergence.png"
    fig.savefig(conv, dpi=150)
    print("wrote %s" % conv)


if __name__ == "__main__":
    main()
