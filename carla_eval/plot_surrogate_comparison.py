"""Compare SA-PGD surrogate gradients on the SNN: attack strength and outcome.

    python carla_eval/plot_surrogate_comparison.py
    python carla_eval/plot_surrogate_comparison.py --scenarios 021 023 --surrogates assg pdsg

Writes two figures:

    <out>                  one column per objective; rows: the attacked hazard divergence (or
                           EPE), and the share of scenarios in each outcome tier, both against
                           the epsilon budget
    <out>_convergence.png  the same attacked quantity against SA-PGD iteration, every budget

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
from matplotlib.ticker import NullFormatter, NullLocator, PercentFormatter  # noqa: E402

# Line style, bar hatch and name per surrogate.
SURROGATES = {"assg": ("-", "", "ASSG"), "pdsg": ("--", "//", "PDSG"),
              "native": (":", "..", "native ATan")}
# Marker for each surrogate's medians and scenario dots; the dots sit too close for line style
# to tell apart.
MARKERS = {"assg": "o", "pdsg": "s", "native": "^"}
EPS_COLOURS = ["#0d0887", "#cc4778", "#f89540", "#7e03a8"]
# Length of one inference window. Flow is in pixels per window, so divergence comes out in
# 1/window; dividing by this puts it in 1/s.
WINDOW_S = 0.1
# Title, what is plotted, and the factor that turns the loss J into it. Suppression maximises
# J = -divergence, so its curves are flipped back to divergence and fall below zero.
OBJECTIVES = {"div_suppress": ("Suppress divergence", "hazard divergence (1/s)", -1.0 / WINDOW_S),
              "div_inflate": ("Inflate divergence", "hazard divergence (1/s)", 1.0 / WINDOW_S),
              "epe_masked": ("Masked EPE", "EPE on hazard (px)", 1.0)}
TIERS = [(0, "avoided, no braking", "#1a9850"), (1, "avoided with braking", "#91cf60"),
         (2, "emergency braking", "#fc8d59"), (3, "collided", "#d73027")]
# Columns that share one y-axis, symmetric about zero, in the strength row (same units).
SHARED_Y = ("div_suppress", "div_inflate")


def style(sg):
    return SURROGATES.get(sg, ("-.", "xx", sg))


def objective_meta(obj):
    return OBJECTIVES.get(obj, (obj, "J", 1.0))


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


def log_budget_axis(ax, percents):
    ax.set_xscale("log")
    ax.set_xticks(percents)
    ax.set_xticklabels(["%g" % p for p in percents])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("budget (% of scene events injected)")


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
    names = " vs ".join(style(s)[2] for s in args.surrogates)
    scen = " ".join(args.scenarios)

    # -------------------------------------------------------------------- main figure
    fig, ax = plt.subplots(2, n_obj, figsize=(4.6 * n_obj, 7.4), squeeze=False)
    for j, obj in enumerate(args.objectives):
        title, y_label, scale = objective_meta(obj)
        for m, sg in enumerate(args.surrogates):
            ls = style(sg)[0]
            # Scenario dots sit either side of each budget so the surrogates' do not overlap;
            # the medians stay on the budget.
            nudge = 1.0 + 0.06 * (m - (len(args.surrogates) - 1) / 2.0)

            # Strength: each scenario a dot, the median over its windows of the best J reached;
            # the line joins the medians across scenarios, so every scenario weighs the same.
            xs, ys = [], []
            for k, (tag, pct) in enumerate(eps):
                per = [scale * np.median(best) for best in
                       ([max(w["loss_history"]) for w in windows_of(args.attack_root, s, obj, sg, tag)
                         if w["loss_history"]] for s in args.scenarios) if best]
                if not per:
                    continue
                ax[0, j].scatter([pct * nudge] * len(per), per, s=16, color=EPS_COLOURS[k % 4],
                                 marker=MARKERS.get(sg, "D"), alpha=0.4, lw=0, zorder=2)
                xs.append(pct)
                ys.append(float(np.median(per)))
                ax[0, j].plot(pct, ys[-1], MARKERS.get(sg, "D"), color=EPS_COLOURS[k % 4],
                              mec="0.2", ms=7, zorder=4)
            ax[0, j].plot(xs, ys, ls=ls, color="0.25", lw=1.4, zorder=3)

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
                    ax[1, j].bar(x, share, width=0.5 if tag is None else width, bottom=bottom,
                                 color=col, edgecolor="0.25", lw=0.6,
                                 hatch="" if sg is None else style(sg)[1])
                    bottom += share
        ax[1, j].set_xticks(range(len(groups)))
        ax[1, j].set_xticklabels([g[0] for g in groups])
        ax[1, j].set_ylim(0, 1)
        ax[1, j].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax[1, j].set_xlabel("budget (% of scene events injected)")

        ax[0, j].set_title(title, fontweight="bold")
        ax[0, j].set_ylabel("median attacked " + y_label)
        ax[1, j].set_ylabel("share of scenarios (%)")
        log_budget_axis(ax[0, j], args.percents)
        ax[0, j].axhline(0.0, color="0.6", lw=0.8)

    # Suppression falls below zero and inflation rises above it, so the divergence columns
    # share one axis symmetric about zero.
    shared = [j for j, o in enumerate(args.objectives) if o in SHARED_Y]
    if len(shared) > 1:
        top = max(abs(v) for j in shared for v in ax[0, j].get_ylim())
        for j in shared:
            ax[0, j].set_ylim(-top, top)

    handles = [Line2D([], [], color="0.25", ls=style(sg)[0], lw=1.6,
                      label=style(sg)[2]) for sg in args.surrogates]
    handles += [Line2D([], [], color="0.6", marker=MARKERS.get(sg, "D"), mec="0.2", ls="none",
                       ms=7, label="median, %s" % style(sg)[2]) for sg in args.surrogates]
    handles += [Line2D([], [], color="0.5", marker=MARKERS.get(sg, "D"), ls="none", ms=4,
                       alpha=0.5, label="one scenario, %s" % style(sg)[2])
                for sg in args.surrogates]
    # The legend fills column by column, two rows deep: pad the budgets to an even count so
    # each group starts a fresh column and the bar keys sit together.
    if len(handles) % 2:
        handles.append(Line2D([], [], ls="none", label=""))
    handles += [Patch(facecolor=col, edgecolor="0.25", label=name) for _t, name, col in TIERS]
    handles += [Patch(facecolor="white", edgecolor="0.25", hatch=style(sg)[1],
                      label="%s (bar)" % style(sg)[2]) for sg in args.surrogates]
    fig.legend(handles=handles, loc="lower center", ncol=(len(handles) + 1) // 2, frameon=False,
               fontsize=8)
    fig.suptitle("%s: %s, scenarios %s" % (args.model, names, scen))
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(args.out, dpi=150)
    print("wrote %s" % args.out)

    # -------------------------------------------------------------------- convergence figure
    fig, ax = plt.subplots(1, n_obj, figsize=(4.6 * n_obj, 3.8), squeeze=False)
    for j, obj in enumerate(args.objectives):
        title, y_label, scale = objective_meta(obj)
        for sg in args.surrogates:
            for k, (tag, _pct) in enumerate(eps):
                hist = [np.array(w["loss_history"]) for s in args.scenarios
                        for w in windows_of(args.attack_root, s, obj, sg, tag)
                        if len(w["loss_history"]) > 1]
                if hist:
                    n = min(len(h) for h in hist)
                    ax[0, j].plot(scale * np.median([h[:n] for h in hist], 0), ls=style(sg)[0],
                                  color=EPS_COLOURS[k % 4])
        ax[0, j].set_title(title, fontweight="bold")
        ax[0, j].set_xlabel("SA-PGD iteration")
        ax[0, j].set_ylabel("median attacked " + y_label)
    handles = [Line2D([], [], color="0.25", ls=style(sg)[0], lw=1.6, label=style(sg)[2])
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
