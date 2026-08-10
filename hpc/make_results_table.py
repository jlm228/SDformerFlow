""":"
exec python "$0" "$@"
"""

__doc__ = """Render evaluation results as a table shaped like the paper's Tables 3/4, for direct
cross-checking. `results_inference/<runid>/metrics_N.yml` is a raw dump of eval_DSEC_flow_SNN.py's
internal variable names, which do NOT match the paper's column labels 1:1 -- see the mapping
below, taken from loss/flow_supervised.py's AEE/AAE classes:

    metrics_N.yml key   meaning                              paper column   scaling
    AEE                 pixel endpoint error                 EPE            as-is
    AEE_outliers        frac. of px with err>3px AND >5% mag  Outlier %      x100 (stored as 0-1)
    AAE                 angular error (already in degrees)    AEE            as-is

(The paper's own third column is labelled "AEE", colliding with this codebase's "AEE" class
name for the *pixel* metric -- confirmed by units: target values like 0.81px vs 4.33deg are the
right order of magnitude for endpoint vs. angular error respectively.)

    python hpc/make_results_table.py                    # both models, using hpc/logs/*_runid.txt
    python hpc/make_results_table.py --ann-runid <id>    # override either run id
    python hpc/make_results_table.py --snn-runid <id>
"""

import argparse
import glob
import os

import yaml

# (EPE, Outlier %, AEE-angular-degrees), from Table 3 (STTFlowNet-en4-b2-p4-w10) and Table 4
# (SDformerFlow-SPE-QK-s10-c2), cropped (C) and full (F) test resolution.
TARGETS = {
    "ANN  STTFlowNet-en4-b2-p4-w10": (0.81, 2.50, 4.33),
    "SNN  SDformerFlow-SPE-QK-s10-c2 (cropped)": (0.93, 3.17, 6.37),
    "SNN  SDformerFlow-SPE-QK-s10-c2 (full)": (1.61, 8.91, 7.23),
}


def _eval_pairs(results_dir):
    """Yield (eval_id, eval_config, metrics) for every eval_N.yml/metrics_N.yml pair present."""
    for eval_path in sorted(glob.glob(os.path.join(results_dir, "eval_*.yml"))):
        eval_id = os.path.basename(eval_path)[len("eval_"):-len(".yml")]
        metrics_path = os.path.join(results_dir, "metrics_{}.yml".format(eval_id))
        if not os.path.isfile(metrics_path):
            continue
        with open(eval_path) as f:
            eval_config = yaml.safe_load(f)
        with open(metrics_path) as f:
            metrics = yaml.safe_load(f)
        yield eval_id, eval_config, metrics


def _extract(metrics):
    """metrics_N.yml values are stored as strings; AEE_outliers is a 0-1 fraction."""
    epe = float(metrics["AEE"])
    outlier_pct = float(metrics["AEE_outliers"]) * 100
    aee_deg = float(metrics["AAE"])
    return epe, outlier_pct, aee_deg


def _row(label, epe, outlier_pct, aee_deg, target):
    # Plain ASCII only: this runs in terminals on both BlueBEAR (locale not guaranteed UTF-8)
    # and whatever the user's own machine defaults to (Windows consoles commonly use cp1252,
    # which cannot encode e.g. the delta sign).
    t_epe, t_out, t_aee = target
    return "{:<42} {:>6.3f} (d{:+.3f})  {:>6.2f}% (d{:+.2f})  {:>6.3f} (d{:+.3f})".format(
        label, epe, epe - t_epe, outlier_pct, outlier_pct - t_out, aee_deg, aee_deg - t_aee)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results_inference")
    p.add_argument("--ann-runid", default=None)
    p.add_argument("--snn-runid", default=None)
    args = p.parse_args()

    def read_runid_file(path):
        if os.path.isfile(path):
            with open(path) as f:
                return f.read().strip()
        return None

    ann_runid = args.ann_runid or read_runid_file("hpc/logs/ann_runid.txt")
    snn_runid = args.snn_runid or read_runid_file("hpc/logs/snn_runid.txt")

    header = "{:<42} {:>18}  {:>16}  {:>16}".format("model", "EPE (px)", "Outlier %", "AEE (deg)")
    print(header)
    print("-" * len(header))

    found_any = False

    if ann_runid:
        results_dir = os.path.join(args.results_dir, ann_runid)
        pairs = list(_eval_pairs(results_dir))
        if not pairs:
            print("ANN  (runid {}): no results_inference/.../metrics_*.yml found yet".format(
                ann_runid))
        else:
            # Only one eval config is used for the ANN, so the latest eval_id is the answer.
            _, _, metrics = pairs[-1]
            epe, outlier_pct, aee_deg = _extract(metrics)
            target = TARGETS["ANN  STTFlowNet-en4-b2-p4-w10"]
            print(_row("ANN  STTFlowNet-en4-b2-p4-w10", epe, outlier_pct, aee_deg, target))
            found_any = True

    if snn_runid:
        results_dir = os.path.join(args.results_dir, snn_runid)
        pairs = list(_eval_pairs(results_dir))
        if not pairs:
            print("SNN  (runid {}): no results_inference/.../metrics_*.yml found yet".format(
                snn_runid))
        else:
            # C and F share one runid (same trained model, two eval configs), distinguished only
            # by the crop each eval_N.yml recorded -- not by eval_id order, which is arbitrary.
            for eval_id, eval_config, metrics in pairs:
                crop = eval_config.get("loader", {}).get("crop")
                label = ("SNN  SDformerFlow-SPE-QK-s10-c2 (full)" if crop is None else
                         "SNN  SDformerFlow-SPE-QK-s10-c2 (cropped)")
                epe, outlier_pct, aee_deg = _extract(metrics)
                print(_row(label, epe, outlier_pct, aee_deg, TARGETS[label]))
                found_any = True

    if not found_any:
        print("\nNo results yet. Run hpc/evaluate.slurm first, then re-run this script.")
    else:
        print("\n'd' is measured minus the paper's target; targets are Table 3's")
        print("STTFlowNet-en4-b2-p4-w10 row and Table 4's SDformerFlow-SPE-QK-s10-c2 row.")


if __name__ == "__main__":
    main()
