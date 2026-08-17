""":"
exec python "$0" "$@"
"""

__doc__ = """Export reproduced DSEC results into OF_EV_SNN/runs/reproduction/reproduction.csv.

    python hpc/export_reproduction.py --out ../OF_EV_SNN/runs/reproduction/reproduction.csv

Reads the `results_inference/<runid>/{eval,metrics}_N.yml` pairs `make_results_table.py` reads,
and rewrites the rows listed in OWNED in place, filling in the run id and tagging them
`source=metrics_yml`. Rows for other models pass through untouched.

Metric name translation is imported from make_results_table.py: AEE -> EPE (px), AEE_outliers
-> outlier % (stored 0-1, so x100), AAE -> AEE (deg).
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_results_table import _eval_pairs, _extract  # noqa: E402

FIELDS = ["model", "variant", "test_resolution", "params_m", "epe", "outlier_pct", "aee_deg",
          "published_epe", "published_outlier_pct", "published_aee_deg", "n_samples", "runid",
          "source"]

# Rows this script rewrites. Anything else in the CSV is left alone.
OWNED = {
    ("STTFlowNet-en4-b2-p4-w10", "full"),
    ("SDformerFlow-SPE-QK-s10-c2", "full"),
    ("SDformerFlow-SPE-QK-s10-c2", "cropped"),
}


def read_runid_file(path):
    if os.path.isfile(path):
        with open(path) as f:
            return f.read().strip() or None
    return None


def collect(results_dir, ann_runid, snn_runid):
    """-> {(model, variant): {epe, outlier_pct, aee_deg, runid}}"""
    out = {}

    if ann_runid:
        pairs = list(_eval_pairs(os.path.join(results_dir, ann_runid)))
        if pairs:
            # Only one eval config is used for the ANN, so the latest pair is the answer.
            epe, outlier, aee = _extract(pairs[-1][2])
            out[("STTFlowNet-en4-b2-p4-w10", "full")] = {
                "epe": epe, "outlier_pct": outlier, "aee_deg": aee, "runid": ann_runid}
        else:
            print("ANN runid %s: no metrics_*.yml yet" % ann_runid)

    if snn_runid:
        pairs = list(_eval_pairs(os.path.join(results_dir, snn_runid)))
        if not pairs:
            print("SNN runid %s: no metrics_*.yml yet" % snn_runid)
        for _, eval_config, metrics in pairs:
            # Cropped and full share one run id: the same trained model evaluated twice.
            # Distinguish them by each eval config's recorded crop, not by ordering, which is
            # arbitrary.
            variant = "full" if eval_config.get("loader", {}).get("crop") is None else "cropped"
            epe, outlier, aee = _extract(metrics)
            out[("SDformerFlow-SPE-QK-s10-c2", variant)] = {
                "epe": epe, "outlier_pct": outlier, "aee_deg": aee, "runid": snn_runid}

    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join("..", "OF_EV_SNN", "runs", "reproduction",
                                                 "reproduction.csv"))
    p.add_argument("--results-dir", default="results_inference")
    p.add_argument("--ann-runid", default=None)
    p.add_argument("--snn-runid", default=None)
    p.add_argument("--n-samples", type=int, default=2152,
                   help="DSEC validation split size, recorded alongside each row")
    args = p.parse_args()

    ann_runid = args.ann_runid or read_runid_file("hpc/logs/ann_runid.txt")
    snn_runid = args.snn_runid or read_runid_file("hpc/logs/snn_runid.txt")
    found = collect(args.results_dir, ann_runid, snn_runid)

    if not found:
        raise SystemExit(
            "Nothing to export: no metrics_*.yml found for ann_runid=%s snn_runid=%s under %s.\n"
            "Run hpc/evaluate.slurm first." % (ann_runid, snn_runid, args.results_dir))

    if not os.path.isfile(args.out):
        raise SystemExit(
            "%s not found.\n"
            "This script updates an existing record rather than creating one, because it only\n"
            "knows about %d of the rows -- the rest (OF_EV_SNN, the published targets) would be\n"
            "lost if it wrote the file from scratch. Either commit the CSV in the OF_EV_SNN\n"
            "checkout and pull it here, or pass --out <path to an existing copy>."
            % (args.out, len(OWNED)))

    with open(args.out, newline="") as f:
        rows = list(csv.DictReader(f))

    updated = 0
    for row in rows:
        key = (row["model"], row["variant"])
        if key not in OWNED or key not in found:
            continue
        m = found[key]
        row["epe"] = "%.4f" % m["epe"]
        row["outlier_pct"] = "%.4f" % m["outlier_pct"]
        row["aee_deg"] = "%.4f" % m["aee_deg"]
        row["runid"] = m["runid"]
        row["n_samples"] = str(args.n_samples)
        row["source"] = "metrics_yml"
        updated += 1
        print("updated %-30s %-8s EPE %.4f  outlier %.2f%%  AEE %.4f deg  (runid %s)"
              % (row["model"], row["variant"], m["epe"], m["outlier_pct"], m["aee_deg"],
                 m["runid"]))

    missing = [k for k in found if k not in {(r["model"], r["variant"]) for r in rows}]
    if missing:
        print("\nWARNING: results found but no matching CSV row: %s" % missing)

    # Write via a temp file and replace, so an interrupted run cannot leave a half-written CSV.
    tmp = args.out + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, args.out)

    print("\nwrote %d row(s) -> %s" % (updated, args.out))
    untouched = [r["model"] for r in rows if (r["model"], r["variant"]) not in OWNED]
    if untouched:
        print("left untouched (not owned by this script): %s" % ", ".join(untouched))


if __name__ == "__main__":
    main()
