"""
trace_experiment.py
==============================================================
Run the two policies (POKL and SPATIALKL) against a workload
trace rather than the synthetic Poisson / exponential generator.

Given a .dat trace (produced by scripts/process_traces.py), this
script runs the C++ simulator twice on the cluster topology at
matched d = 3 :

    POKL       (global  baseline)  : k = 0, L = 2
    SPATIALKL  (proximity-aware )  : k = 1, L = 1

It then reads back the two metrics JSONs and histograms and
produces a load-distribution comparison figure alongside the
other trace artefacts in data/alibaba_test/ and data/google_test/.

Usage :
    python scripts/trace_experiment.py data/alibaba_test/alibaba_trace.dat \\
                                       --label alibaba

    python scripts/trace_experiment.py data/google_test/final_trace.dat \\
                                       --label google \\
                                       --outdir data/google_test
"""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================
BIN_PATH = "./bin/loadbal_sim"

# Fixed parameters for the trace-driven cluster experiment.
N            = 525
NUM_CLUSTERS = 21
TOPOLOGY     = "cluster"
D            = 3            # total candidate count (1 + k + L)

# The two policies being compared, both at d = 3.
POLICIES = [
    {"name": "poKL",      "label": "Global  Po3",  "k": 0, "L": 2},
    {"name": "spatialKL", "label": "Spatial Po3",  "k": 1, "L": 1},
]


# ==========================================================
# RUN ONE POLICY
# ==========================================================

def run_policy(trace_path, policy, tag, out_dir):
    """Invoke the C++ binary with the given trace + policy."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        BIN_PATH,
        "--n",        str(N),
        "--policy",   policy["name"],
        "--topo",     TOPOLOGY,
        "--clusters", str(NUM_CLUSTERS),
        "--k",        str(policy["k"]),
        "--L",        str(policy["L"]),
        "--trace",    str(trace_path),
        "--outdir",   str(out_dir),
        "--tag",      tag,
    ]
    print(f"  running {policy['label']}...")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)


def result_paths(policy, tag, out_dir):
    """
    Mirror the filename format the C++ binary uses for a trace run :
        {policy}_{topo}_n{N}_trace_{tag}_{metrics.json,hist.csv}
    """
    base = f"{policy['name']}_{TOPOLOGY}_n{N}_trace_{tag}"
    return (out_dir / f"{base}_metrics.json",
            out_dir / f"{base}_hist.csv")


# ==========================================================
# COMPARISON PLOT  (load distribution)
# ==========================================================

def plot_comparison(histograms, label, out_path):
    """
    Bar chart overlay of the queue length distribution under
    both policies -- the same kind of figure found in
    data/{alibaba_test,google_test}/*_load_distribution_comparison.png
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Two side-by-side bars at each queue-length bucket
    width = 0.35
    colors = {"Global  Po3": "#1f77b4", "Spatial Po3": "#d62728"}

    for i, (policy_label, hist) in enumerate(histograms.items()):
        offset = (i - 0.5) * width
        ax.bar(hist["QueueLength"] + offset, hist["Probability"],
               width=width,
               label=policy_label,
               color=colors.get(policy_label, None),
               edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Queue length k")
    ax.set_ylabel("P(Q = k)")
    ax.set_title(f"Load distribution on {TOPOLOGY} (d={D}) - {label} trace")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")


# ==========================================================
# PRINT SUMMARY TABLE
# ==========================================================

def print_summary_table(label, metrics_by_label, num_jobs):
    print()
    print(f"--- Trace : {label}   ({num_jobs} jobs) ---")
    print(f"{'Policy':<14}{'E[Q]':>10}{'E[c]':>10}")
    for policy_label, m in metrics_by_label.items():
        print(f"{policy_label:<14}"
              f"{m['mean_Q']:>10.3f}"
              f"{m['avg_req_dist']:>10.3f}")
    print()


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="Run a trace-driven experiment")
    parser.add_argument("trace",
                        help="Path to a .dat trace file (from process_traces.py)")
    parser.add_argument("--label", required=True,
                        help="Short label, e.g. 'alibaba' or 'google'. Used as "
                             "the filename tag for simulator outputs.")
    parser.add_argument("--outdir", default=None,
                        help="Where to write results (default: parent folder of the trace)")
    args = parser.parse_args()

    trace_path = Path(args.trace).resolve()
    if not trace_path.exists():
        raise SystemExit(f"Trace file not found: {trace_path}")

    out_dir = Path(args.outdir) if args.outdir else trace_path.parent
    tag     = f"{args.label}_trace"

    # 1. Build the simulator
    print("Compiling simulator...")
    subprocess.run(["make"], check=True, stdout=subprocess.DEVNULL)

    # 2. Run each policy (skip if already done)
    print(f"Trace experiment : {args.label}")
    for policy in POLICIES:
        json_path, _ = result_paths(policy, tag, out_dir)
        if json_path.exists():
            print(f"  {policy['label']} : already done, skipping")
        else:
            run_policy(trace_path, policy, tag, out_dir)

    # 3. Load results
    metrics_by_label = {}
    histograms       = {}
    num_jobs         = None

    for policy in POLICIES:
        json_path, hist_path = result_paths(policy, tag, out_dir)
        with open(json_path) as f:
            data = json.load(f)
        metrics_by_label[policy["label"]] = data
        histograms      [policy["label"]] = pd.read_csv(hist_path)
        num_jobs = data["m"]

    # 4. Print Table-2-style summary
    print_summary_table(args.label, metrics_by_label, num_jobs)

    # 5. Produce the load-distribution comparison plot
    plot_path = out_dir / f"{args.label}_load_distribution_comparison.png"
    plot_comparison(histograms, args.label, plot_path)


if __name__ == "__main__":
    main()
