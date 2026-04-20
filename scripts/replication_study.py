"""
replication_study.py
==============================================================
Check that the simulation results are stable across different
random realisations.

For each sentinel configuration (topology x policy x d), this
script runs the simulator with multiple different RNG seeds
while keeping every other parameter identical. It then reports

    - per-seed E[R] and E[c] in a CSV
    - mean, std and relative std (%) per configuration
      in a printed table

If the relative std is small compared to the gap between the
two policies, the single-run numbers from the main sweep are
stable enough to trust.

Usage :
    python scripts/replication_study.py                  # 10 seeds, default sentinels
    python scripts/replication_study.py --seeds 20       # 20 seeds instead
    python scripts/replication_study.py --m 5000000      # shorter runs
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev


# ==========================================================
# CONFIGURATION
# ==========================================================
BIN_PATH = "./bin/loadbal_sim"

OUT_DIR  = Path("results_replication")
CSV_PATH = OUT_DIR / "replication_summary.csv"

# A 10x smaller run length than the main sweep (1e8). The
# standard deviation shrinks as 1/sqrt(M), so dropping M by 10x
# only widens sigma by about sqrt(10) ~= 3.16, which is still
# easily small enough to demonstrate stability.
DEFAULT_M      = 10_000_000
DEFAULT_SEEDS  = 10

# Fixed parameters for every replication run
N              = 525
LAMBDA         = 0.95
NUM_CLUSTERS   = 21

# Sentinel configurations -- the small subset we replicate.
# Spans all three topologies and two different probe counts.
SENTINELS = [
    {"topo": "cycle",   "d": 3},
    {"topo": "cycle",   "d": 5},
    {"topo": "grid",    "d": 3},
    {"topo": "grid",    "d": 5},
    {"topo": "cluster", "d": 3},
    {"topo": "cluster", "d": 5},
]

# Both policies at matched d
POLICIES = [
    {"name": "poKL",      "label": "Global" },
    {"name": "spatialKL", "label": "Spatial"},
]


# ==========================================================
# BUILD ONE (topo, policy, d, seed) TASK
# ==========================================================

def policy_kL(policy_name, d):
    """Return (k, L) for the given policy at candidate count d."""
    if policy_name == "poKL":
        return 0, d - 1
    else:
        return d - 2, 1


def expected_json_path(topo, policy_name, d, seed, out_dir):
    tag = f"rep_P{d}_{policy_name}_s{seed}"
    name = f"{policy_name}_{topo}_n{N}_lam{LAMBDA:.2f}_{tag}_metrics.json"
    return out_dir / name


def build_command(topo, policy_name, d, seed, m_jobs, out_dir):
    k, L = policy_kL(policy_name, d)
    tag  = f"rep_P{d}_{policy_name}_s{seed}"

    cmd = [
        BIN_PATH,
        "--n",      str(N),
        "--m",      str(m_jobs),
        "--lambda", str(LAMBDA),
        "--policy", policy_name,
        "--topo",   topo,
        "--k",      str(k),
        "--L",      str(L),
        "--seed",   str(seed),
        "--outdir", str(out_dir),
        "--tag",    tag,
    ]
    if topo == "cluster":
        cmd += ["--clusters", str(NUM_CLUSTERS)]
    return cmd


def run_one(task):
    """
    task = (topo, policy_name, d, seed, m_jobs, out_dir)
    Returns a dict suitable for the CSV summary.
    """
    topo, policy_name, d, seed, m_jobs, out_dir = task
    json_path = expected_json_path(topo, policy_name, d, seed, out_dir)

    # Resume-from-cache: skip if already done
    if not json_path.exists():
        cmd = build_command(topo, policy_name, d, seed, m_jobs, out_dir)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)

    with open(json_path) as f:
        data = json.load(f)
    return {
        "Topology": topo,
        "Policy":   policy_name,
        "d":        d,
        "Seed":     seed,
        "Mean_Q":   data["mean_Q"],
        "Mean_W":   data["mean_W"],
        "E_c":      data["avg_req_dist"],
    }


# ==========================================================
# AGGREGATION
# ==========================================================

def aggregate(rows):
    """Group rows by (topo, policy, d) and compute mean/std."""
    groups = {}
    for row in rows:
        key = (row["Topology"], row["Policy"], row["d"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (topo, policy, d), items in sorted(groups.items()):
        w_vals = [r["Mean_W"] for r in items]
        c_vals = [r["E_c"]    for r in items]

        w_mean, w_std = mean(w_vals), (stdev(w_vals) if len(w_vals) > 1 else 0.0)
        c_mean, c_std = mean(c_vals), (stdev(c_vals) if len(c_vals) > 1 else 0.0)

        summary.append({
            "Topology":   topo,
            "Policy":     policy,
            "d":          d,
            "n_seeds":    len(items),
            "E[R]_mean":  w_mean,
            "E[R]_std":   w_std,
            "E[R]_relstd_pct": 100.0 * w_std / w_mean if w_mean else 0.0,
            "E[c]_mean":  c_mean,
            "E[c]_std":   c_std,
            "E[c]_relstd_pct": 100.0 * c_std / c_mean if c_mean else 0.0,
        })
    return summary


def print_summary_table(summary):
    print()
    print(f"{'Topology':<9}{'Policy':<11}{'d':>3}{'N':>4}  "
          f"{'E[R] mean':>11}{'E[R] std':>10}{'relstd%':>9}  "
          f"{'E[c] mean':>11}{'E[c] std':>10}{'relstd%':>9}")
    print("-" * 95)
    for s in summary:
        print(f"{s['Topology']:<9}{s['Policy']:<11}{s['d']:>3}{s['n_seeds']:>4}  "
              f"{s['E[R]_mean']:>11.4f}{s['E[R]_std']:>10.4f}{s['E[R]_relstd_pct']:>9.3f}  "
              f"{s['E[c]_mean']:>11.4f}{s['E[c]_std']:>10.4f}{s['E[c]_relstd_pct']:>9.3f}")
    print()


def write_outputs(per_run_rows, summary):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-run CSV
    runs_path = OUT_DIR / "replication_runs.csv"
    with open(runs_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Topology", "Policy", "d",
                                               "Seed", "Mean_Q", "Mean_W", "E_c"])
        writer.writeheader()
        writer.writerows(per_run_rows)
    print(f"Wrote per-run data:    {runs_path}")

    # Summary CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote per-config mean: {CSV_PATH}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="Seed-replication study")
    parser.add_argument("--seeds",   type=int, default=DEFAULT_SEEDS,
                        help=f"Number of seeds per config (default: {DEFAULT_SEEDS})")
    parser.add_argument("--m",       type=int, default=DEFAULT_M,
                        help=f"Jobs per run (default: {DEFAULT_M:,})")
    default_workers = max(1, (os.cpu_count() or 2) // 2)
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"Parallel worker count (default: {default_workers}, "
                             "= half the CPU cores so the laptop stays usable)")
    args = parser.parse_args()

    print(f"Replication study")
    print(f"  sentinels : {len(SENTINELS)} x {len(POLICIES)} policies = "
          f"{len(SENTINELS) * len(POLICIES)} configurations")
    print(f"  seeds     : {args.seeds}")
    print(f"  m         : {args.m:,} jobs per run")
    print(f"  total     : {len(SENTINELS) * len(POLICIES) * args.seeds} runs")

    # 1. Make sure the simulator is built
    subprocess.run(["make"], check=True, stdout=subprocess.DEVNULL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Build the full task list
    seeds = [1000 + i for i in range(args.seeds)]   # 1000, 1001, ...
    tasks = []
    for sentinel in SENTINELS:
        out_dir = OUT_DIR / sentinel["topo"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for policy in POLICIES:
            for seed in seeds:
                tasks.append((sentinel["topo"], policy["name"],
                              sentinel["d"], seed, args.m, out_dir))

    # 3. Run them in parallel
    rows     = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, t): t for t in tasks}
        for future in as_completed(futures):
            rows.append(future.result())
            completed += 1
            sys.stdout.write(f"\r  progress : {completed}/{len(tasks)}")
            sys.stdout.flush()
    print()

    # 4. Aggregate + write outputs + print table
    summary = aggregate(rows)
    write_outputs(rows, summary)
    print_summary_table(summary)

    # 5. Single-line summary across all configurations
    max_r_relstd = max(s["E[R]_relstd_pct"] for s in summary)
    max_c_relstd = max(s["E[c]_relstd_pct"] for s in summary)
    print(f"Across {args.seeds} seeds, the worst relative std is "
          f"{max_r_relstd:.3f}% on E[R] and {max_c_relstd:.3f}% on E[c].")


if __name__ == "__main__":
    main()
