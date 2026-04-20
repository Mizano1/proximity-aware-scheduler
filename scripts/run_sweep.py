"""
run_sweep.py
==============================================================
Experimentation harness for the project
"A Simulation Study of Proximity-Aware Load Balancing on
Datacentre Networks".

This script drives the primary synthetic parameter sweep of
Section 3.6 of the report. It compares the two load-balancing
policies under study at matched candidate counts d = 1 + k + L:

    POKL       (global baseline)    : k = 0,     L = d - 1
    SPATIALKL  (proximity-aware)    : k = d - 2, L = 1

For every combination of (topology, policy, lambda, d) the
script calls the C++ simulator once, reads back the metrics
JSON it produces, and caches the result on disk so the sweep
can be safely restarted without re-running configurations
that have already completed.

Any subset of the three topologies can be swept from the
command line; by default all three are run:

    python scripts/run_sweep.py                     # cycle + grid + cluster
    python scripts/run_sweep.py --topos cycle       # cycle only
    python scripts/run_sweep.py --topos grid cluster

Results are written under

    results_topology_sweep/
        cycle/   *.json, *.csv
        grid/    *.json, *.csv
        cluster/ *.json, *.csv

which is the layout expected by scripts/figures_script.py.
Running that script afterwards produces the IEEE figures
used in the Results section of the report.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ==========================================================
# SECTION 1 : CONFIGURATION
# ==========================================================

# --- Paths ---
BIN_PATH     = "./bin/loadbal_sim"
BASE_OUT_DIR = Path("results_topology_sweep")

# --- Simulation scale ---
N = 525                    # number of servers
M = 100_000_000            # total number of jobs

# --- Sweep dimensions ---
TOPOLOGIES = ["cycle", "grid", "cluster"]
POWERS     = [3, 4, 5, 6, 7, 8]
LAMBDAS    = [0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95, 0.98, 0.99]

# --- Cluster-specific parameters (used only when topo == "cluster") ---
SERVERS_PER_CLUSTER = 25
NUM_CLUSTERS        = N // SERVERS_PER_CLUSTER   # 525 / 25 = 21 clusters
COMM_COST           = 1.0                        # inter-cluster communication cost

# --- Parallel workers ---
# Default to half the cores so the laptop stays usable during long sweeps.
MAX_WORKERS = max(1, (os.cpu_count() or 2) // 2)


# ==========================================================
# SECTION 2 : STRATEGIES
# ==========================================================
# For a given "power d", we compare two strategies:
#   - Global  Po(d) : k=0,      L=d-1     (all samples are random)
#   - Spatial Po(d) : k=d-2,    L=1       (d-2 local neighbors + 1 global)
# ==========================================================

def get_strategies(power):
    return [
        {
            "name":   f"Global Po{power}",
            "policy": "poKL",
            "k": 0,
            "L": power - 1,
        },
        {
            "name":   f"Spatial Po{power}",
            "policy": "spatialKL",
            "k": power - 2,
            "L": 1,
        },
    ]


# ==========================================================
# SECTION 3 : WORKER  (runs one simulation)
# ==========================================================

def build_command(topo, lam, strategy, tag, out_dir):
    """Build the argv list passed to the C++ binary."""
    cmd = [
        BIN_PATH,
        "--n",      str(N),
        "--m",      str(M),
        "--lambda", str(lam),
        "--policy", strategy["policy"],
        "--topo",   topo,
        "--cost",   str(COMM_COST),
        "--k",      str(strategy["k"]),
        "--L",      str(strategy["L"]),
        "--outdir", str(out_dir),
        "--tag",    tag,
    ]
    # The --clusters flag is only meaningful for the cluster topology.
    if topo == "cluster":
        cmd += ["--clusters", str(NUM_CLUSTERS)]
    return cmd


def expected_json_path(topo, lam, strategy, tag, out_dir):
    """Mirror the filename format the C++ binary uses."""
    filename = f"{strategy['policy']}_{topo}_n{N}_lam{lam:.2f}_{tag}_metrics.json"
    return out_dir / filename


def make_result_row(status, topo, power, strategy, lam, data):
    """Build the row that will end up in the summary DataFrame."""
    return {
        "status":   status,
        "Topology": topo,
        "Power":    power,
        "Strategy": strategy["name"],
        "Policy":   strategy["policy"],
        "Lambda":   lam,
        "Mean_W":   data.get("mean_W", 0.0),
        "Cost":     data.get("avg_req_dist", 0.0),
    }


def run_single_simulation(args):
    """
    Worker that runs (or skips, if already present) ONE simulation.
    args = (topo, lam, strategy, power, out_dir)
    """
    topo, lam, strategy, power, out_dir = args
    tag       = f"{topo}_P{power}_{strategy['policy']}"
    json_path = expected_json_path(topo, lam, strategy, tag, out_dir)

    # --- RESUME CHECK : skip if the metrics file already exists ---
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
            return make_result_row("skipped", topo, power, strategy, lam, data)
        except Exception:
            pass   # corrupt file, fall through and re-run

    # --- RUN SIMULATION ---
    cmd = build_command(topo, lam, strategy, tag, out_dir)
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)
        with open(json_path) as f:
            data = json.load(f)
        return make_result_row("ran", topo, power, strategy, lam, data)
    except Exception as e:
        return {"status": "failed",
                "error":  f"{e} (Path: {json_path})"}


# ==========================================================
# SECTION 4 : MAIN
# ==========================================================

def parse_args():
    p = argparse.ArgumentParser(description="Unified simulation sweep")
    p.add_argument("--topos", nargs="+", default=TOPOLOGIES,
                   choices=TOPOLOGIES,
                   help="Topologies to sweep (default: all)")
    p.add_argument("--workers", type=int, default=MAX_WORKERS,
                   help=f"Parallel worker count (default: {MAX_WORKERS}, "
                        "= half the CPU cores so the laptop stays usable)")
    return p.parse_args()


def build_task_list(topos):
    """Cartesian product of (topology, power, lambda, strategy)."""
    tasks = []
    for topo in topos:
        out_dir = BASE_OUT_DIR / topo
        out_dir.mkdir(parents=True, exist_ok=True)
        for power in POWERS:
            for lam in LAMBDAS:
                for strat in get_strategies(power):
                    tasks.append((topo, lam, strat, power, out_dir))
    return tasks


def main():
    args = parse_args()

    print(f"--- PARALLEL SIMULATION SWEEP ({args.workers} workers) ---")
    print(f"Topologies : {args.topos}")
    print(f"N = {N} | M = {M:,} | clusters = {NUM_CLUSTERS} | comm_cost = {COMM_COST}")

    # 1. Compile the C++ binary
    subprocess.run(["make"], check=True, stdout=subprocess.DEVNULL)
    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Build the task list
    tasks = build_task_list(args.topos)
    print(f"Queueing {len(tasks)} simulations...")

    # 3. Run them in parallel
    start_time = time.time()
    completed  = 0
    skipped    = 0
    failed     = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_simulation, t): t for t in tasks}
        for future in as_completed(futures):
            res        = future.result()
            completed += 1

            # Simple progress bar on the same line
            sys.stdout.write(f"\rProgress: {completed}/{len(tasks)} ")
            sys.stdout.flush()

            if res["status"] == "failed":
                failed += 1
                print(f"\nFailed: {res.get('error')}")
            elif res["status"] == "skipped":
                skipped += 1

    # 4. Summary
    elapsed_min = (time.time() - start_time) / 60
    print(f"\n\nDone. Ran {completed - skipped - failed} | "
          f"Skipped {skipped} | Failed {failed}")
    print(f"Total time: {elapsed_min:.1f} minutes")
    print(f"Results in : {BASE_OUT_DIR.resolve()}")
    print("Next step  : python scripts/figures_script.py "
          f"--results-root {BASE_OUT_DIR}")


if __name__ == "__main__":
    main()
