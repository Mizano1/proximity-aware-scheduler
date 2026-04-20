"""
validate_pod.py
==============================================================
Step 1 of the research flow : verify the C++ simulator against
the closed-form Power-of-d distribution of Mitzenmacher.

This script corresponds to Section 3.7.1 of the report. Before
running the full parameter sweep, we want to be sure the
simulator reproduces the known analytical result

    P(Q >= i) = lambda^((d^i - 1) / (d - 1))

for d = 2, which reduces to the familiar double-exponential

    P(Q >= i) = lambda^(2^i - 1).

The script runs a small dedicated simulation at (k, L) = (0, 1)
i.e. global Po2, on all three topologies (cycle, grid, cluster).
Because POKL with k = 0 ignores the topology graph entirely,
the three empirical curves should collapse onto a single one
and match the analytical curve.

If the simulator is broken in the queueing core, or if the
topology code is leaking into the supposedly topology-agnostic
global path, the plot produced here will show it.

Usage:
    python scripts/validate_pod.py
"""

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================
BIN_PATH = "./bin/loadbal_sim"

# Outputs are written under a dedicated folder so the results
# of the big sweep (results_topology_sweep/) are not touched.
OUT_DIR  = Path("results_validation")
FIG_DIR  = Path("figures")

# Parameters of the validation run
N             = 525
M             = 10_000_000    # smaller than the main sweep (1e8) but still tight
LAMBDA        = 0.95          # high load -> tail of the distribution is visible
TOPOLOGIES    = ["cycle", "grid", "cluster"]
NUM_CLUSTERS  = 21            # only used when topo == "cluster"


# ==========================================================
# THEORETICAL Po-d (Mitzenmacher closed form)
# ==========================================================

def theoretical_pod_pmf(rho, d, max_k=15):
    """
    Closed-form mass function P(Q = k) derived from the
    complementary CDF  P(Q >= i) = rho^((d^i - 1) / (d - 1)).
    """
    p_ge = []
    for k in range(max_k + 2):
        exponent = (d ** k - 1) / (d - 1)
        p_ge.append(rho ** exponent if exponent < 500 else 0.0)
    pmf = [p_ge[k] - p_ge[k + 1] for k in range(max_k + 1)]
    return np.array(pmf)


# ==========================================================
# SIMULATION WRAPPER
# ==========================================================

def hist_path(topo):
    filename = f"poKL_{topo}_n{N}_lam{LAMBDA:.2f}_validation_d2_hist.csv"
    return OUT_DIR / topo / filename


def run_simulator(topo):
    """Invoke the C++ binary for the Po2 validation run."""
    out_dir = OUT_DIR / topo
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        BIN_PATH,
        "--n",      str(N),
        "--m",      str(M),
        "--lambda", str(LAMBDA),
        "--policy", "poKL",
        "--topo",   topo,
        "--k",      "0",
        "--L",      "1",
        "--outdir", str(out_dir),
        "--tag",    "validation_d2",
    ]
    if topo == "cluster":
        cmd += ["--clusters", str(NUM_CLUSTERS)]

    print(f"  running {topo}...")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)


# ==========================================================
# PLOTTING
# ==========================================================

def plot_validation():
    pmf_theory = theoretical_pod_pmf(rho=LAMBDA, d=2, max_k=12)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Analytical curve (reference)
    ax.plot(range(len(pmf_theory)), pmf_theory,
            color="black", linestyle="--", marker="x", markersize=6,
            label="Analytical Po2", zorder=10)

    # Empirical curves (one per topology)
    colors  = {"cycle": "#1f77b4", "grid": "#2ca02c", "cluster": "#d62728"}
    markers = {"cycle": "o",       "grid": "s",       "cluster": "^"}

    for topo in TOPOLOGIES:
        hist = pd.read_csv(hist_path(topo))
        ax.plot(hist["QueueLength"], hist["Probability"],
                color=colors[topo], marker=markers[topo],
                linestyle="-",
                label=f"Empirical ({topo.capitalize()})")

    ax.set_xlabel("Queue length k")
    ax.set_ylabel("P(Q = k)")
    ax.set_title(f"Po2 validation at lambda = {LAMBDA}")
    ax.set_xlim(0, 8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "validation_pod.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Make sure the simulator is up to date
    print("Compiling simulator...")
    subprocess.run(["make"], check=True, stdout=subprocess.DEVNULL)

    # 2. Run each topology once (skip any that already have a histogram file)
    print(f"Po2 validation on: {', '.join(TOPOLOGIES)}")
    for topo in TOPOLOGIES:
        if hist_path(topo).exists():
            print(f"  {topo}: already done, skipping")
        else:
            run_simulator(topo)

    # 3. Overlay empirical on analytical and save
    plot_validation()


if __name__ == "__main__":
    main()
