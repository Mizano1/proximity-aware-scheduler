"""
quick_compare.py
==============================================================
Step 2 of the research flow : a small diagnostic comparison
before committing to the full 252-configuration sweep.

For ONE topology and ONE candidate count d, this script sweeps
lambda and draws a two-panel plot showing

    (left)  mean response time  E[R]    Global vs Spatial
    (right) mean communication  E[c]    Global vs Spatial

The idea is to sanity-check, on a small and fast run, that
the two policies produce the qualitative pattern we expect:

    - the E[R] curves should track each other within line widths
    - the Spatial E[c] curve should sit well below the Global one

If those features show up, the full sweep is worth running.
If they don't, it is much cheaper to find out here than after
kicking off the whole grid.

Usage:
    python scripts/quick_compare.py                       # cluster, d = 3
    python scripts/quick_compare.py --topo grid --d 4
    python scripts/quick_compare.py --topo cycle --d 5
"""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURATION
# ==========================================================
BIN_PATH = "./bin/loadbal_sim"

OUT_DIR  = Path("results_quick_compare")
FIG_DIR  = Path("figures")

# Small scale : 10^7 jobs is enough to see the shape of the curves
# and keeps each run short (a few seconds to a minute per point).
N             = 525
M             = 10_000_000
LAMBDAS       = [0.6, 0.7, 0.8, 0.9, 0.95]
NUM_CLUSTERS  = 21


# ==========================================================
# RUN ONE SIMULATION
# ==========================================================

def json_path(topo, policy, lam, tag, out_dir):
    name = f"{policy}_{topo}_n{N}_lam{lam:.2f}_{tag}_metrics.json"
    return out_dir / name


def run_one(topo, policy, k, L, lam, tag, out_dir):
    """Run (or reuse) a single (topology, policy, lambda) simulation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = json_path(topo, policy, lam, tag, out_dir)

    if not path.exists():
        cmd = [
            BIN_PATH,
            "--n",      str(N),
            "--m",      str(M),
            "--lambda", str(lam),
            "--policy", policy,
            "--topo",   topo,
            "--k",      str(k),
            "--L",      str(L),
            "--outdir", str(out_dir),
            "--tag",    tag,
        ]
        if topo == "cluster":
            cmd += ["--clusters", str(NUM_CLUSTERS)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)

    with open(path) as f:
        return json.load(f)


def sweep_lambda(topo, policy, k, L, tag, out_dir):
    """Run the policy across all lambdas and return (mean_W, E_c) lists."""
    mean_W = []
    e_c    = []
    for lam in LAMBDAS:
        data = run_one(topo, policy, k, L, lam, tag, out_dir)
        mean_W.append(data["mean_W"])
        e_c.append(data["avg_req_dist"])
    return mean_W, e_c


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="Quick diagnostic comparison")
    parser.add_argument("--topo", choices=["cycle", "grid", "cluster"],
                        default="cluster")
    parser.add_argument("--d", type=int, default=3,
                        help="Total candidate count d = 1 + k + L (default: 3)")
    args = parser.parse_args()

    topo = args.topo
    d    = args.d

    # (k, L) for each policy at the chosen d
    global_k,  global_L  = 0,     d - 1
    spatial_k, spatial_L = d - 2, 1

    print(f"Quick compare : topo={topo}  d={d}")
    print(f"  Global  Po{d} : k={global_k},  L={global_L}")
    print(f"  Spatial Po{d} : k={spatial_k}, L={spatial_L}")

    # Make sure the binary is built
    subprocess.run(["make"], check=True, stdout=subprocess.DEVNULL)

    out_dir = OUT_DIR / topo
    tag_g = f"quick_P{d}_poKL"
    tag_s = f"quick_P{d}_spatialKL"

    print("Running Global Po(d)...")
    global_W, global_c = sweep_lambda(topo, "poKL",      global_k,  global_L,  tag_g, out_dir)

    print("Running Spatial Po(d)...")
    spatial_W, spatial_c = sweep_lambda(topo, "spatialKL", spatial_k, spatial_L, tag_s, out_dir)

    # --- Plot ---
    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(10, 4))

    # Left : response time
    ax_r.plot(LAMBDAS, global_W,  marker="o", linestyle="--", label=f"Global Po{d}")
    ax_r.plot(LAMBDAS, spatial_W, marker="s", linestyle="-",  label=f"Spatial Po{d}")
    ax_r.set_xlabel("System load lambda")
    ax_r.set_ylabel("Mean response time E[R]")
    ax_r.set_title(f"{topo.capitalize()} : response time")
    ax_r.grid(True, alpha=0.3, linestyle="--")
    ax_r.legend()

    # Right : communication cost
    ax_c.plot(LAMBDAS, global_c,  marker="o", linestyle="--", label=f"Global Po{d}")
    ax_c.plot(LAMBDAS, spatial_c, marker="s", linestyle="-",  label=f"Spatial Po{d}")
    ax_c.set_xlabel("System load lambda")
    ax_c.set_ylabel("Mean communication cost E[c]  (hops)")
    ax_c.set_title(f"{topo.capitalize()} : communication cost")
    ax_c.grid(True, alpha=0.3, linestyle="--")
    ax_c.legend()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"quick_compare_{topo}_P{d}.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
