"""
figures_script.py
--------------------------------------------------------------
Produces the four IEEE-compliant figures for the Results section
from the C++ simulator's CSV/JSON output files.

Outputs (all PDF, all sized for IEEEtran two-column figure*):
  - fig_R1_response_time.pdf  : E[R] vs lambda, 3 panels (cycle/grid/cluster)
  - fig_R2_comm_cost.pdf      : E[c] vs lambda, 3 panels
  - fig_R3_cost_savings.pdf   : sigma% vs lambda, 3 panels
  - fig_R4_po2_validation.pdf : Empirical vs analytical Po2 distribution

Expected data layout (produced by scripts/run_sweep.py):
    results_topology_sweep/
        cycle/   *.json, *.csv
        grid/    *.json, *.csv
        cluster/ *.json, *.csv

Usage:
    python scripts/figures_script.py --results-root results_topology_sweep
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter

# ============================================================
# MATPLOTLIB STYLING
# ============================================================
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   9,
    "legend.fontsize":  7,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "lines.linewidth":  1.2,
    "lines.markersize": 4,
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "grid.linestyle":   "--",
    "grid.linewidth":   0.4,
    "savefig.bbox":     "tight",
    "savefig.dpi":      600,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "text.usetex":      False,
})

TOPOLOGIES   = ["cycle", "grid", "cluster"]
TOPO_LABELS  = {"cycle": "Cycle", "grid": "Grid", "cluster": "Cluster"}
POWERS_MAIN  = [3, 4, 5]
LAMBDA_TICKS  = [0.6, 0.7, 0.8, 0.9, 0.95]
LAMBDA_LABELS = ["0.6", "0.7", "0.8", "0.9", "0.95"]
LAMBDA_XLIM   = (0.58, 0.99)

POLICY_COLORS = {
    "Global Po3":  "#1f77b4",
    "Spatial Po3": "#1f77b4",
    "Global Po4":  "#2ca02c",
    "Spatial Po4": "#2ca02c",
    "Global Po5":  "#d62728",
    "Spatial Po5": "#d62728",
}

def line_style(strategy_name):
    return "-" if strategy_name.startswith("Spatial") else "--"

def marker_style(strategy_name):
    return "s" if strategy_name.startswith("Spatial") else "o"


def style_lambda_axis(ax):
    """Apply the standard lambda-axis ticks (incl. 0.95) and limits."""
    ax.set_xlim(*LAMBDA_XLIM)
    ax.xaxis.set_major_locator(FixedLocator(LAMBDA_TICKS))
    ax.xaxis.set_major_formatter(FixedFormatter(LAMBDA_LABELS))


# ============================================================
# DATA LOADING
# ============================================================

def load_metrics(results_root, topology):
    folder = Path(results_root) / topology
    if not folder.exists():
        raise FileNotFoundError(f"Results folder not found: {folder}")

    rows = []
    for json_path in sorted(folder.glob("*_metrics.json")):
        try:
            with open(json_path) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  WARN: could not read {json_path.name}: {e}")
            continue

        policy = d.get("policy")
        power  = d.get("L", 0) + d.get("k", 0) + 1
        lam    = float(d.get("lambda", 0))

        rows.append({
            "Policy": policy,
            "Power":  power,
            "Lambda": lam,
            "Mean_Q": d.get("mean_Q", np.nan),
            "Mean_W": d.get("mean_W", np.nan),
            "E_c":    d.get("avg_req_dist", np.nan),
        })

    if not rows:
        raise RuntimeError(f"No metrics files found in {folder}")
    return pd.DataFrame(rows)


def find_d2_histogram(results_root, topology, lam=0.95):
    """Find the histogram CSV for the d=2 (k=0, L=1) Po2 validation run."""
    folder = Path(results_root) / topology

    patterns = [
        f"poKL_{topology}_n525_lam{lam:.2f}_*P2*_hist.csv",
        f"poKL_{topology}_n525_lam{lam:.2f}*hist.csv",
        f"*poKL*{topology}*lam{lam:.2f}*P2*hist.csv",
        f"*poKL*{topology}*lam{lam:.2f}*hist.csv",
    ]

    for pat in patterns:
        matches = sorted(folder.glob(pat))
        valid = [m for m in matches
                 if not any(f"_P{p}_" in m.name for p in [3,4,5,6,7,8])]
        if valid:
            return pd.read_csv(valid[0])

    raise FileNotFoundError(
        f"\n  No d=2 histogram CSV found for {topology} at lambda={lam}\n"
        f"  Looked in: {folder}\n"
    )


# ============================================================
# THEORETICAL Po-d (Mitzenmacher closed form)
# ============================================================

def theoretical_pod_pmf(rho, d, max_k=15):
    p_ge = []
    for k in range(max_k + 2):
        try:
            exponent = (d ** k - 1) / (d - 1)
            p_ge.append(rho ** exponent if exponent < 500 else 0.0)
        except OverflowError:
            p_ge.append(0.0)
    pmf = [p_ge[k] - p_ge[k + 1] for k in range(max_k + 1)]
    return np.array(pmf)


# ============================================================
# FIGURE R1: E[R] vs lambda (3 panels)
# ============================================================

def make_fig_R1(data_by_topo, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True)

    for ax, topo in zip(axes, TOPOLOGIES):
        df = data_by_topo[topo]

        for power in POWERS_MAIN:
            for policy, label_prefix in [("poKL", "Global"), ("spatialKL", "Spatial")]:
                sub = df[(df["Policy"] == policy) & (df["Power"] == power)]
                if sub.empty:
                    continue
                sub = sub.sort_values("Lambda")
                strat_name = f"{label_prefix} Po{power}"
                ax.plot(sub["Lambda"], sub["Mean_W"],
                        color=POLICY_COLORS[strat_name],
                        linestyle=line_style(strat_name),
                        marker=marker_style(strat_name),
                        label=strat_name)

        ax.set_xlabel(r"System load $\lambda$")
        ax.set_title(TOPO_LABELS[topo])
        style_lambda_axis(ax)

    axes[0].set_ylabel(r"Mean response time $\mathbb{E}[R]$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================
# FIGURE R2: E[c] vs lambda (3 panels)
# ============================================================

def make_fig_R2(data_by_topo, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=False)

    for ax, topo in zip(axes, TOPOLOGIES):
        df = data_by_topo[topo]

        for power in POWERS_MAIN:
            for policy, label_prefix in [("poKL", "Global"), ("spatialKL", "Spatial")]:
                sub = df[(df["Policy"] == policy) & (df["Power"] == power)]
                if sub.empty:
                    continue
                sub = sub.sort_values("Lambda")
                strat_name = f"{label_prefix} Po{power}"
                ax.plot(sub["Lambda"], sub["E_c"],
                        color=POLICY_COLORS[strat_name],
                        linestyle=line_style(strat_name),
                        marker=marker_style(strat_name),
                        label=strat_name)

        ax.set_xlabel(r"System load $\lambda$")
        ax.set_title(TOPO_LABELS[topo])
        style_lambda_axis(ax)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(r"$\mathbb{E}[c]$ (hops)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")

# ============================================================
# FIGURE R3: sigma% cost savings vs lambda (3 panels)
# ============================================================

def make_fig_R3(data_by_topo, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True)

    for ax, topo in zip(axes, TOPOLOGIES):
        df = data_by_topo[topo]

        for power in POWERS_MAIN:
            global_d  = df[(df["Policy"] == "poKL")     & (df["Power"] == power)]
            spatial_d = df[(df["Policy"] == "spatialKL")& (df["Power"] == power)]
            if global_d.empty or spatial_d.empty:
                continue

            merged = pd.merge(
                global_d[["Lambda", "E_c"]].rename(columns={"E_c": "E_c_global"}),
                spatial_d[["Lambda", "E_c"]].rename(columns={"E_c": "E_c_spatial"}),
                on="Lambda",
            ).sort_values("Lambda")

            sigma_pct = 100.0 * (1.0 - merged["E_c_spatial"] / merged["E_c_global"])
            ax.plot(merged["Lambda"], sigma_pct,
                    color=POLICY_COLORS[f"Spatial Po{power}"],
                    marker="o", label=fr"$d={power}$")

        ax.set_xlabel(r"System load $\lambda$")
        ax.set_title(TOPO_LABELS[topo])
        style_lambda_axis(ax)
        ax.set_ylim(0, 100)

    axes[0].set_ylabel(r"Cost saving $\sigma$ (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================
# FIGURE R4: Po2 closed-form validation
# ============================================================

def make_fig_R4(results_root, out_path, lam=0.95):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    pmf_theory = theoretical_pod_pmf(rho=lam, d=2, max_k=12)
    ax.plot(range(len(pmf_theory)), pmf_theory,
            color="black", linestyle="--", linewidth=1.4,
            marker="x", markersize=4,
            label=r"Analytical Po2", zorder=10)

    topo_colors  = {"cycle": "#1f77b4", "grid": "#2ca02c", "cluster": "#d62728"}
    topo_markers = {"cycle": "o", "grid": "s", "cluster": "^"}

    found_any = False
    for topo in TOPOLOGIES:
        try:
            hist = find_d2_histogram(results_root, topo, lam=lam)
        except FileNotFoundError as e:
            print(str(e))
            continue
        ax.plot(hist["QueueLength"], hist["Probability"],
                color=topo_colors[topo],
                marker=topo_markers[topo],
                linestyle="-",
                label=f"Empirical ({TOPO_LABELS[topo]})")
        found_any = True

    if not found_any:
        print("  ERROR: No d=2 histograms found. Skipping Figure R4.")
        plt.close(fig)
        return

    ax.set_xlabel(r"Queue length $k$")
    ax.set_ylabel(r"$P(Q = k)$")
    ax.set_xlim(0, 8)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results_topology_sweep")
    parser.add_argument("--out-dir",      default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data_by_topo = {}
    for topo in TOPOLOGIES:
        try:
            data_by_topo[topo] = load_metrics(args.results_root, topo)
            print(f"  {topo}: {len(data_by_topo[topo])} configurations loaded")
        except Exception as e:
            print(f"  ERROR loading {topo}: {e}")
            return

    print("\nGenerating figures...")
    make_fig_R1(data_by_topo, out_dir / "fig_R1_response_time.pdf")
    make_fig_R2(data_by_topo, out_dir / "fig_R2_comm_cost.pdf")
    make_fig_R3(data_by_topo, out_dir / "fig_R3_cost_savings.pdf")
    make_fig_R4(args.results_root, out_dir / "fig_R4_po2_validation.pdf")

    print("\nDone. Drop the PDFs into your LaTeX project's figure directory.")

if __name__ == "__main__":
    main()