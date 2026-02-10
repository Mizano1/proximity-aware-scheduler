import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import json
import re
import sys
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
# Paths
BIN_PATH = "./bin/loadbal_sim"
RESULTS_DIR = Path("results_topology_sweep")

# Simulation Parameters for the Missing d=2 Case
N = 525
M = 100_000_000
LAMBDA = 0.95
COMM_COST = 1.0

# Cluster Config (Specific to N=525)
SERVERS_PER_CLUSTER = 25
NUM_CLUSTERS = N // SERVERS_PER_CLUSTER  # 21

TOPOLOGIES = ["cycle", "grid", "cluster"]

# Plot Styling
STYLES = {
    # Power of 2 (Baseline) - Black
    "poKL_P2":      {"color": "black",   "marker": "x", "label": "Power of 2 (d=2)", "ls": "-"},

    # Spatial Policies (Solid)
    "spatialKL_P3": {"color": "#e74c3c", "marker": "o", "label": "Spatial Po3", "ls": "-"},
    "spatialKL_P4": {"color": "#e67e22", "marker": "s", "label": "Spatial Po4", "ls": "-"},
    "spatialKL_P5": {"color": "#f1c40f", "marker": "^", "label": "Spatial Po5", "ls": "-"},
    
    # Global Policies (Dashed)
    "poKL_P3":      {"color": "#3498db", "marker": "o", "label": "Global Po3",  "ls": "--"},
    "poKL_P4":      {"color": "#9b59b6", "marker": "s", "label": "Global Po4",  "ls": "--"},
    "poKL_P5":      {"color": "#2ecc71", "marker": "^", "label": "Global Po5",  "ls": "--"},
}

def run_d2_simulation(topo):
    """
    Runs the simulation for Power of 2 (Global, k=0, L=1) if results don't exist.
    """
    # Construct filename matching your C++ convention
    # Tag format: {topo}_P{power}_{policy}
    tag = f"{topo}_P2_poKL"
    json_filename = f"poKL_{topo}_n{N}_lam{LAMBDA:.2f}_{tag}_metrics.json"
    
    # Ensure topo folder exists
    topo_dir = RESULTS_DIR / topo
    topo_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = topo_dir / json_filename

    # Skip if exists
    if json_path.exists():
        print(f"[{topo}] Power of 2 results found. Skipping simulation.")
        return

    print(f"[{topo}] Simulating Power of 2 (d=2, N={N}, M=10^8)...")
    
    # Command for Global Po2 (d=2 -> k=0, L=1)
    cmd = [
        BIN_PATH,
        "--n", str(N), "--m", str(M), "--lambda", str(LAMBDA),
        "--policy", "poKL", "--topo", topo,
        "--cost", str(COMM_COST),
        "--k", "0", "--L", "1",  # 1 Source + 0 Neighbors + 1 Global = 2 Choices
        "--outdir", str(topo_dir),
        "--tag", tag
    ]

    # Add cluster param if needed
    if topo == "cluster":
        cmd.extend(["--clusters", str(NUM_CLUSTERS)])

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)
        print(f"[{topo}] Simulation complete.")
    except Exception as e:
        print(f"[{topo}] Simulation FAILED: {e}")

def load_all_histograms():
    """Loads d=2, 3, 4, 5 histograms for the target lambda."""
    data = []
    files = sorted(RESULTS_DIR.rglob("*_hist.csv"))

    for f in files:
        try:
            fname = f.name
            
            # Filter by Lambda 0.95
            lam_match = re.search(r"lam(\d+\.\d+)", fname)
            if not lam_match: continue
            if abs(float(lam_match.group(1)) - LAMBDA) > 0.001: continue

            # Determine Topology
            if "cycle" in fname or "cycle" in f.parent.name: topo = "cycle"
            elif "grid" in fname or "grid" in f.parent.name: topo = "grid"
            elif "cluster" in fname or "cluster" in f.parent.name: topo = "cluster"
            else: continue
            
            if topo not in TOPOLOGIES: continue

            # Determine Policy and Power
            policy = "spatialKL" if "spatialKL" in fname else "poKL"
            
            p_match = re.search(r"_P(\d+)_", fname)
            power = int(p_match.group(1)) if p_match else 0
            
            # Filter for d=2,3,4,5
            if power not in [2, 3, 4, 5]: continue

            # Load CSV
            df = pd.read_csv(f)
            df.columns = [c.strip().replace("# ", "") for c in df.columns]
            
            data.append({
                "Topology": topo,
                "Key": f"{policy}_P{power}",
                "Power": power,
                "Policy": policy,
                "Data": df
            })
        except:
            pass
            
    return data

def plot_topologies(all_data):
    if not all_data:
        print("No data loaded!")
        return

    for topo in TOPOLOGIES:
        subset = [d for d in all_data if d["Topology"] == topo]
        if not subset: continue
        
        print(f"Plotting {topo}...")
        
        plt.figure(figsize=(10, 6))
        
        # Sort so Legend is orderly: P2, then P3(Global/Spatial), etc.
        subset.sort(key=lambda x: (x["Power"], x["Policy"]))

        for item in subset:
            key = item["Key"]
            df = item["Data"]
            style = STYLES.get(key, {})
            
            # Skip if style not defined (e.g. spatial P2)
            if not style: continue

            plt.plot(df["QueueLength"], df["Probability"], 
                     color=style.get("color", "black"),
                     marker=style.get("marker", None),
                     linestyle=style.get("ls", "-"),
                     linewidth=1.5,
                     markersize=5,
                     label=style.get("label", key),
                     alpha=0.9)

        plt.title(f"Queue Distribution - {topo.capitalize()} (λ={LAMBDA})", fontsize=14)
        plt.xlabel("Queue Length (k)", fontsize=12)
        plt.ylabel("Probability P(Q=k)", fontsize=12)
        
        # Tight Axes starting at 0,0
        ax = plt.gca()
        ax.set_xlim(left=0, right=12) # Focus on the head of the distribution
        ax.set_ylim(bottom=0)
        ax.margins(x=0, y=0.02)
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=10)
        
        out_path = RESULTS_DIR / f"pdf_comparison_with_P2_{topo}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()

def main():
    # 1. Ensure d=2 simulation is done
    print("--- Checking/Running Power of 2 Simulations ---")
    for topo in TOPOLOGIES:
        run_d2_simulation(topo)
    
    # 2. Load and Plot
    print("\n--- Generating Plots ---")
    data = load_all_histograms()
    plot_topologies(data)
    print("\nDone.")

if __name__ == "__main__":
    main()