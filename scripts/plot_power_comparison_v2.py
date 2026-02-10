import matplotlib.pyplot as plt
import pandas as pd
import json
import re
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_DIR = Path("results_topology_sweep")
TOPOLOGIES = ["grid", "cycle"]
POWERS_TO_PLOT = [3, 4, 5]

# Style for Policies (Lines)
STYLES = {
    "poKL_P3":      {"color": "blue",      "marker": "o", "linestyle": "--", "label": "Global Po3"},
    "spatialKL_P3": {"color": "blue",      "marker": "s", "linestyle": "-",  "label": "Spatial Po3"},
    "poKL_P4":      {"color": "green",     "marker": "o", "linestyle": "--", "label": "Global Po4"},
    "spatialKL_P4": {"color": "green",     "marker": "s", "linestyle": "-",  "label": "Spatial Po4"},
    "poKL_P5":      {"color": "red",       "marker": "o", "linestyle": "--", "label": "Global Po5"},
    "spatialKL_P5": {"color": "red",       "marker": "s", "linestyle": "-",  "label": "Spatial Po5"},
}

# Style for Distribution Distance (One line per Power pair)
DIST_STYLES = {
    3: {"color": "blue",  "marker": "^", "linestyle": "-", "label": "Power 3 Divergence"},
    4: {"color": "green", "marker": "^", "linestyle": "-", "label": "Power 4 Divergence"},
    5: {"color": "red",   "marker": "^", "linestyle": "-", "label": "Power 5 Divergence"},
}

def load_metrics():
    """Loads E[Q], E[R], E[c] from JSON files."""
    data = []
    files = sorted(RESULTS_DIR.rglob("*_metrics.json"))
    print(f"Loading {len(files)} metric files...")
    
    for f in files:
        try:
            with open(f, 'r') as file:
                content = json.load(file)
                match = re.search(r"_P(\d+)_", f.name)
                power = int(match.group(1)) if match else 0
                
                if power not in POWERS_TO_PLOT: continue

                # Fallback for topology detection
                topo = f.parent.name
                if topo not in TOPOLOGIES:
                    if "grid" in f.name: topo = "grid"
                    elif "cycle" in f.name: topo = "cycle"

                data.append({
                    "Topology": topo,
                    "Policy": content.get("policy"),
                    "Power": power,
                    "Lambda": content.get("lambda"),
                    "E_Q": content.get("mean_Q", 0),
                    "E_R": content.get("mean_W", 0),
                    "E_c": content.get("avg_req_dist", 0) 
                })
        except: pass
    return pd.DataFrame(data)

def calculate_distribution_l1():
    """Loads Histograms and calculates L1 Distance between policies."""
    results = []
    files = sorted(RESULTS_DIR.rglob("*_hist.csv"))
    print(f"Loading {len(files)} histogram files for L1 calc...")

    # Group files: groups[(topo, power, lambda)][policy] = filepath
    groups = {}
    
    for f in files:
        try:
            match = re.search(r"_P(\d+)_", f.name)
            power = int(match.group(1)) if match else 0
            if power not in POWERS_TO_PLOT: continue

            # Extract Lambda roughly from filename or assume grouping
            # Format: ..._lam0.90_...
            lam_match = re.search(r"lam(\d+\.\d+)", f.name)
            lam = float(lam_match.group(1)) if lam_match else 0.0

            # Determine Policy
            policy = "poKL" if "poKL" in f.name else "spatialKL"
            
            # Determine Topo
            topo = f.parent.name
            if topo not in TOPOLOGIES:
                if "grid" in f.name: topo = "grid"
                elif "cycle" in f.name: topo = "cycle"

            key = (topo, power, lam)
            if key not in groups: groups[key] = {}
            groups[key][policy] = f
        except: pass

    # Calculate L1 for each group
    for (topo, power, lam), pair in groups.items():
        if "poKL" in pair and "spatialKL" in pair:
            try:
                df1 = pd.read_csv(pair["poKL"])
                df2 = pd.read_csv(pair["spatialKL"])
                
                # Standardize columns
                df1.columns = [c.strip().replace("# ", "") for c in df1.columns]
                df2.columns = [c.strip().replace("# ", "") for c in df2.columns]
                
                # Align by QueueLength
                s1 = df1.set_index("QueueLength")["Probability"]
                s2 = df2.set_index("QueueLength")["Probability"]
                
                # Align and Fill 0
                df_compare = pd.DataFrame({'p1': s1, 'p2': s2}).fillna(0.0)
                
                # L1 Distance
                l1_dist = (df_compare['p1'] - df_compare['p2']).abs().sum()
                
                results.append({
                    "Topology": topo,
                    "Power": power,
                    "Lambda": lam,
                    "L1_Dist": l1_dist
                })
            except Exception as e:
                # print(f"Error calculating L1 for {topo} P{power} L{lam}: {e}")
                pass
                
    return pd.DataFrame(results)

def plot_standard_metric(df, topo, metric_col, ylabel, title_suffix, filename_suffix):
    """Plots E[Q], E[R], E[c]"""
    subset = df[df["Topology"] == topo].copy()
    if subset.empty: return

    plt.figure(figsize=(10, 7))
    for power in POWERS_TO_PLOT:
        for policy in ["poKL", "spatialKL"]:
            line_data = subset[(subset["Power"] == power) & (subset["Policy"] == policy)].sort_values("Lambda")
            if line_data.empty: continue
            
            style = STYLES.get(f"{policy}_P{power}", {})
            plt.plot(line_data["Lambda"], line_data[metric_col],
                     color=style.get("color"), marker=style.get("marker"),
                     linestyle=style.get("linestyle"), linewidth=2,
                     label=style.get("label"), alpha=0.8)

    plt.title(f"{title_suffix} - {topo.capitalize()}", fontsize=14)
    plt.xlabel(r"System Load ($\lambda$)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    
    out_dir = RESULTS_DIR / topo
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"combined_{filename_suffix}_{topo}.png", dpi=300)
    plt.close()

def plot_distribution_l1(df, topo):
    """Plots the Statistical L1 Distance between distributions"""
    subset = df[df["Topology"] == topo].copy()
    if subset.empty: return

    plt.figure(figsize=(10, 7))
    for power in POWERS_TO_PLOT:
        line_data = subset[subset["Power"] == power].sort_values("Lambda")
        if line_data.empty: continue
        
        style = DIST_STYLES.get(power, {})
        plt.plot(line_data["Lambda"], line_data["L1_Dist"],
                 color=style.get("color"), marker=style.get("marker"),
                 linestyle=style.get("linestyle"), linewidth=2,
                 label=style.get("label"), alpha=0.8)

    plt.title(f"Distribution Divergence (L1) - {topo.capitalize()}", fontsize=14)
    plt.xlabel(r"System Load ($\lambda$)", fontsize=12)
    plt.ylabel(r"L1 Distance ($\sum |P_{po} - P_{sp}|$)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.ylim(bottom=0)
    
    out_dir = RESULTS_DIR / topo
    plt.savefig(out_dir / f"combined_dist_l1_{topo}.png", dpi=300)
    plt.close()

def main():
    print("--- Generating Summary Plots ---")
    
    # 1. Standard Metrics
    df_metrics = load_metrics()
    if not df_metrics.empty:
        for topo in TOPOLOGIES:
            print(f"Plotting Standard Metrics for {topo}...")
            # E[Q]
            plot_standard_metric(df_metrics, topo, "E_Q", 
                               r"Mean Queue Length ($E[Q]$)", 
                               "Queue Length Comparison", "EQ")
            # E[R]
            plot_standard_metric(df_metrics, topo, "E_R", 
                               r"Mean Response Time ($E[R]$)", 
                               "Response Time Comparison", "ER")
            # E[c]
            plot_standard_metric(df_metrics, topo, "E_c", 
                               r"Avg Hop Distance ($E[c]$)", 
                               "Communication Cost Comparison", "Ec")

    # 2. Distribution L1 Distance
    df_l1 = calculate_distribution_l1()
    if not df_l1.empty:
        for topo in TOPOLOGIES:
            print(f"Plotting Distribution L1 for {topo}...")
            plot_distribution_l1(df_l1, topo)

    print("Done. Check the topology folders.")

if __name__ == "__main__":
    main()