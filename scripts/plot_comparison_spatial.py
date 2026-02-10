import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Compare Spatial vs Random Policies")
    parser.add_argument("--spatial_file", required=True, help="Path to spatialKL histogram CSV")
    parser.add_argument("--random_file", required=True, help="Path to poKL histogram CSV")
    parser.add_argument("--output", default="comparison_spatial_vs_random.png", help="Output image file")
    args = parser.parse_args()

    # Load Data
    try:
        df_spatial = pd.read_csv(args.spatial_file)
        df_random = pd.read_csv(args.random_file)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Extract Data
    # Assumes columns are [QueueLength, Probability] or similar
    k_spatial = df_spatial.iloc[:, 0]
    p_spatial = df_spatial.iloc[:, 1]
    
    k_random = df_random.iloc[:, 0]
    p_random = df_random.iloc[:, 1]

    # Calculate Means
    mean_spatial = sum(k_spatial * p_spatial)
    mean_random = sum(k_random * p_random)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Distribution Plot (Log Scale to see tail differences)
    ax[0].plot(k_spatial, p_spatial, 'b-o', label=f'Hybrid Spatial ($E[Q]={mean_spatial:.2f}$)', linewidth=2)
    ax[0].plot(k_random, p_random, 'r--s', label=f'Global Random ($E[Q]={mean_random:.2f}$)', linewidth=2, alpha=0.7)
    ax[0].set_xlim(0, 10)
    ax[0].set_ylim(0, 0.7)
    ax[0].set_xlabel("Queue Length ($k$)")
    ax[0].set_ylabel("Probability $P(Q=k)$ ")
    ax[0].set_title("Queue Length Distribution Comparison")
    ax[0].legend()
    ax[0].grid(True, which="both", linestyle='--', alpha=0.4)

    # 2. Bar Chart of Means
    policies = ['Hybrid Spatial', 'Global Random']
    means = [mean_spatial, mean_random]
    colors = ['royalblue', 'indianred']
    
    bars = ax[1].bar(policies, means, color=colors, edgecolor='black', alpha=0.8, width=0.5)
    ax[1].set_ylabel("Mean Queue Length $E[Q]$")
    ax[1].set_title("Performance Gap")
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax[1].text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}',
                   ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Comparison plot saved to {args.output}")

if __name__ == "__main__":
    main()