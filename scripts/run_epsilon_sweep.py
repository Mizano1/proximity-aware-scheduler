#!/usr/bin/env python3
"""
run_epsilon_sweep.py

Sweep epsilon from 0.1 to 1.0 for the three probabilistic policies
(probA1, probB, probC) on the three topologies, at fixed lambda=0.95
and M=10^7 jobs. Also runs spatialKL with K=1, L=1 as a reference.

Output: results_epsilon_sweep/<topology>/*.json (one per config).
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# --- Sweep parameters ---
LAMBDA = 0.95
M = 10_000_000        # 10^7 jobs
N = 525
EPSILONS = [round(0.1 * i, 1) for i in range(1, 11)]   # 0.1, 0.2, ..., 1.0
POLICIES = ["probA1", "probB", "probC"]
TOPOLOGIES = ["cycle", "grid", "cluster"]

# Reference policy: spatialKL with K=1, L=1 (one run per topology)
REF_POLICY = "spatialKL"
REF_K = 1
REF_L = 1

# Cluster topology parameters
NUM_CLUSTERS = 21      # 525 / 25 = 21 clusters
COMM_COST = 0.0        # keep cost-aware tiebreak off for fair comparison


def build_binary(repo_root: Path) -> Path:
    """Compile the simulator binary via make."""
    binary = repo_root / "bin" / "loadbal_sim"
    if not binary.exists():
        print("Building loadbal_sim...")
        subprocess.run(["make"], cwd=repo_root, check=True)
    return binary


def run_one(binary: Path, outdir: Path, policy: str, topology: str,
            epsilon: float, k: int, L: int, tag: str) -> dict:
    """
    Run one simulator invocation. Returns the parsed metrics JSON.
    """
    # Resume from cache: if the metrics file already exists, just load it.
    lam_str = re.sub(r"0+$", "", f"{LAMBDA:.3f}").rstrip(".")
    expected = outdir / f"{policy}_{topology}_n{N}_lam{lam_str}_{tag}_metrics.json"
    if expected.exists():
        with open(expected) as f:
            metrics = json.load(f)
        metrics["epsilon"] = epsilon
        metrics["tag"] = tag
        return metrics
    cmd = [
        str(binary),
        "--n",       str(N),
        "--m",       str(M),
        "--lambda",  str(LAMBDA),
        "--policy",  policy,
        "--topo",    topology,
        "--k",       str(k),
        "--L",       str(L),
        "--outdir",  str(outdir),
        "--tag",     tag,
    ]
    if topology == "cluster":
        cmd += ["--clusters", str(NUM_CLUSTERS),
                "--cost",     str(COMM_COST)]
    if policy.startswith("prob"):
        cmd += ["--epsilon", str(epsilon)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "config": tag,
            "error": result.stderr.strip(),
            "stdout": result.stdout.strip(),
        }

    # Locate the metrics file the simulator just wrote.
    # Filename convention: <policy>_<topo>_n<N>_lam<lambda>[_<tag>]_metrics.json
    lam_str = re.sub(r"0+$", "", f"{LAMBDA:.3f}").rstrip(".")  # e.g. 0.95
    pattern = f"{policy}_{topology}_n{N}_lam{lam_str}_{tag}_metrics.json"
    metrics_file = outdir / pattern
    if not metrics_file.exists():
        # Fall back to a glob in case the lambda formatting differs.
        candidates = list(outdir.glob(f"{policy}_{topology}_n{N}_lam*_{tag}_metrics.json"))
        if not candidates:
            return {"config": tag, "error": f"Could not find metrics file matching {pattern}"}
        metrics_file = candidates[0]

    with open(metrics_file) as f:
        metrics = json.load(f)

    metrics["epsilon"] = epsilon
    metrics["tag"] = tag
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".",
                        help="Root of the proximity-aware-scheduler repo")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel worker processes")
    parser.add_argument("--topos", nargs="+", default=TOPOLOGIES,
                        choices=TOPOLOGIES,
                        help="Subset of topologies to run")
    parser.add_argument("--outdir", default="results_epsilon_sweep",
                        help="Output directory")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    binary = build_binary(repo_root)

    base_outdir = repo_root / args.outdir
    base_outdir.mkdir(exist_ok=True)

    # Build job list.
    jobs = []
    for topo in args.topos:
        topo_dir = base_outdir / topo
        topo_dir.mkdir(exist_ok=True)

        # Reference run: spatialKL with K=1, L=1 (one per topology)
        ref_tag = f"ref_spatialKL_K1L1"
        jobs.append(
            (binary, topo_dir, REF_POLICY, topo, 0.0, REF_K, REF_L, ref_tag)
        )

        # Probabilistic policies: epsilon sweep
        for policy in POLICIES:
            policy_k = 0 if policy == "probA1" else 1
            for eps in EPSILONS:
                eps_tag = f"{policy}_eps{int(round(eps * 10)):02d}"
                jobs.append(
                    (binary, topo_dir, policy, topo, eps, policy_k, 0, eps_tag)
                )

    print(f"Total runs: {len(jobs)}")
    print(f"Workers:    {args.workers}")
    print(f"Output:     {base_outdir}")
    print()

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, *job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            try:
                metrics = fut.result()
            except Exception as e:
                metrics = {"config": job[-1], "error": str(e)}
            results.append(metrics)
            tag = metrics.get("tag", metrics.get("config", "?"))
            if "error" in metrics:
                print(f"[{i:3d}/{len(jobs)}] FAIL  {tag}: {metrics['error']}")
            else:
                print(f"[{i:3d}/{len(jobs)}] OK    {tag}  "
                      f"E[R]={metrics.get('mean_W', float('nan')):.3f}  "
                      f"E[c]={metrics.get('avg_req_dist', float('nan')):.4f}")

    # Aggregate dump.
    agg_file = base_outdir / "all_runs.json"
    with open(agg_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAggregated results written to {agg_file}")


if __name__ == "__main__":
    main()