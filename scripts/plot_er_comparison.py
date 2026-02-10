import subprocess
import json
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# CONFIGURATION
# ==========================================
BIN_PATH = "./bin/loadbal_sim"
BASE_OUT_DIR = Path("results_cluster_sweep") 

# System Parameters
N = 525
SERVERS_PER_CLUSTER = 25
NUM_CLUSTERS = N // SERVERS_PER_CLUSTER  # 21 Clusters

M = 100_000_000             
COMM_COST = 1.0         

# Sweep Parameters
# We only run "cluster" here
TOPOLOGIES = ["cycle","grid","cluster"]
LAMBDAS = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85,0.9, 0.95]

def run_simulation(args):
    topo,lam,out_dir = args
    tag= f"{topo}_Pot"
    json_filename= f"Pot_{topo}_n{N}"