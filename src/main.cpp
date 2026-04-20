// =============================================================
//  main.cpp
//  -------------------------------------------------------------
//  Entry point for the load-balancing scheduler simulator.
//  Responsibilities:
//    1. Parse command-line arguments.
//    2. Build the neighbor structure for the chosen topology.
//    3. Run the simulation.
//    4. Write results (histogram CSV + metrics JSON) to disk.
// =============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <filesystem>

#include "Simulation.hpp"
#include "Graph.hpp"

namespace fs = std::filesystem;


// =============================================================
//  SECTION 1 : OUTPUT HELPERS
//  Small functions that serialize simulation results to disk.
// =============================================================

// Write the queue-length distribution as a two-column CSV.
static void write_hist_csv(const std::vector<double>& hist, const std::string& path) {
    std::ofstream out(path);
    out << "QueueLength,Probability\n";
    for (size_t i = 0; i < hist.size(); ++i) {
        if (hist[i] > 0.0)
            out << i << "," << hist[i] << "\n";
    }
}

// Write the run's metadata + aggregate metrics as a JSON file.
static void write_metrics_json(const std::string& path,
                               const std::string& policy,
                               const std::string& graph_type,
                               int n, int m, double lambda_, double mu_,
                               int k, int L, int qmax,
                               int num_clusters, double comm_cost,
                               double total_req_dist,
                               double mean_Q,
                               double mean_W,
                               double avg_req_dist) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"policy\": \"" << policy << "\",\n";
    out << "  \"graph\": \"" << graph_type << "\",\n";
    out << "  \"n\": " << n << ",\n";
    out << "  \"m\": " << m << ",\n";
    out << "  \"lambda\": " << lambda_ << ",\n";
    out << "  \"mu\": " << mu_ << ",\n";
    out << "  \"k\": " << k << ",\n";
    out << "  \"L\": " << L << ",\n";
    out << "  \"qmax\": " << qmax << ",\n";
    out << "  \"num_clusters\": " << num_clusters << ",\n";
    out << "  \"comm_cost\": " << comm_cost << ",\n";
    out << "  \"total_req_dist\": " << total_req_dist << ",\n";
    out << "  \"mean_Q\": " << mean_Q << ",\n";
    out << "  \"mean_W\": " << mean_W << ",\n";
    out << "  \"avg_req_dist\": " << avg_req_dist << "\n";
    out << "}\n";
}


// =============================================================
//  SECTION 2 : MAIN
// =============================================================
int main(int argc, char* argv[]) {

    // -------------------------------------------------------------
    // 2a. Default parameters (can all be overridden from CLI)
    // -------------------------------------------------------------
    int n = 1000;                 // number of servers
    int m = 100000;               // total number of jobs to simulate
    double lambda = 0.9;          // per-server arrival rate
    double mu = 1.0;              // per-server service rate
    std::string policy = "pot";   // "pot", "poKL", or "spatialKL"
    std::string topo = "cycle";   // "cycle", "grid", or "cluster"
    int k = 1;                    // local neighbors sampled
    int L = 1;                    // global random samples
    int qmax = 100;               // max queue length tracked in histogram

    int num_clusters = 1;         // cluster topology parameter
    double comm_cost = 0.0;       // cluster inter-cluster cost

    std::string trace_file = "";  // optional workload trace file

    unsigned long long seed = 123456789ULL;   // RNG seed (override for replication studies)

    std::string outdir = "results";
    std::string tag_suffix = "";  // extra tag appended to filenames

    // -------------------------------------------------------------
    // 2b. Parse command-line arguments
    // -------------------------------------------------------------
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--n")        == 0) n            = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--m")        == 0) m            = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--lambda")   == 0) lambda       = std::stod(argv[++i]);
        else if (strcmp(argv[i], "--mu")       == 0) mu           = std::stod(argv[++i]);
        else if (strcmp(argv[i], "--policy")   == 0) policy       = argv[++i];
        else if (strcmp(argv[i], "--topo")     == 0) topo         = argv[++i];
        else if (strcmp(argv[i], "--k")        == 0) k            = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--L")        == 0) L            = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--clusters") == 0) num_clusters = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--cost")     == 0) comm_cost    = std::stod(argv[++i]);
        else if (strcmp(argv[i], "--trace")    == 0) trace_file   = argv[++i];
        else if (strcmp(argv[i], "--seed")     == 0) seed         = std::stoull(argv[++i]);
        else if (strcmp(argv[i], "--outdir")   == 0) outdir       = argv[++i];
        else if (strcmp(argv[i], "--tag")      == 0) tag_suffix   = argv[++i];
    }

    // Make sure the output directory exists
    fs::create_directories(outdir);

    // -------------------------------------------------------------
    // 2c. Build the neighbor structure (only needed for spatialKL)
    // -------------------------------------------------------------
    std::vector<std::vector<int>> k_nbrs;
    std::vector<std::vector<int>> dist;  // kept empty; topology-based distance is computed in Simulation

    if (policy == "spatialKL") {
        if      (topo == "cycle")   k_nbrs = generate_cycle_neighbors(n, k);
        else if (topo == "grid")    k_nbrs = generate_grid_neighbors(n, k);
        else if (topo == "cluster") k_nbrs = generate_cluster_neighbors(n, num_clusters);
    }

    // -------------------------------------------------------------
    // 2d. Run the simulation
    // -------------------------------------------------------------
    std::cout << "Running: N=" << n << " Policy=" << policy
              << " Topo=" << topo;
    if (!trace_file.empty()) std::cout << " [Trace: " << trace_file << "]";
    std::cout << "..." << std::flush;

    Simulation sim(n, lambda, m, mu, policy, topo, dist, k_nbrs, k, L, qmax,
                   num_clusters, comm_cost, trace_file, seed);

    SimulationResult result = sim.run();

    std::cout << " Done. E[Q]=" << result.mean_Q << "\n";

    // -------------------------------------------------------------
    // 2e. Build the output filenames and write results
    // -------------------------------------------------------------
    std::string filename_base = policy + "_" + topo
                              + "_n" + std::to_string(n);
    if (trace_file.empty()) filename_base += "_lam" + std::to_string(lambda).substr(0, 4);
    else                    filename_base += "_trace";

    if (!tag_suffix.empty()) filename_base += "_" + tag_suffix;

    std::string hist_path = outdir + "/" + filename_base + "_hist.csv";
    std::string meta_path = outdir + "/" + filename_base + "_metrics.json";

    write_hist_csv(result.hist, hist_path);
    write_metrics_json(meta_path, policy, topo, n, m, lambda, mu, k, L, qmax,
                       num_clusters, comm_cost,
                       result.total_req_dist,
                       result.mean_Q,
                       result.mean_W,
                       result.avg_req_dist);

    return 0;
}
