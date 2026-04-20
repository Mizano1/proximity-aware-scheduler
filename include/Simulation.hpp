// =============================================================
//  Simulation.hpp
//  -------------------------------------------------------------
//  Declares:
//    - SimulationResult : struct returned by Simulation::run()
//    - TraceJob         : one job read from a workload trace
//    - Simulation       : main discrete-event simulator class
// =============================================================

#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <vector>
#include <string>
#include <random>
#include <iostream>


// -------------------------------------------------------------
// Result returned at the end of a simulation run.
// -------------------------------------------------------------
struct SimulationResult {
    std::vector<double> hist;     // queue-length distribution P(Q = k)
    double total_req_dist;        // sum of routing distances (post-warmup)
    double mean_Q;                // mean queue length E[Q]
    double mean_W;                // mean waiting time E[W] = E[Q] / lambda
    double avg_req_dist;          // average routing distance per job
};


// -------------------------------------------------------------
// A single entry from a workload trace file.
// -------------------------------------------------------------
struct TraceJob {
    double inter_arrival_time;
    double duration;
};


// -------------------------------------------------------------
// Main simulation class.
// -------------------------------------------------------------
class Simulation {
public:
    // Constructor : sets up all simulation parameters and initial state.
    Simulation(int n_, double lambda__, int m_, double mu__,
               const std::string &policy_,
               const std::string &topology_,
               const std::vector<std::vector<int>> &dist_,
               const std::vector<std::vector<int>> &k_nbrs_,
               int k_, int L_, int qmax_,
               int num_clusters_ = 1,
               double comm_cost_ = 0.0,
               const std::string& trace_file_path = "",
               unsigned long long seed_ = 123456789ULL);

    // Runs the event loop until all jobs are processed and returns metrics.
    SimulationResult run();

private:
    // --- Core system parameters ---
    int n;                  // number of servers
    double lambda_;         // per-server arrival rate
    int m;                  // total jobs to simulate
    double mu_;             // per-server service rate
    std::string policy;     // "pot", "poKL", "spatialKL"
    std::string topology;   // "cycle", "grid", "cluster"

    // --- Topology / policy data ---
    std::vector<std::vector<int>> dist;    // optional precomputed distance matrix
    std::vector<std::vector<int>> k_nbrs;  // per-node neighbor lists
    int k;                                 // local neighbors sampled
    int L;                                 // global random extras
    int qmax;                              // histogram size (max queue length tracked)

    // --- Cluster-specific parameters ---
    int num_clusters;
    double comm_cost;

    // --- Runtime state ---
    double T;                          // accumulated observation time (post-warmup)
    std::vector<int> q;                // current queue length at each node
    std::vector<double> s_time;        // remaining service time at each node
    double t_arr;                      // time until next arrival
    double req_dist;                   // cumulative routing distance
    std::vector<double> q_mid_hist;    // time-weighted queue-length histogram
    int arrivals_recorded;             // count of arrivals actually measured

    // --- Trace support ---
    std::vector<TraceJob> trace_jobs;
    size_t trace_idx;
    bool use_trace;

    // --- Random-number generator ---
    std::mt19937_64 rng;

    // --- Private helpers ---
    double exp_rv(double rate);
    int    choose_node(int s);
    double calculate_distance(int u, int v);
    int    get_cluster_id(int node_index) const;
    void   load_trace(const std::string& filepath);
};

#endif
