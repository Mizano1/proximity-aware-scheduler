// =============================================================
//  simulation.cpp
//  -------------------------------------------------------------
//  Main simulation engine for the scheduler study.
//  Handles: event loop, job arrivals/services, node selection
//  policies, topology distances, and result aggregation.
// =============================================================

#include "Simulation.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <unordered_set>


// =============================================================
//  SECTION 1 : CONSTRUCTOR
//  Sets up initial parameters, RNG, and the very first job.
// =============================================================
Simulation::Simulation(int n_, double lambda__, int m_, double mu__,
                       const std::string &policy_,
                       const std::string &topology_,
                       const std::vector<std::vector<int>> &dist_,
                       const std::vector<std::vector<int>> &k_nbrs_,
                       int k_, int L_, int qmax_,
                       int num_clusters_, double comm_cost_,
                       const std::string& trace_file_path,
                       unsigned long long seed_)
    : n(n_), lambda_(lambda__), m(m_), mu_(mu__),
      policy(policy_), topology(topology_),
      dist(dist_), k_nbrs(k_nbrs_), k(k_), L(L_), qmax(qmax_),
      num_clusters(num_clusters_), comm_cost(comm_cost_),
      T(0.0), q(n_, 0), s_time(n_, 1e30), t_arr(0.0),
      req_dist(0.0), q_mid_hist(qmax_, 0.0),
      arrivals_recorded(0),
      trace_idx(0), use_trace(false)
{
    // Configurable seed so replication studies can vary the RNG
    // while keeping every other parameter identical.
    rng.seed(seed_);

    // --- Load Trace if provided ---
    if (!trace_file_path.empty()) {
        load_trace(trace_file_path);
    }

    // --- Initial System State ---
    // Drop the first job on a random node so the queue isn't empty at t=0
    std::uniform_int_distribution<int> U(0, n - 1);
    int first = U(rng);
    q[first]++;

    // First job's service time + next arrival time:
    // either taken from the trace file, or sampled from exponentials.
    if (use_trace && !trace_jobs.empty()) {
        s_time[first] = trace_jobs[0].duration;
        t_arr = trace_jobs[0].inter_arrival_time;
        trace_idx = 1;
    } else {
        s_time[first] = exp_rv(mu_);
        t_arr = exp_rv(n * lambda_);
    }
}


// =============================================================
//  SECTION 2 : TRACE LOADING
//  Reads a workload trace (inter_arrival_time, duration) pairs.
// =============================================================
void Simulation::load_trace(const std::string& filepath) {
    std::ifstream infile(filepath);
    if (!infile.good()) {
        std::cerr << "Error: Could not open trace file: " << filepath << "\n";
        use_trace = false;
        return;
    }

    // Skip the header line if the first char isn't a digit
    std::string line;
    if (infile.peek() < '0' || infile.peek() > '9') {
        std::getline(infile, line);
    }

    // Read every (dt, d) pair into trace_jobs
    double dt, d;
    while (infile >> dt >> d) {
        trace_jobs.push_back({dt, d});
    }

    if (!trace_jobs.empty()) {
        std::cout << "Loaded " << trace_jobs.size() << " jobs from trace.\n";
        use_trace = true;
    } else {
        use_trace = false;
    }
}


// =============================================================
//  SECTION 3 : HELPER FUNCTIONS
//  Small utilities: RNG sampling, cluster id, pairwise distance.
// =============================================================

// Exponential random variable with the given rate (inverse CDF method)
double Simulation::exp_rv(double rate) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    return -std::log(1.0 - U(rng)) / rate;
}

// Returns which cluster a node belongs to (0 if clustering is off)
int Simulation::get_cluster_id(int node_index) const {
    if (num_clusters <= 1) return 0;
    int servers_per_cluster = (n + num_clusters - 1) / num_clusters;
    return node_index / servers_per_cluster;
}

// Distance between two nodes u and v under the chosen topology
double Simulation::calculate_distance(int u, int v) {
    if (u == v) return 0.0;

    // --- Cluster topology ---
    if (topology == "cluster") {
        // 1. Topological distance in hops
        //    Same cluster     -> 1 hop
        //    Different cluster-> 2 hops
        double hops = (get_cluster_id(u) == get_cluster_id(v)) ? 1.0 : 2.0;

        // 2. Scale by communication cost (defaults to 1 if unset)
        double weight = (comm_cost > 1e-9) ? comm_cost : 1.0;

        return hops * weight;
    }

    // If a precomputed distance matrix was passed in, just look it up
    if (!dist.empty()) return (double)dist[u][v];

    // --- Cycle topology ---
    if (topology == "cycle") {
        int d = std::abs(u - v);
        return (double)std::min(d, n - d);
    }
    // --- Grid topology ---
    else if (topology == "grid") {
        // 1. Find the "best fit" width:
        //    start at sqrt(n) and walk down until we hit a divisor of n.
        int width = (int)std::floor(std::sqrt(n));

        while (n % width != 0) {
            width--;
        }

        // Safety check (1 always divides n, so this shouldn't trigger)
        if (width == 0) return 0.0;

        // 2. Convert linear index -> (row, column)
        int r1 = u / width;
        int c1 = u % width;

        int r2 = v / width;
        int c2 = v % width;

        // 3. Manhattan distance
        int dr = std::abs(r1 - r2);
        int dc = std::abs(c1 - c2);

        return (double)(dr + dc);
    }
    return 0.0;
}


// =============================================================
//  SECTION 4 : NODE SELECTION POLICIES
//  Given a source node s, pick the target node for a new job.
//  Supported policies: "pot", "poKL", "spatialKL".
// =============================================================
int Simulation::choose_node(int s) {
    // Pool of candidate nodes. Always includes s itself.
    std::vector<int> candidates;
    candidates.reserve(1 + k + L);
    candidates.push_back(s);
    std::uniform_int_distribution<int> U(0, n - 1);

    // --- Policy: Power-of-Two (classic) ---
    if (policy == "pot") {
        // Pick ONE extra random node different from s
        int r; do { r = U(rng); } while (r == s);
        candidates.push_back(r);
    }
    // --- Policy: Power-of-(K+L) random choices ---
    else if (policy == "poKL") {
        std::unordered_set<int> used; used.insert(s);
        while ((int)candidates.size() < 1 + k + L) {
            int r = U(rng);
            if (used.find(r) == used.end()) {
                used.insert(r);
                candidates.push_back(r);
            }
        }
    }
    // --- Policy: Spatial K local + L global ---
    else if (policy == "spatialKL") {
        if (topology == "cluster") {
            // --- CLUSTER LOGIC (Corrected) ---
            const std::vector<int>& my_cluster_nodes = k_nbrs[s];

            // 1. Pick 'k' neighbors from the local cluster
            if (!my_cluster_nodes.empty()) {
                if ((int)my_cluster_nodes.size() <= k) {
                    // Cluster is small: take them all
                    for (int v : my_cluster_nodes) candidates.push_back(v);
                } else {
                    // Cluster is bigger than k: sample k distinct indices
                    std::uniform_int_distribution<int> dist_idx(0, my_cluster_nodes.size() - 1);
                    std::unordered_set<int> picked_indices;
                    while ((int)picked_indices.size() < k) {
                        int idx = dist_idx(rng);
                        if (picked_indices.find(idx) == picked_indices.end()) {
                            picked_indices.insert(idx);
                            candidates.push_back(my_cluster_nodes[idx]);
                        }
                    }
                }
            }

            // 2. Pick 'L' global random neighbors (distinct from what we already have)
            std::unordered_set<int> used(candidates.begin(), candidates.end());
            int target_size = candidates.size() + L;

            while ((int)candidates.size() < target_size) {
                int r = U(rng);
                if (used.find(r) == used.end()) {
                    used.insert(r);
                    candidates.push_back(r);
                }
            }

        } else {
            // --- GRID / CYCLE LOGIC ---
            // Start with the fixed k-neighbor set, then add L random extras.
            for (int v : k_nbrs[s]) candidates.push_back(v);
            std::unordered_set<int> used(candidates.begin(), candidates.end());
            int target = 1 + k_nbrs[s].size() + L;
            while ((int)candidates.size() < target) {
                int r = U(rng);
                if (used.find(r) == used.end()) {
                    used.insert(r);
                    candidates.push_back(r);
                }
            }
        }
    }

    // --- SELECTION : Shortest-Queue among candidates ---
    int best = candidates[0];
    double best_score = 1e30;

    for (int cand : candidates) {
        double score;

        // Score = current queue length (smaller is better)
        score = (double)q[cand];

        if (score < best_score) {
            best_score = score;
            best = cand;
        }
    }
    return best;
}


// =============================================================
//  SECTION 5 : MAIN EVENT LOOP (run)
//  Discrete-event simulation driven by the earliest of:
//    - next arrival (t_arr)
//    - next service completion (min over s_time)
// =============================================================
SimulationResult Simulation::run() {
    int arrivals = 1;
    int max_jobs = use_trace ? trace_jobs.size() : m;

    // First 20% of jobs are warmup -> stats are ignored
    int warmup = static_cast<int>(max_jobs * 0.2);

    std::uniform_int_distribution<int> U(0, n - 1);

    while (arrivals < max_jobs) {
        // -----------------------------------------------------
        // 1. Find the next event time
        //    (minimum remaining service across busy queues
        //     vs. next arrival time t_arr)
        // -----------------------------------------------------
        int min_idx = -1;
        double min_service = 1e30;
        for (int i = 0; i < n; i++) {
            if (q[i] > 0 && s_time[i] < min_service) {
                min_service = s_time[i];
                min_idx = i;
            }
        }

        double dt = std::min(t_arr, min_service);

        // -----------------------------------------------------
        // 2. Time-Weighted Histogram Update
        //    Only record stats after warmup period
        // -----------------------------------------------------
        if (arrivals > warmup && dt > 0) {
            T += dt;
            // For every queue, add duration 'dt' to its length bin
            for (int i = 0; i < n; ++i) {
                int len = q[i];
                if (len < qmax) {
                    q_mid_hist[len] += dt;
                } else {
                    q_mid_hist[qmax-1] += dt; // overflow bin
                }
            }
        }

        // -----------------------------------------------------
        // 3. Advance all clocks by dt
        // -----------------------------------------------------
        if (dt > 0) {
             t_arr -= dt;
             for (int i=0; i<n; i++) if(q[i]>0) s_time[i] -= dt;
        }

        // -----------------------------------------------------
        // 4. Handle the event that actually fired
        // -----------------------------------------------------
        if (t_arr <= 1e-9) {
            // ===== ARRIVAL EVENT =====
            arrivals++;

            // Decide the new job's service duration
            double job_duration;
            if (use_trace) {
                job_duration = trace_jobs[trace_idx-1].duration;
            } else {
                job_duration = exp_rv(mu_);
            }

            // Pick source node s uniformly, then route via policy
            int s = U(rng);
            int chosen = choose_node(s);
            q[chosen]++;

            // Track routing distance (only after warmup)
            if (arrivals > warmup) {
                req_dist += calculate_distance(s, chosen);
                arrivals_recorded++;
            }

            // If the chosen queue was empty, start service now
            if (q[chosen] == 1) s_time[chosen] = job_duration;

            // Schedule the next arrival
            if (use_trace) {
                if (trace_idx < trace_jobs.size()) {
                    t_arr = trace_jobs[trace_idx].inter_arrival_time;
                    trace_idx++;
                } else {
                    t_arr = 1e30; // trace exhausted -> no more arrivals
                }
            } else {
                t_arr = exp_rv(n * lambda_);
            }

        }
        else {
            // ===== SERVICE COMPLETION EVENT =====
            q[min_idx]--;
            if (q[min_idx] == 0) {
                // Queue became empty -> no service scheduled
                s_time[min_idx] = 1e30;
            } else {
                // Still jobs waiting -> draw next service time
                s_time[min_idx] = exp_rv(mu_);
            }
        }
    }

    // =============================================================
    //  SECTION 6 : POST-PROCESSING
    //  Normalize histogram, compute mean queue length and mean wait.
    // =============================================================

    // Normalize the time-weighted histogram.
    // Total time accumulated across all N nodes is T * n.
    double total_time_n = T * n;

    if (total_time_n > 0) {
        for(double &v : q_mid_hist) v /= total_time_n;
    }

    // Mean queue length = sum_k k * P(Q = k)
    double mean_Q_dist = 0.0;
    for (size_t k = 0; k < q_mid_hist.size(); ++k) {
        mean_Q_dist += k * q_mid_hist[k];
    }

    // Little's law : E[W] = E[Q] / lambda
    double mean_W = (lambda_ > 0) ? mean_Q_dist / lambda_ : 0;

    // Bundle everything into the result struct
    return {
        q_mid_hist,
        req_dist,
        mean_Q_dist,
        mean_W,
        (arrivals_recorded>0 ? req_dist/arrivals_recorded : 0)
    };
}
