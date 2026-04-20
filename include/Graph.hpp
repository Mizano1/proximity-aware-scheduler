// =============================================================
//  Graph.hpp
//  -------------------------------------------------------------
//  Neighbor-list generators for the three supported topologies:
//    - cycle   : ring of N nodes
//    - grid    : rectangular 2D grid (non-toroidal)
//    - cluster : fully-connected clusters of equal size
//
//  Each generator returns a vector< vector<int> > where
//  entry [i] holds the list of neighbors for node i.
// =============================================================

#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>


// -------------------------------------------------------------
// CYCLE topology
// -------------------------------------------------------------
// Each node i is connected to its closest neighbors on a ring:
// i+1, i-1, i+2, i-2, ... until we have k_neighbors of them.
// -------------------------------------------------------------
inline std::vector<std::vector<int>> generate_cycle_neighbors(int n, int k_neighbors) {
    std::vector<std::vector<int>> k_nbrs(n);
    for (int i = 0; i < n; i++) {
        for (int offset = 1; offset <= (k_neighbors + 1) / 2; ++offset) {
            if (k_nbrs[i].size() < (size_t)k_neighbors)
                k_nbrs[i].push_back((i + offset) % n);
            if (k_nbrs[i].size() < (size_t)k_neighbors)
                k_nbrs[i].push_back((i - offset + n) % n);
        }
    }
    return k_nbrs;
}


// -------------------------------------------------------------
// GRID topology
// -------------------------------------------------------------
// Each node is placed on a rectangular grid of width W x height H
// where W * H = n. Neighbors are the Right/Left/Down/Up cells,
// limited to k_neighbors per node and with strict (non-toroidal)
// boundaries.
// -------------------------------------------------------------
inline std::vector<std::vector<int>> generate_grid_neighbors(int n, int k_neighbors) {
    std::vector<std::vector<int>> k_nbrs(n);

    // 1. Find the best-fit width:
    //    start at sqrt(n) and walk down until we find a divisor of n.
    int width = (int)std::floor(std::sqrt(n));
    while (width > 0 && n % width != 0) {
        width--;
    }

    // --- PRIME CHECK ---
    // If width ended at 1, n has no factor other than 1 and itself,
    // which gives a 1xN line rather than a real 2D grid.
    if (width == 1 && n > 1) {
        std::cerr << "Error: N=" << n << " is prime. Cannot form a rectangular grid." << std::endl;
        return k_nbrs;
    }

    int height = n / width;

    // 2. For each node, add its four grid neighbors (if in-bounds).
    for (int i = 0; i < n; i++) {
        int r = i / width;
        int c = i % width;

        auto add = [&](int nr, int nc) {
            if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
                int neighbor = nr * width + nc;
                if (k_nbrs[i].size() < (size_t)k_neighbors) {
                    k_nbrs[i].push_back(neighbor);
                }
            }
        };

        add(r,     c + 1); // Right
        add(r,     c - 1); // Left
        add(r + 1, c    ); // Down
        add(r - 1, c    ); // Up
    }
    return k_nbrs;
}


// -------------------------------------------------------------
// CLUSTER topology
// -------------------------------------------------------------
// Nodes are partitioned into num_clusters groups. Inside each
// cluster, every node is a neighbor of every other node
// (fully connected). Used by the "spatialKL" policy so that
// local-cluster sampling is O(cluster_size).
// -------------------------------------------------------------
inline std::vector<std::vector<int>> generate_cluster_neighbors(int n, int num_clusters) {
    std::vector<std::vector<int>> k_nbrs(n);
    if (num_clusters <= 0) return k_nbrs;

    // Ceil division: servers_per_cluster = ceil(n / num_clusters)
    int servers_per_cluster = (n + num_clusters - 1) / num_clusters;

    for (int i = 0; i < n; i++) {
        int my_cluster = i / servers_per_cluster;

        // Node id range of this cluster: [start_node, end_node)
        int start_node = my_cluster * servers_per_cluster;
        int end_node   = std::min(start_node + servers_per_cluster, n);

        // Every other node in the cluster becomes a neighbor.
        for (int candidate = start_node; candidate < end_node; candidate++) {
            if (candidate != i) {
                k_nbrs[i].push_back(candidate);
            }
        }
    }
    return k_nbrs;
}

#endif
