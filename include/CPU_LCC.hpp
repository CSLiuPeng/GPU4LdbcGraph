#pragma once

#include <vector>
#include <utility>
#include "../include/ldbc.hpp"
template<typename T> // T is float or double
std::vector<double> computeLCC(LDBC<double> & input_graph) {
        int V = input_graph.size();
        std::cout<<"The lcc graph input size is:"<<V<<std::endl;
        std::vector<double> LCC(V, 0.0);

        for (int u = 0; u < V; ++u) {
            int triangle_count = 0;
            int deg_u = input_graph.OUTs[u].size();

            for (auto &neighbor_uv : input_graph.OUTs[u]) {
                int v = neighbor_uv.first;

                for (auto &neighbor_vw : input_graph.OUTs[v]) {
                    int w = neighbor_vw.first;

                    // Check if there's a back edge from w to u
                    for (auto &in_neighbor_w : input_graph.INs[w]) {
                        if (in_neighbor_w.first == u) {
                            // Found a triangle (u, v, w)
                            triangle_count++;
                            break;
                        }
                    }
                }
            }

            // Calculate LCC for u
            if (deg_u > 1) {
                LCC[u] = 2.0 * triangle_count / (deg_u * (deg_u - 1));
            } else {
                LCC[u] = 0.0; // if deg(u) <= 1, LCC is undefined (or considered 0)
            }
        }

        return LCC;
}
