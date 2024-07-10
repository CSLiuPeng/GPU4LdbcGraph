#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <utility>
#include "ldbc.hpp"
#include <iostream>

__global__ void computeLCC(int num_vertices, int* d_degrees, int* d_offsets, int* d_neighbors, double* d_lcc);

std::vector<double> computeLCC_GPU(LDBC<double> graph);
