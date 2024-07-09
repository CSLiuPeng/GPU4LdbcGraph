#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <utility>

// CUDA核函数，计算每个顶点的局部聚类系数
__global__
void computeLccKernel(int V, int *d_OUTs_indices, int *d_OUTs_edges,
                                             int *d_INs_indices, int *d_INs_edges,
                                             double *d_LCC);

// 主机函数，调用CUDA核函数计算每个顶点的局部聚类系数
std::vector<double> computeLccGPU(int V,
                                                          std::vector<int> &h_OUTs_indices,
                                                          std::vector<int> &h_OUTs_edges,
                                                          std::vector<int> &h_INs_indices,
                                                          std::vector<int> &h_INs_edges);
