#include <GPU_LCC.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <utility>

// CUDA核函数，计算每个顶点的局部聚类系数
__global__
void computeLccKernel(int V, int *d_OUTs_indices, int *d_OUTs_edges,
                                             int *d_INs_indices, int *d_INs_edges,
                                             double *d_LCC)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x; // 每个线程处理一个顶点

    if (u < V) {
        int triangle_count = 0;
        int deg_u = d_OUTs_indices[u + 1] - d_OUTs_indices[u]; // u的出度

        // 遍历u的每个出边(v, u)
        for (int i = d_OUTs_indices[u]; i < d_OUTs_indices[u + 1]; ++i) {
            int v = d_OUTs_edges[i];

            // 遍历v的每个出边(w, v)，查找是否存在(v, u, w)形成三角形
            for (int j = d_OUTs_indices[v]; j < d_OUTs_indices[v + 1]; ++j) {
                int w = d_OUTs_edges[j];

                // 检查是否存在(w, u)，即检查INs[w]中是否有(u, w)的入边
                for (int k = d_INs_indices[w]; k < d_INs_indices[w + 1]; ++k) {
                    if (d_INs_edges[k] == u) {
                        // 找到三角形(u, v, w)
                        triangle_count++;
                        break;
                    }
                }
            }
        }

        // 计算局部聚类系数 LCC(u) = 2 * triangle_count / (deg(u) * (deg(u) - 1))
        if (deg_u > 1) {
            d_LCC[u] = 2.0 * triangle_count / (deg_u * (deg_u - 1));
        } else {
            d_LCC[u] = 0.0; // 如果deg(u) <= 1，LCC为0
        }
    }
}

// 主机函数，调用CUDA核函数计算每个顶点的局部聚类系数
std::vector<double> computeLccGPU(int V,
                                                          std::vector<int> &h_OUTs_indices,
                                                          std::vector<int> &h_OUTs_edges,
                                                          std::vector<int> &h_INs_indices,
                                                          std::vector<int> &h_INs_edges)
{
    // 分配主机和设备内存
    double *h_LCC = new double[V];
    double *d_LCC;
    int *d_OUTs_indices, *d_OUTs_edges, *d_INs_indices, *d_INs_edges;

    // 分配设备内存
    cudaMalloc((void**)&d_LCC, V * sizeof(double));
    cudaMalloc((void**)&d_OUTs_indices, (V + 1) * sizeof(int));
    cudaMalloc((void**)&d_OUTs_edges, h_OUTs_edges.size() * sizeof(int));
    cudaMalloc((void**)&d_INs_indices, (V + 1) * sizeof(int));
    cudaMalloc((void**)&d_INs_edges, h_INs_edges.size() * sizeof(int));

    // 复制图数据到设备
    cudaMemcpy(d_OUTs_indices, h_OUTs_indices.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_OUTs_edges, h_OUTs_edges.data(), h_OUTs_edges.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_INs_indices, h_INs_indices.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_INs_edges, h_INs_edges.data(), h_INs_edges.size() * sizeof(int), cudaMemcpyHostToDevice);

    // 计算块大小和线程大小
    int blockSize = 256;
    int numBlocks = (V + blockSize - 1) / blockSize;

    // 调用CUDA核函数计算局部聚类系数
    computeLccKernel<<<numBlocks, blockSize>>>(V,
                                                                       d_OUTs_indices, d_OUTs_edges,
                                                                       d_INs_indices, d_INs_edges,
                                                                       d_LCC);

    // 复制结果回主机
    cudaMemcpy(h_LCC, d_LCC, V * sizeof(double), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_LCC);
    cudaFree(d_OUTs_indices);
    cudaFree(d_OUTs_edges);
    cudaFree(d_INs_indices);
    cudaFree(d_INs_edges);

    // 将结果存储在std::vector中
    std::vector<double> result(h_LCC, h_LCC + V);

    // 释放主机内存
    delete[] h_LCC;

    return result;
}
