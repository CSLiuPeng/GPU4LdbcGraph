#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <utility>
#include "ldbc.hpp"
#include "../include/gpu_lcc2.cuh"

__global__ void computeLCC(int num_vertices, int* d_degrees, int* d_offsets, int* d_neighbors, double* d_lcc) {
    int v = blockIdx.x * blockDim.x + threadIdx.x; // 每个线程处理一个顶点v

    if (v < num_vertices) {
        int degree_v = d_degrees[v];
        int start_offset_v = d_offsets[v];

        int triangles = 0;
        int possible_triangles = degree_v * (degree_v - 1) / 2;

        // 计算顶点v的邻居之间的连接情况
        for (int i = 0; i < degree_v; ++i) {
            int u = d_neighbors[start_offset_v + i];

            for (int j = i + 1; j < degree_v; ++j) {
                int w = d_neighbors[start_offset_v + j];

                // 检查是否存在边 (u, w)
                // 检查 OUTs[u] 和 INs[w]
                for (int k = 0; k < d_degrees[u]; ++k) {
                    if (d_neighbors[d_offsets[u] + k] == w) {
                        triangles++;
                        break;
                    }
                }
            }
        }

        // 计算局部聚类系数
        if (possible_triangles > 0) {
            d_lcc[v] = (double)triangles / possible_triangles;
        } else {
            d_lcc[v] = 0.0f;
        }
    }
}


std::vector<double> computeLCC_GPU(LDBC<double> graph) {
    int num_vertices = graph.size();
    std::vector<std::vector<std::pair<int, double>>>OUTs = graph.OUTs;
    std::vector<std::vector<std::pair<int, double>>> INs = graph.INs;


    int* d_degrees;
    int* d_offsets;
    int* d_neighbors;
    double* d_lcc;

    // Step 1: Calculate degrees (h_degrees)
    int* h_degrees = new int[num_vertices];
    for (int v = 0; v < num_vertices; ++v) {
        h_degrees[v] = OUTs[v].size();
    }

    // Step 2: Calculate offsets (h_offsets)
    int* h_offsets = new int[num_vertices];
    int total_neighbors = 0;
    for (int v = 0; v < num_vertices; ++v) {
        h_offsets[v] = total_neighbors;
        total_neighbors += OUTs[v].size();
    }

    // Step 3: Build neighbors list (h_neighbors)
    int* h_neighbors = new int[total_neighbors];
    int current_index = 0;
    for (int v = 0; v < num_vertices; ++v) {
        for (auto& neighbor : OUTs[v]) {
            h_neighbors[current_index++] = neighbor.first;
        }
    }

    // 分配GPU内存
    cudaMalloc((void**)&d_degrees, num_vertices * sizeof(int));
    cudaMalloc((void**)&d_offsets, num_vertices * sizeof(int));
    cudaMalloc((void**)&d_neighbors, total_neighbors * sizeof(int)); // 总邻居数
    cudaMalloc((void**)&d_lcc, num_vertices * sizeof(double));

    // 将数据传输到GPU
    cudaMemcpy(d_degrees, h_degrees, num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, h_neighbors, total_neighbors * sizeof(int), cudaMemcpyHostToDevice);

    // 计算LCC核函数的启动配置
    int blockSize = 256;
    int numBlocks = (num_vertices + blockSize - 1) / blockSize;

    // 调用核函数计算LCC
    computeLCC<<<numBlocks, blockSize>>>(num_vertices, d_degrees, d_offsets, d_neighbors, d_lcc);

    // 将结果从GPU复制回主机
    double* h_lcc = new double[num_vertices];
    std::vector<double> vec;

    cudaMemcpy(h_lcc, d_lcc, num_vertices * sizeof(double), cudaMemcpyDeviceToHost);

    double value = *h_lcc;
    vec.push_back(value);

    // 打印结果示例
    // for (int i = 0; i < num_vertices; ++i) {
    //     std::cout << "Vertex " << i << " LCC: " << h_lcc[i] << std::endl;
    // }



    // 释放GPU内存
    cudaFree(d_degrees);
    cudaFree(d_offsets);
    cudaFree(d_neighbors);
    cudaFree(d_lcc);

    delete[] h_lcc;

    return vec;
}
