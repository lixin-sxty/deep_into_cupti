#include "kernel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// 模拟耗时任务的 kernel
__global__ void matmul(float *A, float *B, float *C, int N, int iterations) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    float sum = 0.0f;
    for (int iter = 0; iter < iterations; ++iter) {
      for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
      }
    }
    C[row * N + col] = sum;
  }
}

void kernel(int N, int iterations) {
  if (N <= 0 || iterations <= 0) {
    std::cerr << "Invalid input: N and iterations must be positive."
              << std::endl;
    return;
  }

  size_t size = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);

  float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

  // 1. 主机内存分配 & 初始化
  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);

  if (!h_A || !h_B || !h_C) {
    std::cerr << "Host memory allocation failed!" << std::endl;
    if (h_A)
      free(h_A);
    if (h_B)
      free(h_B);
    if (h_C)
      free(h_C);
    return;
  }

  for (int i = 0; i < N * N; ++i) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  // 2. 设备内存分配
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  if (!d_A || !d_B || !d_C) {
    std::cerr << "Device memory allocation failed!" << std::endl;

    if (d_A)
      cudaFree(d_A);
    if (d_B)
      cudaFree(d_B);
    if (d_C)
      cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    return;
  }

  // 3. Host -> Device 数据拷贝
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // 4. 设置 kernel 参数
  dim3 threads(16, 16);
  dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

  // 5. 启动 kernel
  matmul<<<blocks, threads>>>(d_A, d_B, d_C, N, iterations);

  // 6. 同步并检查错误
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    return;
  }

  // 7. 可选：拷贝结果回 Host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // 8. 可选：输出部分结果
  std::cout << "Result[0][0]: " << h_C[0] << std::endl;

  // 9. 清理资源
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);
}
