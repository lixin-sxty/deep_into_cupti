#include "kernel.h"       // <--- 新增：包含核函数声明
#include <cuda_runtime.h> // 用于 CUDA 运行时 API
#include <cupti.h>        // 用于 CUPTI API
#include <iostream>

// CUPTI 错误检查宏
#define CUPTI_CALL(call)                                                       \
  do {                                                                         \
    CUptiResult _status = call;                                                \
    if (_status != CUPTI_SUCCESS) {                                            \
      const char *errstr;                                                      \
      std::cerr << "CUPTI Error: " << errstr << " in " << __FILE__ << ":"      \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main() {
  std::cout << "Initializing CUPTI learning project..." << std::endl;

  // 1. 检查 CUDA 设备
  int deviceCount;
  cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(cudaStatus)
              << std::endl;
    return 1;
  }
  if (deviceCount == 0) {
    std::cerr << "No CUDA devices found." << std::endl;
    return 1;
  }
  std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;

  // 2. 选择第一个设备并创建 CUDA 上下文
  // 通过调用一个简单的 CUDA API (如 cudaSetDevice 或启动一个核函数) 来确保 CUDA
  // 上下文被创建
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(cudaStatus)
              << std::endl;
    return 1;
  }
  std::cout << "Set CUDA device to 0." << std::endl;

  // 启动一个空核函数以确保 CUDA 上下文被激活
  simple_kernel<<<1, 1>>>();
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus)
              << std::endl;
    return 1;
  }
  cudaStatus = cudaDeviceSynchronize(); // 等待核函数完成
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaDeviceSynchronize failed: "
              << cudaGetErrorString(cudaStatus) << std::endl;
    return 1;
  }
  std::cout << "CUDA context activated." << std::endl;

  // 3. 尝试获取 CUPTI 版本 (简单的 CUPTI API 调用)
  uint32_t version;
  CUPTI_CALL(cuptiGetVersion(&version));
  std::cout << "CUPTI Version: " << version << std::endl;

  std::cout << "Project initialized successfully. You can now add CUPTI "
               "activity/callback code."
            << std::endl;

  return 0;
}
