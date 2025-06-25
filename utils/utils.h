#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include <spdlog/spdlog.h> // 用于 SPDLOG_ERROR
#include <cuda_runtime.h>   // 用于 cudaError_t, cudaSuccess, cudaGetErrorString
#include <cupti.h>          // 用于 CUptiResult, CUPTI_SUCCESS, cuptiGetResultString
#include <cstdlib>          // 用于 exit

// CUDA API 错误检查宏
#define CUDA_CALL(call)                                                        \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      SPDLOG_ERROR("CUDA Error: {} in {} at {}:{}", cudaGetErrorString(err),  \
                   __FILE__, __LINE__, __func__);                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// CUPTI API 错误检查宏
#define CUPTI_CALL(call)                                                       \
  do {                                                                         \
    CUptiResult err = call;                                                    \
    if (err != CUPTI_SUCCESS) {                                                \
      const char *errStr;                                                      \
      cuptiGetResultString(err, &errStr);                                      \
      SPDLOG_ERROR("CUPTI Error: {} in {} at {}:{}", errStr, __FILE__,        \
                   __LINE__, __func__);                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif // UTILS_UTILS_H
