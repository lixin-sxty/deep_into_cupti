#include <cstdlib>        // For malloc, free, exit
#include <cuda_runtime.h> // CUDA 运行时 API
#include <cupti.h>        // CUPTI API
#include <iostream>

#include "kernel.h" // 包含 simple_kernel 的声明

// CUPTI 错误检查宏
#define CUPTI_CALL(call)                                                       \
  do {                                                                         \
    CUptiResult _status = call;                                                \
    if (_status != CUPTI_SUCCESS) {                                            \
      const char *errstr;                                                      \
      cuptiGetResultString(_status, &errstr);                                  \
      std::cerr << "CUPTI Error: " << errstr << " in " << __FILE__ << ":"      \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// CUPTI 活动记录缓冲区分配回调函数
// CUPTI 在需要新的缓冲区来存储活动记录时会调用此函数
void bufferAlloc(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  // 为活动记录分配 1MB 的缓冲区
  *size = 1024 * 1024; // 1MB
  *buffer = (uint8_t *)malloc(*size);
  if (!*buffer) {
    std::cerr << "Failed to allocate CUPTI activity buffer." << std::endl;
    exit(EXIT_FAILURE);
  }
  *maxNumRecords = 0; // CUPTI 将根据缓冲区大小自动确定最大记录数
}

// CUPTI 活动记录缓冲区释放和处理回调函数
// CUPTI 在缓冲区满或被刷新时会调用此函数
void bufferComplete(CUcontext context, uint32_t streamId, uint8_t *buffer,
                    size_t size, size_t validSize) {
  CUpti_Activity *record = NULL;
  size_t kernel_records_count = 0; // 添加核函数记录计数器
  CUptiResult status;
  const char *errstr;

  // 遍历缓冲区中的所有活动记录
  // 循环条件改为检查 cuptiActivityGetNextRecord 的返回值
  while (true) {
    status = cuptiActivityGetNextRecord(buffer, validSize,
                                        &record); // 使用 validSize
    if (status == CUPTI_SUCCESS) {
      // 使用 switch 语句处理不同类型的活动记录，提高可读性和可扩展性
      switch (record->kind) {
      case CUPTI_ACTIVITY_KIND_KERNEL: {
        CUpti_ActivityKernel *kernel_record = (CUpti_ActivityKernel *)record;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Kernel Activity Record:" << std::endl;
        std::cout << "  Name: " << kernel_record->name << std::endl;
        std::cout << "  Correlation ID: " << kernel_record->correlationId
                  << std::endl;
        std::cout << "  Device ID: " << kernel_record->deviceId << std::endl;
        std::cout << "  Context ID: " << kernel_record->contextId << std::endl;
        std::cout << "  Stream ID: " << kernel_record->streamId << std::endl;
        std::cout << "  Grid: (" << kernel_record->gridX << ", "
                  << kernel_record->gridY << ", " << kernel_record->gridZ << ")"
                  << std::endl;
        std::cout << "  Block: (" << kernel_record->blockX << ", "
                  << kernel_record->blockY << ", " << kernel_record->blockZ
                  << ")" << std::endl;
        std::cout << "  Static Shared Memory: "
                  << kernel_record->staticSharedMemory << " bytes" << std::endl;
        std::cout << "  Dynamic Shared Memory: "
                  << kernel_record->dynamicSharedMemory << " bytes"
                  << std::endl;
        std::cout << "  Registers Per Thread: "
                  << kernel_record->registersPerThread << std::endl;
        std::cout << "  Start Time: " << kernel_record->start << " ns"
                  << std::endl;
        std::cout << "  End Time: " << kernel_record->end << " ns" << std::endl;
        std::cout << "  Duration: "
                  << (kernel_record->end - kernel_record->start) << " ns"
                  << std::endl;
        kernel_records_count++; // 增加计数
        break;
      }
      // 如果需要，可以在这里添加其他活动类型的处理，例如
      // CUPTI_ACTIVITY_KIND_MEMCPY case CUPTI_ACTIVITY_KIND_MEMCPY: {
      //     CUpti_ActivityMemcpy *memcpy_record = (CUpti_ActivityMemcpy
      //     *)record; std::cout << "Memcpy Activity Record: " <<
      //     memcpy_record->bytes << " bytes" << std::endl; break;
      // }
      default:
        // 对于其他未处理的活动类型，可以选择忽略或打印通用信息
        // cuptiGetResultString(record->kind, &errstr);
        // std::cout << "Unhandled activity kind: " << errstr << std::endl;
        break;
      }
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      // 正确地退出循环：当所有记录都被处理时
      break;
    } else {
      // 处理 cuptiActivityGetNextRecord 返回的其他错误
      cuptiGetResultString(status, &errstr); // 使用 cuptiGetResultString
      std::cerr << "CUPTI ActivityGetNextRecord Error: " << errstr << std::endl;
      break;
    }
  }

  // 检查是否有被丢弃的记录
  size_t dropped;
  // 修正 cuptiActivityGetNumDroppedRecords 的参数
  status = cuptiActivityGetNumDroppedRecords(
      context, streamId, &dropped); // 修正为 context 和 streamId
  if (status == CUPTI_SUCCESS && dropped > 0) {
    std::cerr << "Warning: Dropped " << dropped << " activity records!"
              << std::endl;
  } else if (status != CUPTI_SUCCESS) { // 处理获取丢弃记录时的错误
    cuptiGetResultString(status, &errstr);
    std::cerr << "CUPTI GetNumDroppedRecords Error: " << errstr << std::endl;
  }

  // 打印处理的活动记录总结
  std::cout << "\n--- CUPTI Activity Buffer Summary ---" << std::endl;
  std::cout << "Processed " << kernel_records_count
            << " kernel activity records." << std::endl;
  // numRecords 是 bufferComplete 回调函数的参数，表示 CUPTI 报告的记录总数
  std::cout << "Total records in buffer (reported by CUPTI): " << validSize
            << " bytes" << std::endl; // Changed to validSize as numRecords is
                                      // not passed to bufferComplete
  std::cout << "-------------------------------------\n" << std::endl;

  // 释放由 bufferAlloc 分配的缓冲区
  free(buffer);
}

int main() {
  std::cout << "Starting CUPTI Kernel Activity Demo..." << std::endl;

  // 1. 检查 CUDA 设备并创建上下文
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
  cudaStatus = cudaSetDevice(0); // 选择第一个设备
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(cudaStatus)
              << std::endl;
    return 1;
  }
  std::cout << "Set CUDA device to 0." << std::endl;

  // 2. 注册 CUPTI 活动记录回调函数
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferAlloc, bufferComplete));

  // 3. 启用 CUPTI Kernel 活动收集
  // CUPTI_ACTIVITY_KIND_KERNEL 收集核函数执行信息
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  std::cout << "CUPTI Kernel Activity collection enabled." << std::endl;

  // 4. 启动 CUDA 核函数以生成活动记录
  std::cout << "Launching simple_kernel 5 times..." << std::endl;
  for (int i = 0; i < 5; ++i) {
    kernel(100, 10);
  }

  // 确保所有 CUDA 操作完成，以便 CUPTI 能够收集到所有记录
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaDeviceSynchronize failed: "
              << cudaGetErrorString(cudaStatus) << std::endl;
    return 1;
  }
  std::cout << "All kernels launched and synchronized." << std::endl;

  // 5. 禁用 CUPTI 活动收集
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
  std::cout << "CUPTI Kernel Activity collection disabled." << std::endl;

  // 6. 刷新所有剩余的活动记录
  // 这将强制 CUPTI 调用 bufferFree 回调来处理所有未处理的记录
  std::cout << "Flushing remaining CUPTI activity records..." << std::endl;
  CUPTI_CALL(cuptiActivityFlushAll(0)); // 0 表示没有超时

  std::cout << "CUPTI Kernel Activity Demo finished successfully." << std::endl;

  return 0;
}
