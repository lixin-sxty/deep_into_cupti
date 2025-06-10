#include <cstdlib>        // For malloc, free, exit
#include <cuda_runtime.h> // CUDA 运行时 API
#include <cupti.h>        // CUPTI API
#include <iostream>       // 暂时保留，但实际输出将由 spdlog 接管

#include "kernel.h" // 包含 simple_kernel 的声明

// spdlog 头文件
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

// CUPTI 错误检查宏
#define CUPTI_CALL(call)                                                       \
  do {                                                                         \
    CUptiResult _status = call;                                                \
    if (_status != CUPTI_SUCCESS) {                                            \
      const char *errstr;                                                      \
      cuptiGetResultString(_status, &errstr);                                  \
      spdlog::error("CUPTI Error: {} in {}:{}", errstr, __FILE__, __LINE__);   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// CUPTI 活动记录缓冲区分配回调函数
void bufferAlloc(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  *size = 1024 * 1024; // 1MB
  *buffer = (uint8_t *)malloc(*size);
  if (!*buffer) {
    spdlog::error("Failed to allocate CUPTI activity buffer.");
    exit(EXIT_FAILURE);
  }
  *maxNumRecords = 0;
}

// CUPTI 活动记录缓冲区释放和处理回调函数
void bufferComplete(CUcontext context, uint32_t streamId, uint8_t *buffer,
                    size_t size, size_t validSize) {
  CUpti_Activity *record = NULL;
  size_t kernel_records_count = 0;
  CUptiResult status;
  const char *errstr;

  while (true) {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      switch (record->kind) {
      case CUPTI_ACTIVITY_KIND_KERNEL: {
        CUpti_ActivityKernel *kernel_record = (CUpti_ActivityKernel *)record;
        spdlog::info("----------------------------------------");
        spdlog::info("Kernel Activity Record:");
        spdlog::info("  Name: {}", std::string(kernel_record->name));
        spdlog::info("  Correlation ID: {}",
                     uint32_t(kernel_record->correlationId));
        spdlog::info("  Device ID: {}", uint32_t(kernel_record->deviceId));
        spdlog::info("  Context ID: {}", uint32_t(kernel_record->contextId));
        spdlog::info("  Stream ID: {}", uint32_t(kernel_record->streamId));
        spdlog::info("  Grid: ({}, {}, {})", int32_t(kernel_record->gridX),
                     int32_t(kernel_record->gridY),
                     int32_t(kernel_record->gridZ));
        spdlog::info("  Block: ({}, {}, {})", int32_t(kernel_record->blockX),
                     int32_t(kernel_record->blockY),
                     int32_t(kernel_record->blockZ));
        spdlog::info("  Static Shared Memory: {} bytes",
                     int32_t(kernel_record->staticSharedMemory));
        spdlog::info("  Dynamic Shared Memory: {} bytes",
                     int32_t(kernel_record->dynamicSharedMemory));
        spdlog::info("  Registers Per Thread: {}",
                     uint16_t(kernel_record->registersPerThread));
        spdlog::info("  Start Time: {} ns", uint64_t(kernel_record->start));
        spdlog::info("  End Time: {} ns", uint64_t(kernel_record->end));
        spdlog::info("  Duration: {} ns",
                     (kernel_record->end - kernel_record->start));
        kernel_records_count++;
        break;
      }
      default:
        break;
      }
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      cuptiGetResultString(status, &errstr);
      spdlog::error("CUPTI ActivityGetNextRecord Error: {}", errstr);
      break;
    }
  }

  size_t dropped;
  status = cuptiActivityGetNumDroppedRecords(context, streamId, &dropped);
  if (status == CUPTI_SUCCESS && dropped > 0) {
    SPDLOG_WARN("Dropped {} activity records!", dropped);
  } else if (status != CUPTI_SUCCESS) {
    cuptiGetResultString(status, &errstr);
    spdlog::error("CUPTI GetNumDroppedRecords Error: {}", errstr);
  }

  spdlog::info("\n--- CUPTI Activity Buffer Summary ---");
  spdlog::info("Processed {} kernel activity records.", kernel_records_count);
  spdlog::info("Total records in buffer (reported by CUPTI): {} bytes",
               validSize);
  spdlog::info("-------------------------------------\n");

  free(buffer);
}

int main() {
  // 初始化 spdlog
  auto console = spdlog::stdout_color_mt("console");
  spdlog::set_default_logger(console);
  spdlog::set_level(spdlog::level::info); // 设置日志级别为 INFO

  spdlog::info("Starting CUPTI Kernel Activity Demo...");

  int deviceCount;
  cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
  if (cudaStatus != cudaSuccess) {
    spdlog::error("cudaGetDeviceCount failed: {}",
                  cudaGetErrorString(cudaStatus));
    return 1;
  }
  if (deviceCount == 0) {
    spdlog::error("No CUDA devices found.");
    return 1;
  }
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    spdlog::error("cudaSetDevice failed: {}", cudaGetErrorString(cudaStatus));
    return 1;
  }
  spdlog::info("Set CUDA device to 0.");

  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferAlloc, bufferComplete));

  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  spdlog::info("CUPTI Kernel Activity collection enabled.");

  spdlog::info("Launching simple_kernel 5 times...");
  for (int i = 0; i < 5; ++i) {
    kernel(100, 10);
  }

  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    spdlog::error("cudaDeviceSynchronize failed: {}",
                  cudaGetErrorString(cudaStatus));
    return 1;
  }
  spdlog::info("All kernels launched and synchronized.");

  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
  spdlog::info("CUPTI Kernel Activity collection disabled.");

  spdlog::info("Flushing remaining CUPTI activity records...");
  CUPTI_CALL(cuptiActivityFlushAll(0));

  spdlog::info("CUPTI Kernel Activity Demo finished successfully.");

  spdlog::shutdown(); // 关闭 spdlog

  return 0;
}
