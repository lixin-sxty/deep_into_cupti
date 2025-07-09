#include <cstdlib>        // For malloc, free, exit
#include <cuda_runtime.h> // CUDA 运行时 API
#include <cupti.h>        // CUPTI API
#include <iostream>       // 暂时保留，但实际输出将由 spdlog 接管
#include <unistd.h>

#include "cupti_activity.h"
#include "kernel.h" // 包含 simple_kernel 的声明

#include "utils/utils.h"

// CUPTI 活动记录缓冲区分配回调函数
void bufferAlloc(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  SPDLOG_INFO("CUPTI allocate buffer.");
  *size = 1024 * 1024; // 1MB
  // *size = 1024; // 1KB
  *buffer = (uint8_t *)malloc(*size);
  if (!*buffer) {
    SPDLOG_ERROR("Failed to allocate CUPTI activity buffer.");
    exit(EXIT_FAILURE);
  }
  *maxNumRecords = 0;
}

// CUPTI 活动记录缓冲区释放和处理回调函数
void bufferComplete(CUcontext context, uint32_t streamId, uint8_t *buffer,
                    size_t size, size_t validSize) {
  SPDLOG_INFO(
      "CUPTI buffer complete, streamId: {}, buffer size: {}, valid size: {}",
      streamId, size, validSize);
  CUpti_Activity *record = NULL;
  size_t kernel_records_count = 0;
  CUptiResult status;
  const char *errstr;

  while (true) {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      switch (record->kind) {
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      case CUPTI_ACTIVITY_KIND_KERNEL: {
        CUpti_ActivityKernel7 *kernel_record = (CUpti_ActivityKernel7 *)record;
        SPDLOG_DEBUG("----------------------------------------");
        SPDLOG_DEBUG("Kernel Activity Record:");
        SPDLOG_DEBUG("  kernel name: {}",
                     kernel_record->name ? kernel_record->name : "unknown");
        SPDLOG_DEBUG("  Correlation ID: {}",
                     uint32_t(kernel_record->correlationId));
        SPDLOG_DEBUG("  Device ID: {}", uint32_t(kernel_record->deviceId));
        SPDLOG_DEBUG("  Context ID: {}", uint32_t(kernel_record->contextId));
        SPDLOG_DEBUG("  Stream ID: {}", uint32_t(kernel_record->streamId));
        SPDLOG_DEBUG("  Grid: ({}, {}, {})", int32_t(kernel_record->gridX),
                     int32_t(kernel_record->gridY),
                     int32_t(kernel_record->gridZ));
        SPDLOG_DEBUG("  Block: ({}, {}, {})", int32_t(kernel_record->blockX),
                     int32_t(kernel_record->blockY),
                     int32_t(kernel_record->blockZ));
        SPDLOG_DEBUG("  Static Shared Memory: {} bytes",
                     int32_t(kernel_record->staticSharedMemory));
        SPDLOG_DEBUG("  Dynamic Shared Memory: {} bytes",
                     int32_t(kernel_record->dynamicSharedMemory));
        SPDLOG_DEBUG("  Registers Per Thread: {}",
                     uint16_t(kernel_record->registersPerThread));
        SPDLOG_DEBUG("  Start Time: {} ns", uint64_t(kernel_record->start));
        SPDLOG_DEBUG("  End Time: {} ns", uint64_t(kernel_record->end));
        SPDLOG_DEBUG("  Duration: {} ns",
                     (kernel_record->end - kernel_record->start));
        kernel_records_count++;
        break;
      }
      case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
        CUpti_ActivityExternalCorrelation *external_correlation_record =
            (CUpti_ActivityExternalCorrelation *)record;
        SPDLOG_DEBUG("----------------------------------------");
        SPDLOG_DEBUG("External Correlation Activity Record:");
        SPDLOG_DEBUG("  kind: {}", (int)external_correlation_record->kind);
        SPDLOG_DEBUG("  external kind: {}",
                     (int)external_correlation_record->externalKind);
        SPDLOG_DEBUG("  correlation ID: {}",
                     uint32_t(external_correlation_record->correlationId));
        //            externalId
        SPDLOG_DEBUG("externalId: {}",
                     uint32_t(external_correlation_record->externalId));

        break;
      }
      case CUPTI_ACTIVITY_KIND_DRIVER: {
        CUpti_ActivityAPI *runtime_record = (CUpti_ActivityAPI *)record;
        SPDLOG_DEBUG("----------------------------------------");
        SPDLOG_DEBUG("Driver Activity Record:");
        SPDLOG_DEBUG(" cbid: {}", uint32_t(runtime_record->cbid));
        SPDLOG_DEBUG(" correlationId: {}",
                     uint32_t(runtime_record->correlationId));
        break;
      }
      case CUPTI_ACTIVITY_KIND_RUNTIME: {
        CUpti_ActivityAPI *runtime_record = (CUpti_ActivityAPI *)record;
        SPDLOG_DEBUG("----------------------------------------");
        SPDLOG_DEBUG("Runtime Activity Record:");
        SPDLOG_DEBUG(" cbid: {}", uint32_t(runtime_record->cbid));
        SPDLOG_DEBUG(" correlationId: {}",
                     uint32_t(runtime_record->correlationId));
        break;
      }
      default:
        break;
      }
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      cuptiGetResultString(status, &errstr);
      SPDLOG_ERROR("CUPTI ActivityGetNextRecord Error: {}", errstr);
      break;
    }
  }

  size_t dropped;
  status = cuptiActivityGetNumDroppedRecords(context, streamId, &dropped);
  if (status == CUPTI_SUCCESS && dropped > 0) {
    SPDLOG_WARN("Dropped {} activity records!", dropped);
  } else if (status != CUPTI_SUCCESS) {
    cuptiGetResultString(status, &errstr);
    SPDLOG_ERROR("CUPTI GetNumDroppedRecords Error: {}", errstr);
  }

  SPDLOG_INFO("### CUPTI Activity Buffer Summary ###");
  SPDLOG_INFO("Processed {} kernel activity records.", kernel_records_count);
  SPDLOG_INFO("Total records in buffer (reported by CUPTI): {} bytes",
              validSize);
  SPDLOG_INFO("#####################################");

  free(buffer);
}

int main() {
  init_spdlog();

  SPDLOG_INFO("Starting CUPTI Kernel Activity Demo...");

  int deviceCount;
  CUDA_CALL(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    SPDLOG_ERROR("No CUDA devices found.");
    return 1;
  }
  CUDA_CALL(cudaSetDevice(0));
  SPDLOG_INFO("Set CUDA device to 0.");

  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferAlloc, bufferComplete));

  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  SPDLOG_INFO("CUPTI Kernel Activity collection enabled.");

  SPDLOG_DEBUG("Flushing remaining CUPTI activity records...");

  for (int i = 0; i < 10; ++i) {
    CUPTI_CALL(cuptiActivityPushExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, i + 10));
    CUPTI_CALL(cuptiActivityPushExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, i + 10));
    kernel(100, 1);
    CUPTI_CALL(cuptiActivityPopExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, nullptr));
    CUPTI_CALL(cuptiActivityPopExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, nullptr));
  }

  CUDA_CALL(cudaDeviceSynchronize());
  SPDLOG_INFO("All kernels launched and synchronized.");

  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  SPDLOG_INFO("CUPTI Kernel Activity collection disabled.");

  SPDLOG_INFO("Flushing remaining CUPTI activity records...");
  CUPTI_CALL(cuptiActivityFlushAll(0));

  SPDLOG_INFO("CUPTI Kernel Activity Demo finished successfully.");

  spdlog::shutdown(); // 关闭 spdlog

  return 0;
}
