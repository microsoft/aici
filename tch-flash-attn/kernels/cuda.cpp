#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>

#define PROTECT(call)                                                          \
  try {                                                                        \
    call;                                                                      \
    return nullptr;                                                            \
  } catch (const std::exception &e) {                                          \
    return strdup(e.what());                                                   \
  }

struct Stats {
  int64_t current;
  int64_t peak;
  int64_t allocated;
  int64_t freed;
};

struct CudaProps {
  int32_t major;
  int32_t minor;
  int32_t multi_processor_count;
  int32_t max_threads_per_multi_processor;
  int64_t total_memory;
};

extern "C" {

char *cuda_reset_peak_memory_stats_C(int device) {
  PROTECT(c10::cuda::CUDACachingAllocator::resetPeakStats(device));
}

char *cuda_empty_cache_C() {
  PROTECT(c10::cuda::CUDACachingAllocator::emptyCache());
}

char *cuda_get_stats_allocated_bytes_C(int device, Stats *outp) {
  PROTECT({
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    outp->current = stats.allocated_bytes[0].current;
    outp->peak = stats.allocated_bytes[0].peak;
    outp->allocated = stats.allocated_bytes[0].allocated;
    outp->freed = stats.allocated_bytes[0].freed;
  });
}

char *cuda_get_device_properties_C(int64_t device, CudaProps *outp) {
  PROTECT({
    auto p = at::cuda::getDeviceProperties(device);
    outp->major = p->major;
    outp->minor = p->minor;
    outp->multi_processor_count = p->multiProcessorCount;
    outp->max_threads_per_multi_processor = p->maxThreadsPerMultiProcessor;
    outp->total_memory = p->totalGlobalMem;
  });
}

} // extern "C"