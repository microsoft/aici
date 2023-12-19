#include <c10/cuda/CUDACachingAllocator.h>

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

}