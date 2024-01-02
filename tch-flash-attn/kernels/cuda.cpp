#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

using namespace c10::cuda;
using namespace at::cuda;

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
  PROTECT(CUDACachingAllocator::resetPeakStats(device));
}

char *cuda_empty_cache_C() { PROTECT(CUDACachingAllocator::emptyCache()); }

char *cuda_get_stats_allocated_bytes_C(int device, Stats *outp) {
  PROTECT({
    auto stats = CUDACachingAllocator::getDeviceStats(device);
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

#define STR_DEFAULT 0
#define STR_CURRENT 1
#define STR_HIGH_PRI 2
#define STR_LOW_PRI 3

char *cuda_stream_get_C(int typ, int device, CUDAStream **outp) {
  PROTECT({
    auto r = typ == STR_DEFAULT    ? getDefaultCUDAStream(device)
             : typ == STR_CURRENT  ? getCurrentCUDAStream(device)
             : typ == STR_HIGH_PRI ? getStreamFromPool(true, device)
                                   : getStreamFromPool(false, device);
    *outp = new CUDAStream(r);
  });
}

char *cuda_stream_free_C(CUDAStream *cuStr) {
  PROTECT({ delete cuStr; });
}

char *cuda_stream_clone_C(CUDAStream *cuStr, CUDAStream **outp) {
  PROTECT({
    // streams are never destroyed on the torch side
    *outp = new CUDAStream(*cuStr);
  });
}

char *cuda_stream_query_C(CUDAStream *cuStr, int *done) {
  PROTECT({ *done = cuStr->query() ? 1 : 0; });
}

char *cuda_stream_synchronize_C(CUDAStream *cuStr) {
  PROTECT({ cuStr->synchronize(); });
}

char *cuda_stream_set_current_C(CUDAStream *cuStr) {
  PROTECT({ setCurrentCUDAStream(*cuStr); });
}

char *cuda_stream_device_index_C(CUDAStream *cuStr, int *id) {
  PROTECT({ *id = cuStr->device_index(); });
}

char *cuda_stream_id_C(CUDAStream *cuStr, int64_t *id) {
  PROTECT({ *id = cuStr->id(); });
}

//
// Events
//

char *cuda_event_create_C(bool timing, bool blocking, CUDAEvent **cuEv) {
  PROTECT({
    int flags = !timing ? cudaEventDisableTiming : cudaEventDefault;
    if (blocking)
      flags |= cudaEventBlockingSync;
    *cuEv = new CUDAEvent(flags);
  });
}

// Note: cudaEventRecord must be called on the same device as the event.
char *cuda_event_record_C(CUDAEvent *cuEv, CUDAStream *cuStr) {
  PROTECT({ cuEv->record(*cuStr); });
}

// Note: cudaStreamWaitEvent must be called on the same device as the stream.
char *cuda_event_block_C(CUDAEvent *cuEv, CUDAStream *cuStr) {
  PROTECT({ cuEv->block(*cuStr); });
}

char *cuda_event_elapsed_time_C(CUDAEvent *cuEv, CUDAEvent *cuEv2,
                                float *elapsed) {
  PROTECT({ *elapsed = cuEv->elapsed_time(*cuEv2); });
}

// Note: cudaEventQuery can be safely called from any device
char *cuda_event_query_C(CUDAEvent *cuEv, int *done) {
  PROTECT({ *done = cuEv->query() ? 1 : 0; });
}

// Note: cudaEventSynchronize can be safely called from any device
char *cuda_event_synchronize_C(CUDAEvent *cuEv) {
  PROTECT({ cuEv->synchronize(); });
}

char *cuda_event_free_C(CUDAEvent *cuEv) {
  PROTECT({ delete cuEv; });
}

} // extern "C"