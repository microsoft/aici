use crate::{check_res, CUDAStream, Stream};
use std::{
    os::raw::{c_char, c_int},
    sync::Arc,
};

#[repr(C)]
pub(crate) struct CUDAEvent {
    priv_: [u8; 0],
}

extern "C" {
    fn cuda_event_create_C(
        timing: c_int,
        blocking: c_int,
        cu_ev: *mut *mut CUDAEvent,
    ) -> *mut c_char;
    fn cuda_event_record_C(cu_ev: *mut CUDAEvent, cu_str: *mut CUDAStream) -> *mut c_char;
    fn cuda_event_block_C(cu_ev: *mut CUDAEvent, cu_str: *mut CUDAStream) -> *mut c_char;
    fn cuda_event_elapsed_time_C(
        cu_ev: *mut CUDAEvent,
        cu_ev2: *mut CUDAEvent,
        elapsed: *mut f32,
    ) -> *mut c_char;
    fn cuda_event_query_C(cu_ev: *mut CUDAEvent, done: *mut c_int) -> *mut c_char;
    fn cuda_event_synchronize_C(cu_ev: *mut CUDAEvent) -> *mut c_char;
    fn cuda_event_free_C(cu_ev: *mut CUDAEvent) -> *mut c_char;
}

struct EventInner {
    ptr: *mut CUDAEvent,
}

#[derive(Clone)]
pub struct Event {
    cu_ev: Arc<EventInner>,
}

impl Event {
    /// Create a new event, where .synchronize() yields and .elapsed_time() is not available.
    pub fn new() -> Self {
        Self::create(false, false)
    }

    /// Create a new event, where .synchronize() will busy-wait (instead of yielding).
    pub fn new_blocking() -> Self {
        Self::create(false, true)
    }

    /// Create a new event that can be used for elapsed_time().
    pub fn new_timing() -> Self {
        Self::create(true, false)
    }

    fn create(timing: bool, blocking: bool) -> Self {
        let mut cu_ev: *mut CUDAEvent = std::ptr::null_mut();
        unsafe {
            check_res(
                "cuda_event_create_C",
                cuda_event_create_C(
                    if timing { 1 } else { 0 },
                    if blocking { 1 } else { 0 },
                    &mut cu_ev,
                ),
            )
        };
        Self {
            cu_ev: Arc::new(EventInner { ptr: cu_ev }),
        }
    }

    /// Captures in event the contents of `stream` at the time of this call.
    /// Has to be called always with the same `stream`.
    /// Calls to `.query()` and `.block()` then wait for the completion of the work
    /// captured here.
    pub fn record(&mut self, stream: &Stream) {
        unsafe {
            check_res(
                "cuda_event_record_C",
                cuda_event_record_C(self.cu_ev.ptr, stream.cu_str),
            )
        };
    }

    /// Makes all future work submitted to the given stream wait for this event.
    /// Does not block the CPU.
    /// Note: cudaStreamWaitEvent must be called on the same device as the stream.
    pub fn wait(&mut self, stream: &Stream) {
        unsafe {
            check_res(
                "cuda_event_block_C",
                cuda_event_block_C(self.cu_ev.ptr, stream.cu_str),
            )
        };
    }

    /// Compute time in milliseconds between the two events.
    pub fn elapsed_time(&self, end_event: &Event) -> f32 {
        let mut elapsed = 0.0;
        unsafe {
            check_res(
                "cuda_event_elapsed_time_C",
                cuda_event_elapsed_time_C(self.cu_ev.ptr, end_event.cu_ev.ptr, &mut elapsed),
            )
        };
        // TODO we could do Duration but the time can be negative I guess
        elapsed
    }

    /// Returns true if the event has completed.
    pub fn query(&self) -> bool {
        let mut done = 0;
        unsafe {
            check_res(
                "cuda_event_query_C",
                cuda_event_query_C(self.cu_ev.ptr, &mut done),
            );
        }
        done != 0
    }

    /// Block the CPU (yield or busy wait, depending on creation flags) until the current event completes.
    /// Can be called from any stream.
    pub fn synchronize(&self) {
        unsafe {
            check_res(
                "cuda_event_synchronize_C",
                cuda_event_synchronize_C(self.cu_ev.ptr),
            )
        };
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { check_res("cuda_event_free_C", cuda_event_free_C(self.cu_ev.ptr)) };
    }
}
