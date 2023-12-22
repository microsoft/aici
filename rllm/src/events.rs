use cudarc::driver::{result, sys};

pub struct Event {
    event: sys::CUevent,
}

pub struct Stream {
    stream: sys::CUstream,
}

impl Event {
    pub fn new() -> Self {
        let event = result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
            .expect("failed to create event");
        Self { event }
    }

    pub fn new_blocking() -> Self {
        // TODO we should also disable timing here, but Rust won't let me
        let event = result::event::create(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC)
            .expect("failed to create event");
        Self { event }
    }

    pub fn record(&mut self, stream: &Stream) {
        unsafe { result::event::record(self.event, stream.stream) }
            .expect("failed to record event");
    }

    pub fn query(&self) -> bool {
        match unsafe { sys::cuEventQuery(self.event) } {
            sys::CUresult::CUDA_SUCCESS => true,
            sys::CUresult::CUDA_ERROR_NOT_READY => false,
            r => {
                r.result().unwrap();
                panic!()
            }
        }
    }

    pub fn completed(&self) -> bool {
        self.query()
    }

    pub fn synchronize(&self) {
        unsafe { sys::cuEventSynchronize(self.event).result().unwrap() };
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let event = std::mem::replace(&mut self.event, std::ptr::null_mut());
        if !event.is_null() {
            unsafe { result::event::destroy(event) }.unwrap();
        }
    }
}
