use crate::check_res;
use std::os::raw::{c_char, c_int};
use tch::Device;

#[repr(C)]
pub(crate) struct CUDAStream {
    priv_: [u8; 0],
}

#[repr(C)]
enum StreamType {
    StrDefault = 0,
    StrCurrent = 1,
    StrHighPri = 2,
    StrLowPri = 3,
}

extern "C" {
    fn cuda_stream_get_C(typ: StreamType, device: c_int, outp: *mut *mut CUDAStream)
        -> *mut c_char;
    fn cuda_stream_free_C(cu_str: *mut CUDAStream) -> *mut c_char;
    fn cuda_stream_clone_C(cu_str: *mut CUDAStream, outp: *mut *mut CUDAStream) -> *mut c_char;
    fn cuda_stream_query_C(cu_str: *mut CUDAStream, done: *mut c_int) -> *mut c_char;
    fn cuda_stream_synchronize_C(cu_str: *mut CUDAStream) -> *mut c_char;
    fn cuda_stream_set_current_C(cu_str: *mut CUDAStream) -> *mut c_char;
    fn cuda_stream_device_index_C(cu_str: *mut CUDAStream, id: *mut c_int) -> *mut c_char;
    fn cuda_stream_id_C(cu_str: *mut CUDAStream, id: *mut i64) -> *mut c_char;
}

pub struct CudaStream {
    pub(crate) cu_str: *mut CUDAStream,
}

impl Clone for CudaStream {
    fn clone(&self) -> Self {
        let mut cu_str: *mut CUDAStream = std::ptr::null_mut();
        unsafe {
            check_res(
                "cuda_stream_clone_C",
                cuda_stream_clone_C(self.cu_str, &mut cu_str),
            )
        };
        Self { cu_str }
    }
}

impl CudaStream {
    pub fn new(device: Device) -> Self {
        Self::create(StreamType::StrLowPri, device)
    }

    pub fn new_high_pri(device: Device) -> Self {
        Self::create(StreamType::StrHighPri, device)
    }

    pub fn default(device: Device) -> Self {
        Self::create(StreamType::StrDefault, device)
    }

    pub fn current(device: Device) -> Self {
        Self::create(StreamType::StrCurrent, device)
    }

    fn create(typ: StreamType, device: Device) -> Self {
        let mut cu_str: *mut CUDAStream = std::ptr::null_mut();
        match device {
            Device::Cuda(i) => unsafe {
                check_res(
                    "cuda_stream_get_C",
                    cuda_stream_get_C(typ, i as c_int, &mut cu_str),
                )
            },
            _ => panic!("only CUDA devices supported for streams"),
        };
        Self { cu_str }
    }

    pub fn synchronize(&self) {
        unsafe {
            check_res(
                "cuda_stream_synchronize_C",
                cuda_stream_synchronize_C(self.cu_str),
            )
        };
    }

    pub fn query(&self) -> bool {
        let mut done: c_int = 0;
        unsafe {
            check_res(
                "cuda_stream_query_C",
                cuda_stream_query_C(self.cu_str, &mut done),
            )
        };
        done != 0
    }

    pub fn set_current(&self) {
        unsafe {
            check_res(
                "cuda_stream_set_current_C",
                cuda_stream_set_current_C(self.cu_str),
            )
        };
    }

    pub fn device(&self) -> Device {
        let mut id: c_int = 0;
        unsafe {
            check_res(
                "cuda_stream_device_index_C",
                cuda_stream_device_index_C(self.cu_str, &mut id),
            )
        };
        Device::Cuda(id as usize)
    }

    pub fn id(&self) -> i64 {
        let mut id = 0;
        unsafe { check_res("cuda_stream_id_C", cuda_stream_id_C(self.cu_str, &mut id)) };
        id
    }

    pub fn guard(&self) -> StreamGuard {
        let prev = CudaStream::current(self.device());
        self.set_current();
        StreamGuard { prev }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe { check_res("cuda_stream_free_C", cuda_stream_free_C(self.cu_str)) };
    }
}

pub struct StreamGuard {
    prev: CudaStream,
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        self.prev.set_current();
    }
}
