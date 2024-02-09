use tch::Device;

pub struct CudaEvent {}

impl CudaEvent {
    pub fn new() -> Self {
        CudaEvent {}
    }

    pub fn wait(&self, _stream: &CudaStream) {}
}

pub struct CudaStream {}

impl CudaStream {
    pub fn new(_device: Device) -> Self {
        CudaStream {}
    }

    pub fn current(_device: Device) -> Self {
        CudaStream {}
    }
}
