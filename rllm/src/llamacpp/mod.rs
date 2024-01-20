pub mod blocks;
pub mod loader;
pub mod tmodel;
pub mod util;

#[derive(Clone)]
pub struct Tensor {
    ptr: *mut f32,
    size: usize,
}

impl Tensor {
    pub fn from_slice(slice: &'static [f32]) -> Self {
        Tensor {
            ptr: slice.as_ptr() as *mut f32,
            size: slice.len(),
        }
    }

    pub fn to_kind(&self, kind: DType) -> Self {
        assert!(kind == DType::Float);
        self.clone()
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

pub struct BlockRef {}

impl BlockRef {
    pub fn get_index(&self) -> usize {
        0
    }
    pub fn fork(&self) -> Self {
        BlockRef {}
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    Uint8,
    Int8,
    Int16,
    Int,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    BFloat16,
}

impl DType {
    pub fn elt_size_in_bytes(self) -> usize {
        match self {
            DType::Uint8 => 1,
            DType::Int8 => 1,
            DType::Int16 => 2,
            DType::Int => 4,
            DType::Int64 => 8,
            DType::Half => 2,
            DType::Float => 4,
            DType::Double => 8,
            DType::ComplexHalf => 4,
            DType::ComplexFloat => 8,
            DType::ComplexDouble => 16,
            DType::Bool => 1,
            DType::QInt8 => 1,
            DType::QUInt8 => 1,
            DType::QInt32 => 4,
            DType::BFloat16 => 2,
        }
    }
}

/// A torch device.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    /// The main CPU device.
    Cpu,
    /// The main GPU device.
    Cuda(usize),
    /// The main MPS device.
    Mps,
    /// The main Vulkan device.
    Vulkan,
}
