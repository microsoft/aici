use crate::TensorOps;

pub mod blocks;
pub mod loader;
pub mod tmodel;
pub mod seqid;

pub type Model = llama_cpp_low::Model;

#[derive(Clone)]
pub struct Tensor {
    ptr: *mut f32,
    size: usize,
}

impl TensorOps for Tensor {
    fn to_vec1(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }
}

impl Tensor {
    pub fn from_slice(slice: &'static [f32]) -> Self {
        Tensor {
            ptr: slice.as_ptr() as *mut f32,
            size: slice.len(),
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}
