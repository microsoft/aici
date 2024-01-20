use crate::Tensor;

pub fn to_vec1(t: &Tensor) -> Vec<f32> {
    t.as_slice().to_vec()
}
