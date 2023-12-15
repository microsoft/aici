use tch::{Device, Tensor};

#[cfg(not(feature = "cuda"))]
pub use crate::llm::refkernels::*;

#[cfg(feature = "cuda")]
pub use tch_flash_attn::*;
#[cfg(feature = "cuda")]
pub use tch_flash_attn::flash_attn_varlen as varlen_attn;

/// Convert a vector of lengths into a tensor of offsets, as expected by flash attn.
pub fn to_offsets(seqlens: &[usize], device: Device) -> (usize, Tensor) {
    let mut offsets = Vec::with_capacity(seqlens.len() + 1);
    let mut offset = 0;
    let mut max = 0;
    for len in seqlens {
        max = std::cmp::max(*len, max);
        offsets.push(offset as i32);
        offset += len;
    }
    offsets.push(offset as i32);
    (max, Tensor::from_slice(offsets.as_slice()).to(device))
}
