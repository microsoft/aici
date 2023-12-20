#[cfg(not(feature = "cuda"))]
pub use crate::llm::refkernels::*;
use tch::{Device, Tensor};
#[cfg(feature = "cuda")]
pub use tch_flash_attn::flash_attn_varlen as varlen_attn;
#[cfg(feature = "cuda")]
pub use tch_flash_attn::*;

/// Convert a vector of lengths into a tensor of offsets, as expected by flash attn.
pub fn to_offsets(seqlens: impl Iterator<Item = usize>, device: Device) -> (usize, Tensor) {
    let mut offsets = Vec::new();
    let mut offset = 0;
    let mut max = 0;
    for len in seqlens {
        max = std::cmp::max(len, max);
        offsets.push(offset as i32);
        offset += len;
    }
    offsets.push(offset as i32);
    (max, Tensor::from_slice(offsets.as_slice()).to(device))
}
