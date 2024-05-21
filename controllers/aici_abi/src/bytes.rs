use std::mem::size_of;

use anyhow::{anyhow, Result};
use bytemuck::{NoUninit, Pod};
use bytemuck_derive::{Pod, Zeroable};

pub(crate) type TokenId = u32;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct TokRxInfo {
    pub vocab_size: u32,
    pub tok_eos: TokenId,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct U32Pair(pub u32, pub u32);

pub fn clone_vec_as_bytes<T: NoUninit>(input: &[T]) -> Vec<u8> {
    bytemuck::cast_slice(input).to_vec()
}

pub fn vec_from_bytes<T: Pod>(bytes: &[u8]) -> Vec<T> {
    if bytes.len() % size_of::<T>() != 0 {
        panic!(
            "vecT: got {} bytes, needed multiple of {}",
            bytes.len(),
            size_of::<T>()
        );
    }
    bytemuck::cast_slice(bytes).to_vec()
}

pub fn limit_str(s: &str, max_len: usize) -> String {
    limit_bytes(s.as_bytes(), max_len)
}

pub fn limit_bytes(s: &[u8], max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", String::from_utf8_lossy(&s[0..max_len]))
    } else {
        String::from_utf8_lossy(s).to_string()
    }
}

pub fn to_hex_string(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join("")
}

pub fn from_hex_string(s: &str) -> Result<Vec<u8>> {
    let mut result = Vec::with_capacity(s.len() / 2);
    let mut iter = s.chars();
    while let Some(c1) = iter.next() {
        let c2 = iter
            .next()
            .ok_or_else(|| anyhow!("expecting even number of chars"))?;
        let byte = u8::from_str_radix(&format!("{}{}", c1, c2), 16)?;
        result.push(byte);
    }
    Ok(result)
}
