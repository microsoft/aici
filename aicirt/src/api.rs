use crate::{shm::ShmAllocator, HashMap};
use aici_abi::{ProcessResultOffset, StorageCmd, TokenId};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub type ModuleInstId = usize;

#[derive(Serialize, Deserialize, Clone)]
pub struct InferenceCapabilities {
    #[serde(default)]
    pub backtrack: bool,
    #[serde(default)]
    pub ff_tokens: bool,
    #[serde(default)]
    pub fork: bool,
}

#[derive(Serialize, Deserialize)]
pub struct AiciMidProcessReq {
    pub ops: Vec<AiciMidOp>,
    pub freed: Vec<ModuleInstId>,
}

#[derive(Serialize, Deserialize)]
pub struct AiciMidProcessResp {
    pub seqs: HashMap<ModuleInstId, SequenceResult<ProcessResultOffset>>,
    pub dtype: String,
    pub first_mask_byte_offset: usize,
    pub mask_num_bytes: usize,
    pub mask_num_elts: usize,
    pub num_masks: usize,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
/// At most one of clone_id or req_id should be set.
/// If either of them is set, then id should be fresh.
pub struct AiciMidOp {
    pub id: ModuleInstId,
    /// Set to None, except upon first call for a branch after forking.
    pub clone_id: Option<ModuleInstId>,
    /// This is index of branch, set iff clone_id is set.
    pub clone_idx: Option<usize>,
    /// Set to None, except upon first call for a user request.
    pub req_id: Option<String>,
    /// Sampling result for the previous iteration.
    /// For simple sampled token 't', backtrack==0 and tokens==[t].
    /// For first request, backtrack==0 and tokens==[] (prompt is passed separetely, before).
    /// Can be more complex when splices are used.
    pub backtrack: u32,
    pub tokens: Vec<Token>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SequenceResult<T = ()> {
    pub result: Option<T>,
    pub error: String,
    // StorageCmd::ReadVar are not recorded
    pub storage: Vec<StorageCmd>,
    pub logs: String,
    pub micros: u64,
}

impl<T> SequenceResult<T> {
    pub fn from_error(error: String) -> SequenceResult<T> {
        SequenceResult {
            logs: error.clone(),
            error,
            result: None,
            storage: vec![],
            micros: 0,
        }
    }
    pub fn clone_with<S>(&self, result: Option<S>) -> SequenceResult<S> {
        SequenceResult {
            error: self.error.clone(),
            result,
            storage: self.storage.clone(),
            logs: self.logs.clone(),
            micros: self.micros,
        }
    }
    pub fn map_result<S, F>(self, f: F) -> SequenceResult<S>
    where
        F: FnOnce(T) -> S,
    {
        SequenceResult {
            error: self.error,
            result: self.result.map(f),
            storage: self.storage,
            logs: self.logs,
            micros: self.micros,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct MkModuleReq {
    pub binary: String,
}

#[derive(Serialize, Deserialize)]
pub struct MkModuleResp {
    pub module_id: String,
    pub wasm_size: usize,
    pub compiled_size: usize,
    pub time: u64,
}

#[derive(Serialize, Deserialize)]
pub struct SetTagsReq {
    pub module_id: String,
    pub tags: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TagInfo {
    pub tag: String,
    pub module_id: String,
    pub updated_at: u64, // unix time
    pub updated_by: String,
    pub wasm_size: u64,
    pub compiled_size: u64,
}

#[derive(Serialize, Deserialize)]
pub struct GetTagsResp {
    pub tags: Vec<TagInfo>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct InstantiateReq {
    pub req_id: String,
    // [TokenId] or str
    pub prompt: Value,
    pub module_id: String, // or tag name
    #[serde(default)]
    pub module_arg: Value,
}

pub type Token = TokenId;

#[derive(Serialize, Deserialize, Debug)]
pub struct TokensResp {
    pub vocab_size: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AuthInfo {
    pub user: String,
    pub is_admin: bool,
}

impl AuthInfo {
    pub fn local_user() -> Self {
        AuthInfo {
            user: "local".to_string(),
            is_admin: false,
        }
    }

    pub fn admin_user() -> Self {
        AuthInfo {
            user: "admin".to_string(),
            is_admin: true,
        }
    }
}

pub enum BiasType {
    F32,
    F16,
    BF16,
    Bool,
}

impl BiasType {
    pub fn from_u32(v: u32) -> Result<Self> {
        match v {
            1 => Ok(BiasType::F32),
            2 => Ok(BiasType::F16),
            3 => Ok(BiasType::BF16),
            4 => Ok(BiasType::Bool),
            _ => Err(anyhow!("invalid BiasType")),
        }
    }

    pub fn to_u32(&self) -> u32 {
        match self {
            BiasType::F32 => 1,
            BiasType::F16 => 2,
            BiasType::BF16 => 3,
            BiasType::Bool => 4,
        }
    }

    pub fn elt_size(&self) -> Option<usize> {
        match self {
            BiasType::F32 => Some(4),
            BiasType::F16 => Some(2),
            BiasType::BF16 => Some(2),
            BiasType::Bool => None,
        }
    }

    pub fn bytes_to_elts(&self, bytes: usize) -> usize {
        match self {
            BiasType::Bool => bytes * 8,
            _ => bytes / self.elt_size().unwrap(),
        }
    }

    pub fn size_in_bytes(&self, vocab_size: usize) -> usize {
        match self {
            BiasType::Bool => 4 * (vocab_size + 31) / 32,
            _ => vocab_size * self.elt_size().unwrap(),
        }
    }

    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "f32" => Ok(BiasType::F32),
            "f16" => Ok(BiasType::F16),
            "bf16" => Ok(BiasType::BF16),
            "bool" => Ok(BiasType::Bool),
            _ => Err(anyhow!("invalid BiasType")),
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            BiasType::F32 => "f32".to_string(),
            BiasType::F16 => "f16".to_string(),
            BiasType::BF16 => "bf16".to_string(),
            BiasType::Bool => "bool".to_string(),
        }
    }

    const LOGIT_BIAS_ALLOW: f32 = 0.0;
    const LOGIT_BIAS_DISALLOW: f32 = -f32::INFINITY;
    const LOGIT_BIAS_ALLOW_F16: u16 = 0;
    const LOGIT_BIAS_DISALLOW_F16: u16 = 0b1111_1100_0000_0000; // -inf
    const LOGIT_BIAS_ALLOW_BF16: u16 = 0;
    const LOGIT_BIAS_DISALLOW_BF16: u16 = 0b1111_1111_1000_0000; // -inf

    pub fn apply_to_shm_allocator(&self, src: &[u8], shm: &ShmAllocator, off: usize) {
        let vocab_size = self.bytes_to_elts(shm.elt_size());
        assert!(src.len() * 8 <= vocab_size);
        match self {
            BiasType::F32 => apply_to_slice(
                src,
                &mut shm.slice_at_byte_offset::<f32>(off, vocab_size),
                Self::LOGIT_BIAS_ALLOW,
                Self::LOGIT_BIAS_DISALLOW,
            ),
            BiasType::F16 => apply_to_slice(
                src,
                &mut shm.slice_at_byte_offset::<u16>(off, vocab_size),
                Self::LOGIT_BIAS_ALLOW_F16,
                Self::LOGIT_BIAS_DISALLOW_F16,
            ),
            BiasType::BF16 => apply_to_slice(
                src,
                &mut shm.slice_at_byte_offset::<u16>(off, vocab_size),
                Self::LOGIT_BIAS_ALLOW_BF16,
                Self::LOGIT_BIAS_DISALLOW_BF16,
            ),
            BiasType::Bool => {
                let trg = shm.slice_at_byte_offset::<u8>(off, self.size_in_bytes(vocab_size));
                trg[0..src.len()].copy_from_slice(src);
            }
        }
    }
}

fn apply_to_slice<T: Copy>(src: &[u8], dst: &mut [T], allow: T, disallow: T) {
    let mut dp = 0;
    for idx in 0..src.len() {
        let sb = src[idx];
        for bit in 0..8 {
            let mask = 1 << bit;
            dst[dp] = if sb & mask != 0 { allow } else { disallow };
            dp += 1;
        }
    }
    while dp < dst.len() {
        dst[dp] = disallow;
        dp += 1;
    }
}
