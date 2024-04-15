use crate::HashMap;
use aici_abi::{ProcessResultOffset, StorageCmd, TokenId};
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
    /// This is the number of bytes in the bias/sample_mask tensor.
    pub mask_num_bytes: usize,
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
