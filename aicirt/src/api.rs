use std::collections::HashMap;

use aici_abi::{StorageCmd, TokenId};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub type ModuleInstId = usize;

#[derive(Serialize, Deserialize)]
pub struct AiciPreProcessReq {
    pub max_context_len: usize, // in tokens
    pub freed: Vec<ModuleInstId>,
    pub ops: Vec<AiciPreOp>,
}

#[derive(Serialize, Deserialize)]
pub struct AiciPreProcessResp {
    pub seqs: HashMap<ModuleInstId, SequenceResult>,
    pub fork_map: Vec<usize>,
    pub suspend_ids: Vec<ModuleInstId>,
}

#[derive(Serialize, Deserialize)]
pub struct AiciMidProcessReq {
    pub ops: Vec<AiciMidOp>,
}

#[derive(Serialize, Deserialize)]
pub struct AiciMidProcessResp {
    pub seqs: HashMap<ModuleInstId, SequenceResult<AiciMidProcessResultInner>>,
    pub num_seqs: usize,
}

#[derive(Serialize, Deserialize)]
pub struct AiciMidProcessResultInner {
    pub ff_tokens: Vec<TokenId>,
    pub backtrack: u32,
}

#[derive(Serialize, Deserialize)]
pub struct AiciPostProcessReq {
    pub ops: Vec<AiciPostOp>,
}

#[derive(Serialize, Deserialize)]
pub struct AiciPostProcessResp {
    pub seqs: HashMap<ModuleInstId, SequenceResult<AiciPostProcessResultInner>>,
}

#[derive(Serialize, Deserialize)]
pub struct AiciPostProcessResultInner {
    pub stop: bool,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct AiciPreOp {
    pub id: ModuleInstId,
    pub req_id: Option<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct AiciMidOp {
    pub id: ModuleInstId,
    pub clone_id: Option<ModuleInstId>,
}

#[derive(Serialize, Deserialize)]
pub struct AiciPostOp {
    pub id: ModuleInstId,
    pub tokens: Vec<Token>,
    #[serde(default)]
    pub backtrack: u32,
    pub clone_id: Option<ModuleInstId>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SequenceResult<T = ()> {
    pub is_success: bool,
    pub logs: String,
    // StorageCmd::ReadVar are not recorded
    pub storage: Vec<StorageCmd>,
    pub micros: u64,
    pub result: Option<T>,
}

#[derive(Serialize, Deserialize)]
pub struct MkModuleReq {
    pub binary: String,
    #[serde(default)]
    pub meta: Value,
}

#[derive(Serialize, Deserialize)]
pub struct MkModuleResp {
    pub module_id: String,
    pub wasm_size: usize,
    pub meta_size: usize,
    pub compiled_size: usize,
    pub time: u64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct InstantiateReq {
    pub req_id: String,
    // [TokenId] or str
    pub prompt: Value,
    pub module_id: String,
    #[serde(default)]
    pub module_arg: Value,
}

pub type Token = TokenId;

#[derive(Serialize, Deserialize, Debug)]
pub struct TokensResp {
    pub vocab_size: u32,
}
