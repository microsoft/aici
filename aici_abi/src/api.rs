use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::TokenId;

pub type ModuleInstId = usize;

#[derive(Serialize, Deserialize)]
pub struct AiciPreProcessReq {
    pub max_context_len: usize, // in tokens
    pub freed: Vec<ModuleInstId>,
    pub ops: Vec<AiciPreOp>,
}

#[derive(Serialize, Deserialize)]
pub struct AiciProcessReq {
    pub ops: Vec<AiciMidOp>,
}

#[derive(Serialize, Deserialize)]
pub struct AiciPostProcessReq {
    pub ops: Vec<AiciPostOp>,
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

#[derive(Serialize, Deserialize)]
pub struct MkModuleReq {
    pub binary: String,
    #[serde(default)]
    pub meta: Value,
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

#[derive(Serialize, Deserialize)]
pub struct SpecialTokenIds {
    pub bos: Option<Token>,
    pub eos: Option<Token>,
    pub unk: Option<Token>,
    pub sep: Option<Token>,
    pub pad: Option<Token>,
    pub cls: Option<Token>,
}

#[derive(Serialize, Deserialize)]
pub struct TokensReq {
    pub tokens: Vec<String>,
    pub special: SpecialTokenIds,
}

