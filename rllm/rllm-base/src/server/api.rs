use aici_abi::StorageCmd;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRequest {
    pub controller: String,
    pub controller_arg: serde_json::Value,
    pub prompt: String,
    pub temperature: Option<f32>,  // defl 0.0
    pub top_p: Option<f32>,        // defl 1.0
    pub top_k: Option<isize>,      // defl -1
    pub max_tokens: Option<usize>, // defl context size
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunUsageResponse {
    pub sampled_tokens: usize,
    pub ff_tokens: usize,
    pub cost: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialRunResponse {
    pub id: String,
    pub object: &'static str, // "initial-run"
    pub created: u64,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResponse {
    pub object: &'static str, // "run"
    pub forks: Vec<RunForkResponse>,
    pub usage: RunUsageResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunForkResponse {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    pub text: String,
    pub error: String,
    pub logs: String,
    pub storage: Vec<StorageCmd>,
    pub micros: u64,
}
