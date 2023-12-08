use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Messages {
    Map(Vec<HashMap<String, String>>),
    Literal(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StopTokens {
    Multi(Vec<String>),
    Single(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Messages,
    #[serde(default)]
    pub temperature: Option<f32>, //0.7
    #[serde(default)]
    pub top_p: Option<f32>, //1.0
    #[serde(default)]
    pub n: Option<usize>, //1
    #[serde(default)]
    pub max_tokens: Option<usize>, //None
    #[serde(default)]
    pub stop: Option<StopTokens>,
    #[serde(default)]
    pub stream: Option<bool>, //false
    #[serde(default)]
    pub presence_penalty: Option<f32>, //0.0
    #[serde(default)]
    pub frequency_penalty: Option<f32>, //0.0
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>, //None
    #[serde(default)]
    pub user: Option<String>, //None
    #[serde(default)]
    //Additional candle-vllm params
    pub top_k: Option<isize>, //-1
    #[serde(default)]
    pub best_of: Option<usize>, //None
    #[serde(default)]
    pub use_beam_search: Option<bool>, //false
    #[serde(default)]
    pub ignore_eos: Option<bool>, //false
    #[serde(default)]
    pub skip_special_tokens: Option<bool>, //false
    #[serde(default)]
    pub stop_token_ids: Option<Vec<usize>>, //[]
}
