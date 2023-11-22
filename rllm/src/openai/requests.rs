use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub temperature: Option<f32>,  //0.7
    pub top_p: Option<f32>,        //1.0
    pub n: Option<usize>,          //1
    pub max_tokens: Option<usize>, //None
    pub stop: Option<StopTokens>,
    pub stream: Option<bool>,                     //false
    pub presence_penalty: Option<f32>,            //0.0
    pub frequency_penalty: Option<f32>,           //0.0
    pub logit_bias: Option<HashMap<String, f32>>, //None
    pub user: Option<String>,                     //None
    //Additional candle-vllm params
    pub top_k: Option<isize>,               //-1
    pub best_of: Option<usize>,             //None
    pub use_beam_search: Option<bool>,      //false
    pub ignore_eos: Option<bool>,           //false
    pub skip_special_tokens: Option<bool>,  //false
    pub stop_token_ids: Option<Vec<usize>>, //[]
}
