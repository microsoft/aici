use actix_web::error;
use derive_more::{Display, Error};

use serde::{Deserialize, Serialize};

#[derive(Debug, Display, Error, Serialize)]
#[display(fmt = "Error: {}", data)]
pub struct APIError {
    data: String,
}

impl error::ResponseError for APIError {}

impl APIError {
    pub fn new(data: String) -> Self {
        Self { data }
    }

    pub fn new_str(data: &str) -> Self {
        Self {
            data: data.to_string(),
        }
    }

    pub fn from<T: ToString>(value: T) -> Self {
        //panic!("{}", value.to_string());
        Self::new(value.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionUsageResponse {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

// tool_calls, function_call not supported!
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoiceData {
    pub content: Option<String>,
    pub role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub message: ChatChoiceData,
    pub finish_reason: Option<String>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<ChatChoice>,
    pub created: u64,
    pub model: String,
    pub object: &'static str,
    pub usage: ChatCompletionUsageResponse,
}

// tool_calls, function_call not supported!
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChoiceData {
    pub content: Option<String>,
    pub role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChoice {
    pub delta: StreamingChoiceData,
    pub finish_reason: Option<String>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChatCompletionResponse {
    pub id: String,
    pub choices: Vec<StreamingChoice>,
    pub created: u64,
    pub model: String,
    pub object: &'static str,
}
