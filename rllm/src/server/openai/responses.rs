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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub finish_reason: Option<String>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub choices: Vec<CompletionChoice>,
    pub created: u64,
    pub model: String,
    pub object: &'static str, // "text_completion"
    pub usage: ChatCompletionUsageResponse,
}

// tool_calls, function_call not supported!
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChoiceData {
    pub content: Option<String>,
    pub role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChatChoice {
    pub delta: StreamingChoiceData,
    pub finish_reason: Option<String>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChatCompletionResponse {
    pub id: String,
    pub choices: Vec<StreamingChatChoice>,
    pub created: u64,
    pub model: String,
    pub object: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct List<T> {
    pub object: &'static str, // "list"
    pub data: Vec<T>,
}

impl<T> List<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self {
            object: "list",
            data,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub object: &'static str, // "model"
    pub id: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingCompletionChoice {
    pub index: usize,
    pub finish_reason: Option<String>,
    pub text: String,
    // pub logprobs: Option<LogProbs>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingCompletionResponse {
    pub object: &'static str, // "text_completion"
    pub id: String,
    pub model: String,
    pub created: u64,
    pub choices: Vec<StreamingCompletionChoice>,
}