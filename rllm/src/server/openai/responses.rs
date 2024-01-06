use actix_web::error;
use aici_abi::StorageCmd;
use aicirt::WasmError;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

#[derive(Debug)]
pub struct APIError {
    code: actix_web::http::StatusCode,
    msg: String,
}

impl From<anyhow::Error> for APIError {
    fn from(e: anyhow::Error) -> Self {
        Self::from_anyhow(e)
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for APIError {
    fn from(e: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Self::from_anyhow(anyhow::anyhow!(e))
    }
}

impl std::error::Error for APIError {}
impl error::ResponseError for APIError {
    fn status_code(&self) -> actix_web::http::StatusCode {
        self.code
    }
}

impl Display for APIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "APIError: {}", self.msg)
    }
}

impl APIError {
    pub fn new(data: String) -> Self {
        Self {
            code: actix_web::http::StatusCode::BAD_REQUEST,
            msg: data,
        }
    }

    pub fn new_str(data: &str) -> Self {
        Self::new(data.to_string())
    }

    pub fn from_anyhow(value: anyhow::Error) -> Self {
        if WasmError::is_self(&value) {
            log::info!("WasmError: {value}");
            Self {
                code: actix_web::http::StatusCode::BAD_REQUEST,
                msg: format!("{value}"),
            }
        } else {
            log::warn!("APIError: {value:?}");
            Self {
                code: actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
                msg: format!("{value:?}"),
            }
        }
    }

    pub fn just_msg(value: anyhow::Error) -> Self {
        log::info!("Error: {value}");
        Self {
            code: actix_web::http::StatusCode::BAD_REQUEST,
            msg: format!("{value}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionUsageResponse {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    pub fuel_tokens: usize,
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

    pub error: String,
    pub logs: String,
    pub storage: Vec<StorageCmd>,
    // pub logprobs: Option<LogProbs>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingCompletionResponse {
    pub object: &'static str, // "text_completion"
    pub id: String,
    pub model: String,
    pub created: u64,
    pub choices: Vec<StreamingCompletionChoice>,
    pub usage: ChatCompletionUsageResponse,
}
