use self::responses::APIError;
use crate::InferenceWorker;
use std::sync::{Arc, Mutex};
use tokenizers::{Encoding, Tokenizer};

pub mod requests;
pub mod responses;
// pub mod sampling_params;
// mod streaming;

pub trait TokenizerWrapper {
    fn tokenize(&self, input: &str, add_special: bool) -> Result<Encoding, APIError>;
}

impl TokenizerWrapper for Tokenizer {
    fn tokenize(&self, input: &str, add_special: bool) -> Result<Encoding, APIError> {
        self.encode(input, add_special)
            .map_err(|x| APIError::new(x.to_string()))
    }
}

#[derive(Clone)]
pub struct PipelineConfig {
    pub max_model_len: usize,
}

#[derive(Clone)]
pub struct OpenAIServerData {
    pub worker: Arc<Mutex<InferenceWorker>>,
    pub pipeline_config: PipelineConfig,
    pub tokenizer: Arc<Tokenizer>,
}

// pub mod conversation;
// pub mod models;
pub mod openai_server;
// pub mod pipelines;
pub mod utils;
