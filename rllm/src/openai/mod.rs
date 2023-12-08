use std::sync::{Arc, Mutex};

use candle_core::Device;
use tokenizers::{EncodeInput, Encoding, Tokenizer};

use self::{pipelines::ModulePipeline, responses::APIError};

pub mod requests;
pub mod responses;
pub mod sampling_params;
mod streaming;

pub trait TokenizerWrapper<'s, E>
where
    E: Into<EncodeInput<'s>>,
{
    fn tokenize(&self, input: E) -> Result<Encoding, APIError>;
}

impl<'s, E> TokenizerWrapper<'s, E> for Tokenizer
where
    E: Into<EncodeInput<'s>>,
{
    fn tokenize(&self, input: E) -> Result<Encoding, APIError> {
        self.encode(input, false)
            .map_err(|x| APIError::new(x.to_string()))
    }
}

#[derive(Clone)]
pub struct PipelineConfig {
    pub max_model_len: usize,
}

#[derive(Clone)]
pub struct OpenAIServerData<'s> {
    pub model: Arc<Mutex<Box<dyn ModulePipeline<'s>>>>,
    pub pipeline_config: PipelineConfig,
    pub device: Device,
}

pub mod conversation;
pub mod models;
pub mod openai_server;
pub mod pipelines;
pub mod utils;
