use std::path::PathBuf;

use actix_web::web::Bytes;
use candle_core::{DType, Device};
use tokenizers::Encoding;
use tokio::sync::mpsc::Sender;

use super::{
    conversation::Conversation,
    responses::{APIError, ChatChoice, ChatCompletionUsageResponse},
    sampling_params::SamplingParams,
    streaming::SenderError,
    PipelineConfig, TokenizerWrapper,
};

pub mod llama;
pub mod mistral;

/// A module pipeline that encompasses the inference pass, tokenizer, and conversation.
pub trait ModulePipeline<'s>: Send + Sync {
    fn forward(
        &mut self,
        xs: &Encoding,
        sampling: SamplingParams,
        device: Device,
        streamer: Option<Sender<Result<Bytes, SenderError>>>,
    ) -> Result<(Option<Vec<ChatChoice>>, ChatCompletionUsageResponse), APIError>;

    fn name(&self) -> String;

    fn tokenizer(&self) -> &dyn TokenizerWrapper<'s, String>;

    fn get_conversation(&mut self) -> &mut dyn Conversation;
}

pub trait ModelPaths {
    fn get_weight_filenames(&self) -> &Vec<PathBuf>;
    fn get_config_filename(&self) -> &PathBuf;
    fn get_tokenizer_filename(&self) -> &PathBuf;
}

pub trait ModelLoader<'a> {
    fn download_model(
        &self,
        model_id: String,
        revision: Option<String>,
        hf_token: Option<String>,
    ) -> Result<Box<dyn ModelPaths>, APIError>;

    fn load_model(
        &self,
        paths: Box<dyn ModelPaths>,
        dtype: DType,
        device: Device,
    ) -> Result<(Box<dyn ModulePipeline<'a>>, PipelineConfig), APIError>;
}
