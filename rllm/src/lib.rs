use clap::Subcommand;
use openai::pipelines::{
    llama::{LlamaLoader, LlamaSpecificConfig},
    mistral::{Mistral7BLoader, Mistral7BSpecificConfig},
    ModelLoader,
};

#[derive(Debug, Subcommand)]
pub enum ModelSelected {
    /// Select the llama model.
    Llama {
        #[arg(long)]
        no_kv_cache: bool,
        #[arg(long)]
        repeat_last_n: usize,
        #[arg(long)]
        use_flash_attn: bool,
    },

    /// Select the mistral model.
    Mistral {
        #[arg(long)]
        repeat_penalty: f32,
        #[arg(long)]
        repeat_last_n: usize,
        #[arg(long)]
        use_flash_attn: bool,
    },
}

pub fn get_model_loader<'a>(selected_model: ModelSelected) -> (Box<dyn ModelLoader<'a>>, String) {
    match selected_model {
        ModelSelected::Llama {
            no_kv_cache,
            repeat_last_n,
            use_flash_attn,
        } => (
            Box::new(LlamaLoader::new(LlamaSpecificConfig::new(
                no_kv_cache,
                repeat_last_n,
                use_flash_attn,
            ))),
            "meta-llama/Llama-2-7b-hf".to_string(),
        ),
        ModelSelected::Mistral {
            repeat_penalty,
            repeat_last_n,
            use_flash_attn,
        } => (
            Box::new(Mistral7BLoader::new(Mistral7BSpecificConfig::new(
                repeat_penalty,
                repeat_last_n,
                use_flash_attn,
            ))),
            "mistralai/Mistral-7B-v0.1".to_string(),
        ),
    }
}

pub mod openai;
