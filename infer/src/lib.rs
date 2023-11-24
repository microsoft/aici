use anyhow::Result;
use candle::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as model;
use hf_hub::{api::sync::Api, Repo, RepoType};
use model::{Llama, LlamaConfig};
use tokenizers::Tokenizer;

pub struct LoaderArgs {
    pub model_id: Option<String>,
    pub revision: Option<String>,
    pub local_weights: Option<String>,
}

pub fn load_llama(args: LoaderArgs) -> Result<(Tokenizer, Llama, Device)> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;

    let api = Api::new()?;
    let model_id = args
        .model_id
        .unwrap_or_else(|| "NousResearch/Llama-2-7b-hf".to_string());
    println!("loading the model weights from {model_id}");
    let revision = args.revision.unwrap_or("main".to_string());
    let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

    let tokenizer_filename = match &args.local_weights {
        Some(path) => (path.to_owned() + "tokenizer.json").into(),
        _ => api.get("tokenizer.json")?,
    };

    let config_filename = match &args.local_weights {
        Some(path) => (path.to_owned() + "config.json").into(),
        _ => api.get("config.json")?,
    };
    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let use_flash_attn = true;
    let config = config.into_config(use_flash_attn);

    let mut filenames = vec![];
    for rfilename in [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ] {
        match &args.local_weights {
            Some(path) => {
                filenames.push((path.to_owned() + rfilename).into());
            }
            _ => {
                let filename = api.get(rfilename)?;
                filenames.push(filename);
            }
        };
    }

    println!("building the model");
    let cache = model::Cache::new(true, dtype, &config, &device)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;
    let llama = Llama::load(vb, &cache, &config)?;

    Ok((tokenizer, llama, device))
}
