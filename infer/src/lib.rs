use std::{collections::HashSet, fmt::Display, path::PathBuf};

use anyhow::{Error as E, Result};
use candle::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as model;
use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};
use model::{Llama, LlamaConfig};
use tokenizers::Tokenizer;

pub struct LoaderArgs {
    pub model_id: Option<String>,
    pub revision: Option<String>,
    pub local_weights: Option<String>,
}

enum AnyRepo {
    Api(ApiRepo),
    Local(String),
}

impl AnyRepo {
    fn from(args: &LoaderArgs) -> Result<AnyRepo> {
        match &args.local_weights {
            Some(path) => Ok(AnyRepo::Local(path.to_owned())),
            None => {
                let api = Api::new()?;
                let model_id = args
                    .model_id
                    .clone()
                    .unwrap_or_else(|| "NousResearch/Llama-2-7b-hf".to_string());
                let revision = args.revision.clone().unwrap_or("main".to_string());
                let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
                Ok(AnyRepo::Api(api))
            }
        }
    }

    fn get(&self, filename: &str) -> Result<PathBuf> {
        match self {
            AnyRepo::Api(api) => api.get(filename).map_err(E::msg),
            AnyRepo::Local(path) => Ok((path.to_owned() + filename).into()),
        }
    }

    fn read(&self, filename: &str) -> Result<Vec<u8>> {
        std::fs::read(self.get(filename)?).map_err(E::msg)
    }
}

impl Display for AnyRepo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnyRepo::Api(api) => write!(f, "{}", api.url("")),
            AnyRepo::Local(path) => write!(f, "{}", path),
        }
    }
}

pub fn load_llama(args: LoaderArgs) -> Result<(Tokenizer, Llama, Device)> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;

    let api = AnyRepo::from(&args)?;
    println!("loading the model weights from {}", api);

    let tokenizer_filename = api.get("tokenizer.json")?;

    let config: LlamaConfig = serde_json::from_slice(&api.read("config.json")?)?;
    let use_flash_attn = true;
    let config = config.into_config(use_flash_attn);

    let st_index: serde_json::Value =
        serde_json::from_slice(&api.read("model.safetensors.index.json")?)?;

    let entries = st_index["weight_map"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_str().unwrap().to_owned());

    let h = HashSet::<String>::from_iter(entries);
    let mut filenames = h.iter().collect::<Vec<_>>();
    filenames.sort();
    let filenames = filenames
        .iter()
        .map(|f| api.get(f))
        .collect::<Result<Vec<_>>>()?;

    println!("building the model");
    let cache = model::Cache::new(true, dtype, &config, &device)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;
    let llama = Llama::load(vb, &cache, &config)?;

    Ok((tokenizer, llama, device))
}
