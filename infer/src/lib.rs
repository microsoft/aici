pub mod llama;

use std::{collections::HashSet, fmt::Display, path::PathBuf};

use anyhow::{Error as E, Result};
use candle::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{
    api::sync::{Api, ApiRepo},
    RepoType,
};
use llama::{Llama, LlamaConfig};
use tokenizers::Tokenizer;

pub struct LoaderArgs {
    pub model_id: Option<String>,
    pub revision: Option<String>,
    pub local_weights: Option<String>,
}

enum Repo {
    Api(ApiRepo),
    Local(String),
}

impl Repo {
    fn from(args: &LoaderArgs) -> Result<Repo> {
        match &args.local_weights {
            Some(path) => Ok(Repo::Local(path.to_owned())),
            None => {
                let api = Api::new()?;
                let model_id = args
                    .model_id
                    .clone()
                    .unwrap_or_else(|| "NousResearch/Llama-2-7b-hf".to_string());
                let revision = args.revision.clone().unwrap_or("main".to_string());
                let api = api.repo(hf_hub::Repo::with_revision(
                    model_id,
                    RepoType::Model,
                    revision,
                ));
                Ok(Repo::Api(api))
            }
        }
    }

    fn get(&self, filename: &str) -> Result<PathBuf> {
        match self {
            Repo::Api(api) => api.get(filename).map_err(E::msg),
            Repo::Local(path) => Ok((path.to_owned() + filename).into()),
        }
    }

    fn read(&self, filename: &str) -> Result<Vec<u8>> {
        std::fs::read(self.get(filename)?).map_err(E::msg)
    }
}

impl Display for Repo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Repo::Api(api) => write!(f, "{}", api.url("")),
            Repo::Local(path) => write!(f, "{}", path),
        }
    }
}

pub fn load_llama(args: LoaderArgs) -> Result<(Tokenizer, Llama, Device)> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;

    let repo = Repo::from(&args)?;
    println!("loading the model weights from {}", repo);

    let tokenizer_filename = repo.get("tokenizer.json")?;

    let config: LlamaConfig = serde_json::from_slice(&repo.read("config.json")?)?;
    let use_flash_attn = true;
    let config = config.into_config(use_flash_attn);

    let st_index: serde_json::Value =
        serde_json::from_slice(&repo.read("model.safetensors.index.json")?)?;

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
        .map(|f| repo.get(f))
        .collect::<Result<Vec<_>>>()?;

    println!("building the model");
    let cache = llama::Cache::new(true, dtype, &config, &device)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;
    let llama = Llama::load(vb, &cache, &config)?;

    Ok((tokenizer, llama, device))
}
