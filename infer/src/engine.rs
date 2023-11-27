use anyhow::{anyhow, Error as E, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{
    api::sync::{Api, ApiRepo},
    RepoType,
};
use std::{collections::HashSet, fmt::Display, path::PathBuf};
use tokenizers::Tokenizer;

use candle_transformers::models::llama as llama_ref;

use crate::seq::{BatchInfo, SeqId, Sequence, StepType};
use crate::{
    cache_engine::CacheEngine,
    config::{CacheConfig, ModelConfig, ParallelConfig, RllmConfig, SchedulerConfig},
};
use crate::{llama, rtrace, set_trace, LogitsProcessor};
use crate::{
    llama::{Llama, LlamaConfig},
    LoaderArgs,
};

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

pub enum Model {
    Llama(Llama),
    Reference(llama_ref::Llama),
}

impl Model {
    pub fn forward(&self, info: &BatchInfo) -> Result<Tensor> {
        match self {
            Model::Llama(llama) => Ok(llama.forward(info)?),
            Model::Reference(llama) => {
                let index_pos = info.positions.i(0..1)?.to_vec1::<i64>()?[0];
                let input = info.tokens.unsqueeze(0)?;
                Ok(llama.forward(&input, index_pos as usize)?)
            }
        }
    }
}

pub struct RllmEngine {
    pub tokenizer: Tokenizer,
    pub model: Model,
    seq_id: SeqId,
    cache: Option<llama::Cache>,
    #[allow(dead_code)]
    pub alt: usize,
    pub device: Device,
    pub eos_token_id: u32,
}

impl RllmEngine {
    pub fn load(args: LoaderArgs) -> Result<RllmEngine> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        let repo = Repo::from(&args)?;
        println!("loading the model weights from {}", repo);

        let tokenizer_filename = repo.get("tokenizer.json")?;

        let json_config: LlamaConfig = serde_json::from_slice(&repo.read("config.json")?)?;
        let model_config: ModelConfig = json_config.into_config();

        let mut rllm_config = RllmConfig {
            model: model_config.clone(),
            parallel: ParallelConfig::single(),
            cache: CacheConfig::default(),
            scheduler: SchedulerConfig::new(2560, 256, model_config.max_sequence_length),
            dtype,
            device: device.clone(),
        };

        // TODO infer these
        let elt_size = CacheEngine::get_cache_block_size(&rllm_config);
        let cache_mem = 4 << 30; // 4GiB
        rllm_config.cache.num_cpu_blocks = Some(cache_mem / elt_size);
        rllm_config.cache.num_gpu_blocks = Some(cache_mem / elt_size);

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

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

        let eos_token_id = tokenizer
            .token_to_id("</s>")
            .ok_or(anyhow!("</s> not found"))?;

        let (model, cache) = if args.use_reference {
            let config: llama_ref::LlamaConfig =
                serde_json::from_slice(&repo.read("config.json")?)?;
            let use_flash_attn = true;
            let config = config.into_config(use_flash_attn);
            let use_kv_cache = true;
            let cache = llama_ref::Cache::new(use_kv_cache, dtype, &config, &device)?;
            let llama = llama_ref::Llama::load(vb, &cache, &config)?;
            (Model::Reference(llama), None)
        } else {
            let cache = llama::Cache::new(dtype, &model_config, &device)?;
            let llama = Llama::load(vb, &cache, &model_config)?;
            (Model::Llama(llama), Some(cache))
        };

        Ok(RllmEngine {
            tokenizer,
            model,
            cache,
            seq_id: 1,
            device,
            eos_token_id,
            alt: args.alt,
        })
    }

    pub fn new_seq(&mut self, prompt: &str) -> Result<Sequence> {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let seq = Sequence::new(self.seq_id, &tokens, 16);
        self.seq_id += 1;
        Ok(seq)
    }

    pub fn decode_seq(&self, seq: &Sequence) -> Result<String> {
        let tokens = &seq.tokens[seq.prompt_len..];
        let generated = self
            .tokenizer
            .decode(tokens, true)
            .map_err(anyhow::Error::msg)?;
        Ok(generated)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        sample_len: usize,
        logits_processor: &mut LogitsProcessor,
    ) -> Result<String> {
        self.cache.as_ref().map(|x| x.clear());

        let trace = false;

        let seq = self.new_seq(prompt)?;
        rtrace!("seq: {:?}", seq);
        let mut seqs = vec![seq];
        // seqs.push(self.new_seq(prompt)?);
        // seqs.push(self.new_seq(prompt)?);

        if self.alt == 1 {
            set_trace(trace);
            let off = seqs[0].tokens.len() / 2;
            let rest = seqs[0].tokens.drain(off..).collect::<Vec<_>>();
            seqs[0].prompt_len = seqs[0].tokens.len();

            let info0 = BatchInfo::from_seqs(&seqs, &self.device)?;
            let _ = self.model.forward(&info0)?;

            seqs[0].step_type = StepType::Fixed(rest.len());
            seqs[0].tokens.extend(rest);
            seqs[0].prompt_len = seqs[0].tokens.len();
        }

        set_trace(trace);

        for _idx in 0..sample_len {
            let info = BatchInfo::from_seqs(&seqs, &self.device)?;
            rtrace!("batch_info #{_idx}: {:?}", info);
            let logits = self.model.forward(&info)?;
            rtrace!("logits: {}", logits);
            for idx in 0..seqs.len() {
                let logits = logits.i((idx, ..))?;
                let next_token = logits_processor.sample(&logits)?;
                seqs[idx].tokens.push(next_token);
                seqs[idx].step_type = StepType::Gen;
                // if next_token == self.eos_token_id {
                //     break;
                // }
            }
        }

        Ok(self.decode_seq(&seqs[0])?)
    }
}
