use anyhow::{anyhow, Error as E, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{
    api::sync::{Api, ApiRepo},
    RepoType,
};
use std::{collections::HashSet, fmt::Display, path::PathBuf, sync::Arc, time::Instant};
use tokenizers::Tokenizer;

use candle_transformers::models::llama as llama_ref;

use crate::{
    cache_engine::CacheEngine,
    config::{
        CacheConfig, ModelConfig, ParallelConfig, RllmConfig, SamplingParams, SchedulerConfig,
    },
    scheduler::SchedulerOutputs,
    seq::{FinishReason, RequestOutput, SchedulingPhase, SequenceGroup, Token},
};
use crate::{llama, LogitsProcessor};
use crate::{
    llama::{Llama, LlamaConfig},
    LoaderArgs,
};
use crate::{
    scheduler::Scheduler,
    seq::{BatchInfo, SeqId, Sequence, StepType},
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
    step_no: usize,
    #[allow(dead_code)]
    pub alt: usize,
    pub device: Device,
    pub eos_token_id: u32,

    scheduler: Scheduler,
}

impl RllmEngine {
    pub fn load(args: LoaderArgs) -> Result<RllmEngine> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        let repo = Repo::from(&args)?;
        log::info!("loading the model weights from {}", repo);

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

        log::info!("building the model");

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

        log::info!("model loaded");

        let scheduler = Scheduler::new(Arc::new(rllm_config));

        Ok(RllmEngine {
            tokenizer,
            model,
            cache,
            seq_id: 1,
            step_no: 0,
            device,
            eos_token_id,
            alt: args.alt,
            scheduler,
        })
    }

    pub fn add_request(
        &mut self,
        request_id: String,
        prompt: &str,
        sampling_params: SamplingParams,
    ) -> Result<()> {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let seq = Sequence::new(self.seq_id, &tokens, self.scheduler.config.cache.block_size);
        self.seq_id += 1;

        let logits_processor = LogitsProcessor::new(&sampling_params);
        let sg = SequenceGroup {
            request_id,
            seqs: vec![seq],
            sampling_params,
            arrival_time: Instant::now(),
            logits_processor,
        };

        self.scheduler.add_seq_group(sg);

        Ok(())
    }

    fn generate_outputs(
        &self,
        logits: &Tensor,
        sched_out: &mut SchedulerOutputs,
    ) -> Result<Vec<RequestOutput>> {
        let mut outputs = Vec::new();
        let mut idx = 0;

        for sg in sched_out.next_seq_groups.iter_mut() {
            let mut outp = RequestOutput {
                request_id: sg.request_id.clone(),
                seq_outputs: Vec::new(),
            };
            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase == SchedulingPhase::Running {
                    let logits = logits.i((idx, ..))?;
                    let next_token = sg.logits_processor.sample(&logits)?;
                    seq.tokens.push(next_token);
                    seq.step_type = StepType::Gen;
                    idx += 1;

                    if next_token == self.eos_token_id {
                        self.scheduler.finish_seq(seq, FinishReason::FoundEos);
                    } else if seq.get_gen_len() >= sg.sampling_params.max_tokens {
                        self.scheduler
                            .finish_seq(seq, FinishReason::MaxTokensReached);
                    }
                }
                outp.seq_outputs.push(seq.get_output());
            }
            outputs.push(outp);
        }

        Ok(outputs)
    }

    fn run_model(&mut self, sched_out: &mut SchedulerOutputs) -> Result<Vec<RequestOutput>> {
        if sched_out.is_empty() {
            log::debug!("no seqs to run");
            return Ok(Vec::new());
        }

        let seqs = sched_out
            .next_seq_groups
            .iter()
            .flat_map(|sg| sg.get_seqs(Some(SchedulingPhase::Running)));

        let info = BatchInfo::from_seqs(seqs, &self.device)?;

        log::trace!("batch_info #{}: {:?}", self.step_no, info);
        let logits = self.model.forward(&info)?;
        log::trace!("logits: {:?}", logits);

        self.generate_outputs(&logits, sched_out)
    }

    pub fn step(&mut self) -> Result<Vec<RequestOutput>> {
        self.step_no += 1;
        let mut sched_out = self.scheduler.schedule();
        log::trace!(
            "scheduled: {} groups, dropped: {}",
            sched_out.next_seq_groups.len(),
            sched_out.dropped_seq_groups.len()
        );
        let outputs = self.run_model(&mut sched_out);
        // we run step_finished() regardless if model failed
        self.scheduler.step_finished(sched_out);
        Ok(outputs?)
    }

    pub fn decode_seq(&self, tokens: &Vec<Token>) -> Result<String> {
        let generated = self
            .tokenizer
            .decode(tokens, true)
            .map_err(anyhow::Error::msg)?;
        Ok(generated)
    }

    pub fn generate(&mut self, prompt: &str, sampling_params: SamplingParams) -> Result<String> {
        self.cache.as_ref().map(|x| x.clear());

        let req_id = format!("R{}", self.step_no);
        self.add_request(req_id, prompt, sampling_params)?;

        let mut outputs = Vec::new();

        while self.scheduler.has_unfinished_seqs() {
            let outp = self.step()?;
            if !outp.is_empty() {
                assert!(outp.len() == 1);
                assert!(outp[0].seq_outputs.len() == 1);
                outputs = outp[0].seq_outputs[0].output_tokens.clone();
            }
        }

        Ok(self.decode_seq(&outputs)?)
    }
}
