use crate::{
    config::{
        CacheConfig, CommonModelConfig, ModelConfig, ModelMeta, ModelType, ParallelConfig,
        RllmConfig, SamplingParams, SchedulerConfig,
    },
    iface::AiciRtIface,
    llm::{llama, phi},
    paged::{BatchInfo, BatchInfoBuilder, CacheEngine, CacheSize, Scheduler, SchedulerOutputs},
    seq::{
        AiciSampling, FinishReason, RequestOutput, SchedulingPhase, SeqId, SeqOutput, Sequence,
        SequenceGroup, Token, TokenUsage,
    },
    util::{
        get_setting, gpu_memory_size, gpu_peak_allocated_bytes, log_mem_stats, reset_mem_stats,
        scalar_tensor, synchronize, to_vec1, to_vec2,
    },
    DType, Device, IndexOp, LoaderArgs, LogitsProcessor, Tensor,
};
use aici_abi::toktree::TokTrie;
use aicirt::{
    api::{
        AiciMidOp, AiciMidProcessReq, AiciPostOp, AiciPostProcessReq, AiciPreOp, AiciPreProcessReq,
        ModuleInstId, SequenceResult,
    },
    with_timer, TimerRef, TimerSet,
};
use anyhow::{bail, Error as E, Result};
use hf_hub::{
    api::sync::{Api, ApiRepo},
    RepoType,
};
use safetensors::Dtype;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    path::PathBuf,
    rc::Rc,
    sync::Arc,
    time::Instant,
};
use tch::{nn::VarStore, Kind};
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct ExpectedToken {
    pub sampled: Token,
    pub prob_mass: f32,
    pub logits: Vec<(Token, f32)>,
    pub ff_section_len: usize, // typically 1 for non-ff
}

#[derive(Clone)]
pub struct ExpectedGeneration {
    pub prompt: Vec<Token>,
    pub output: Vec<ExpectedToken>,
}

fn read_tensor(s: &safetensors::SafeTensors, name: &str) -> Result<Tensor> {
    let view = s.tensor(name)?;
    let size: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();
    let kind: DType = kind_from_dt(view.dtype());
    // Using from_blob here instead of from_data_size avoids some unnecessary copy.
    let tensor = unsafe { Tensor::from_blob(view.data().as_ptr(), &size, &[], kind, Device::Cpu) };
    Ok(tensor)
}

fn kind_from_dt(dtype: Dtype) -> Kind {
    match dtype {
        Dtype::BOOL => Kind::Bool,
        Dtype::U8 => Kind::Uint8,
        Dtype::I8 => Kind::Int8,
        Dtype::I16 => Kind::Int16,
        Dtype::I32 => Kind::Int,
        Dtype::I64 => Kind::Int64,
        Dtype::BF16 => Kind::BFloat16,
        Dtype::F16 => Kind::Half,
        Dtype::F32 => Kind::Float,
        Dtype::F64 => Kind::Double,
        dtype => panic!("unsupported dtype {dtype:?}"),
    }
}

impl ExpectedGeneration {
    pub fn load(f: &PathBuf) -> Result<Self> {
        let fp = std::fs::File::open(f)?;
        let content = unsafe { memmap2::MmapOptions::new().map(&fp)? };
        let s = safetensors::SafeTensors::deserialize(&content)?;

        let prompt = to_vec1::<i32>(&read_tensor(&s, "prompt")?.to_kind(Kind::Int));
        let output = to_vec1::<i32>(&read_tensor(&s, "output")?.to_kind(Kind::Int));
        let prob_mass = to_vec1::<f32>(&read_tensor(&s, "prob_mass")?.to_kind(Kind::Float));
        let tokens = to_vec2::<i32>(&read_tensor(&s, "tokens")?.to_kind(Kind::Int));
        let logits = to_vec2::<f32>(&read_tensor(&s, "logits")?.to_kind(Kind::Float));

        let num_tokens = output.len();
        assert!(tokens.len() == num_tokens);
        assert!(logits.len() == num_tokens);
        assert!(prob_mass.len() == num_tokens);

        Ok(ExpectedGeneration {
            prompt: prompt.into_iter().map(|x| x as Token).collect(),
            output: (0..num_tokens)
                .map(|i| ExpectedToken {
                    sampled: output[i] as Token,
                    ff_section_len: 1,
                    prob_mass: prob_mass[i],
                    logits: tokens[i]
                        .iter()
                        .zip(logits[i].iter())
                        .map(|(t, p)| (*t as Token, *p))
                        .collect(),
                })
                .collect(),
        })
    }
}

pub struct AddRequest {
    pub request_id: String,
    pub prompt: Vec<Token>,
    pub sampling_params: SamplingParams,
    pub expected: Option<ExpectedGeneration>,
}

enum Repo {
    Api(ApiRepo),
    Local(String),
}

impl Repo {
    fn from(args: &LoaderArgs) -> Result<Repo> {
        match &args.local_weights {
            Some(path) => Ok(Repo::Local(path.to_owned() + "/")),
            None => {
                let api = Api::new()?;
                let model_id = args.model_id.clone();
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

    fn is_local(&self) -> bool {
        match self {
            Repo::Api(_) => false,
            Repo::Local(_) => true,
        }
    }

    fn get(&self, filename: &str) -> Result<PathBuf> {
        match self {
            Repo::Api(api) => api.get(filename).map_err(E::msg),
            Repo::Local(path) => {
                let p: PathBuf = (path.to_owned() + filename).into();
                if p.exists() {
                    Ok(p)
                } else {
                    bail!("file {p:?} doesn't exists")
                }
            }
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

pub trait RllmModel {
    fn forward(&self, batch_info: &mut BatchInfo) -> Tensor;
    fn finalize(&mut self) {}
}

pub trait RllmModelConfig {
    fn into_config(self, common: CommonModelConfig) -> ModelConfig;
}

fn load_one_config<T>(
    err: &mut String,
    args: &LoaderArgs,
    name: &str,
    bytes: &[u8],
) -> Option<ModelConfig>
where
    T: RllmModelConfig + serde::de::DeserializeOwned,
{
    let common = CommonModelConfig {
        meta: ModelMeta {
            id: args.model_id.clone(),
        },
        dtype: args.dtype,
        device: args.device.clone(),
    };
    let json = serde_json::from_slice::<T>(bytes);
    if let Ok(json) = json {
        Some(json.into_config(common))
    } else {
        *err += &format!("{name}: {}\n", json.err().unwrap());
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    pub free_gpu_blocks: usize,
    pub free_cpu_blocks: usize,
}

impl Stats {
    pub fn same_as(&self, other: &Self) -> bool {
        self.free_gpu_blocks == other.free_gpu_blocks
            && self.free_cpu_blocks == other.free_cpu_blocks
    }
}

pub struct RllmEngine {
    pub config: Arc<RllmConfig>,
    pub tokenizer: Arc<Tokenizer>,
    pub tok_trie: Arc<TokTrie>,
    pub model_id: String,
    pub model: Box<dyn RllmModel>,
    seq_id: SeqId,
    step_no: usize,
    pub profile_step_no: usize,
    req_id_cnt: usize,
    #[allow(dead_code)]
    pub alt: usize,
    pub device: Device,
    pub eos_token_id: Token,
    pub space_token_id: Token,
    pub nv_profile: bool,
    pub num_errors: usize,

    pub timers: TimerSet,

    tim_step: TimerRef,

    tim_aici_pre: TimerRef,
    tim_schedule: TimerRef,
    tim_aici_mid: TimerRef,
    tim_run_model: TimerRef,

    tim_model_fwd: TimerRef,
    tim_sample: TimerRef,

    tim_aici_bias: TimerRef,
    tim_logit_sample: TimerRef,
    tim_logit_sync: TimerRef,
    tim_aici_post: TimerRef,

    aicirt: Option<AiciRtIface>,

    cache_engine: CacheEngine,
    scheduler: Scheduler,
}

impl RllmEngine {
    pub fn load_tokenizer(args: &LoaderArgs) -> Result<(Tokenizer, TokTrie)> {
        let byte_tokenizer = aici_tokenizers::find_tokenizer(&args.tokenizer)?;
        let hf_bytes = byte_tokenizer.get_hf_bytes();
        // std::fs::write("toks.json", &hf_bytes).unwrap();
        let tokenizer = Tokenizer::from_bytes(&hf_bytes).expect("can't load hf tokenizer");
        let tokens = byte_tokenizer.token_bytes();
        let trie = TokTrie::from(&byte_tokenizer.tokrx_info(), &tokens);
        trie.check_against(&tokens);
        Ok((tokenizer, trie))
    }

    pub fn load_model_config(args: &LoaderArgs) -> Result<ModelConfig> {
        let repo = Repo::from(args)?;
        log::info!("loading the model from {}", repo);

        let bytes = repo.read("config.json")?;
        let mut err = String::new();

        let cfg = load_one_config::<llama::LlamaConfig>(&mut err, args, "llama", &bytes)
            .or_else(|| load_one_config::<phi::PhiConfig>(&mut err, args, "phi", &bytes));

        match cfg {
            Some(mut v) => {
                let tok = aici_tokenizers::find_tokenizer(&args.tokenizer)?;
                v.tok_vocab_size = tok.tokrx_info().vocab_size as usize;
                Ok(v)
            }
            None => bail!("failed to load model config:\n{}", err),
        }
    }

    fn load_model(rllm_config: &RllmConfig, filenames: Vec<PathBuf>) -> Result<Box<dyn RllmModel>> {
        let mut vs = VarStore::new(rllm_config.device.clone());

        let rc_cfg = Rc::new(rllm_config.model.clone());
        let mut model: Box<dyn RllmModel> = match rllm_config.model.model_type {
            ModelType::Llama => Box::new(llama::Llama::load(vs.root(), &rc_cfg).unwrap()),
            ModelType::Phi => {
                Box::new(phi::MixFormerSequentialForCausalLM::new(&rc_cfg, vs.root()))
            }
        };

        vs.set_kind(rllm_config.dtype);

        let mut vars = vs.variables();

        let bar = indicatif::ProgressBar::new(vars.len() as u64);
        bar.set_style(
            indicatif::ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:60.cyan/blue} {pos:>4}/{len:4} [{eta_precise}] {msg}",
            )
            .unwrap(),
        );

        for f in &filenames {
            let fp = std::fs::File::open(f)?;
            let content = unsafe { memmap2::MmapOptions::new().map(&fp)? };
            let safetensors = safetensors::SafeTensors::deserialize(&content)?;

            for vname in safetensors.names() {
                let target_name = vname.to_string();
                if !vars.contains_key(&target_name) {
                    if vname.ends_with(".inv_freq") {
                        // OK
                    } else {
                        log::warn!("variable {} not found in the model", target_name);
                    }
                    continue;
                }

                // Using from_blob here instead of from_data_size avoids some unnecessary copy.
                let src_tensor = read_tensor(&safetensors, vname)?;
                let mut var = vars.remove(&target_name).unwrap();
                assert!(var.size() == src_tensor.size());
                // println!("copying to {var:?} from {src_tensor:?}");
                var.f_copy_(&src_tensor)?;

                bar.inc(1);
                if bar.is_hidden() {
                    eprint!(".");
                }
            }
        }

        if vars.len() > 0 {
            bail!("{} variables not found in the model: {vars:?}", vars.len());
        }

        if bar.is_hidden() {
            eprintln!(" done");
        }
        bar.finish();

        log::info!("model loaded");

        model.finalize();

        Ok(model)
    }

    fn model_filenames(repo: &Repo) -> Result<Vec<PathBuf>> {
        let idx = repo.read("model.safetensors.index.json");

        let filenames = if let Ok(idx) = idx {
            let st_index: serde_json::Value = serde_json::from_slice(&idx)?;
            let entries = st_index["weight_map"]
                .as_object()
                .unwrap()
                .values()
                .map(|v| v.as_str().unwrap().to_owned());

            let h = HashSet::<String>::from_iter(entries);
            let mut filenames = h.into_iter().collect::<Vec<_>>();
            filenames.sort();
            filenames
        } else {
            if repo.is_local() && repo.get("model.safetensors-rust").is_ok() {
                vec!["model.safetensors-rust".to_string()]
            } else {
                vec!["model.safetensors".to_string()]
            }
        };

        let filenames = filenames
            .iter()
            .map(|f| repo.get(f))
            .collect::<Result<Vec<_>>>()?;

        Ok(filenames)
    }

    pub fn load(args: LoaderArgs) -> Result<RllmEngine> {
        let _no_grad = tch::no_grad_guard();

        let device = args.device;
        let repo = Repo::from(&args)?;

        let (tokenizer, tok_trie) = Self::load_tokenizer(&args)?;
        let model_config = Self::load_model_config(&args)?;
        let model_len = model_config.max_sequence_length;

        let mut aici = args.aici.clone();
        if aici.max_fuel == 0 {
            aici.max_fuel = model_len * 10;
        }

        let rllm_config = RllmConfig {
            model: model_config.clone(),
            parallel: ParallelConfig::single(),
            cache: CacheConfig::default(),
            scheduler: SchedulerConfig {
                max_num_batched_tokens: model_len,
                max_num_kv_tokens: model_len * 10,
                max_num_seqs: 100,
                max_model_len: model_len,
            },
            aici,
            dtype: model_config.dtype,
            device: device.clone(),
        };

        let filenames = Self::model_filenames(&repo)?;

        rllm_config.verify_args()?;

        log::info!("building the model");

        let eos_token_id = tok_trie.info().tok_eos;
        let space_token_id = tok_trie.greedy_tokenize(b" ")[0];

        let _ = Tensor::zeros(&[1], (model_config.dtype, device));
        reset_mem_stats(device);
        log_mem_stats("initial", device);

        let model = Self::load_model(&rllm_config, filenames)?;

        log_mem_stats("model fully loaded", device);

        let rllm_config = Arc::new(rllm_config);
        let cache_size = Self::profile_model(rllm_config.clone(), &model);

        let scheduler = Scheduler::new(rllm_config.clone(), &cache_size);
        let cache_engine = CacheEngine::new(rllm_config.clone(), &cache_size);

        let timers = TimerSet::new();

        Ok(RllmEngine {
            config: rllm_config,
            tokenizer: Arc::new(tokenizer),
            tok_trie: Arc::new(tok_trie),
            model_id: format!("{}", repo),
            model,
            seq_id: 1,
            step_no: 0,
            profile_step_no: 0,
            req_id_cnt: 0,
            num_errors: 0,
            device,
            eos_token_id,
            space_token_id,
            alt: args.alt,
            scheduler,
            cache_engine,
            nv_profile: false,
            aicirt: None,
            tim_step: timers.new_timer("step"),
            tim_aici_pre: timers.new_timer("step.aici_pre"),
            tim_schedule: timers.new_timer("step.schedule"),
            tim_aici_mid: timers.new_timer("step.aici_mid"),
            tim_run_model: timers.new_timer("step.run_model"),
            tim_model_fwd: timers.new_timer("step.run_model.model_fwd"),
            tim_sample: timers.new_timer("step.run_model.sample"),
            tim_aici_bias: timers.new_timer("step.run_model.sample.aici_bias"),
            tim_logit_sample: timers.new_timer("step.run_model.sample.sample"),
            tim_logit_sync: timers.new_timer("step.run_model.sample.sync"),
            tim_aici_post: timers.new_timer("step.run_model.sample.aici_post"),
            timers,
        })
    }

    fn profile_model(config: Arc<RllmConfig>, model: &Box<dyn RllmModel>) -> CacheSize {
        let mut info = BatchInfoBuilder::new(config.clone()).profile_run();
        let device = config.device.clone();
        reset_mem_stats(device);
        log_mem_stats("before model profile", device);
        let _logits = model.forward(&mut info);
        log_mem_stats("after model profile", device);

        let gpu_mem = gpu_memory_size(device);
        let gpu_cache_size = if gpu_mem > 0 {
            let frac = config.cache.gpu_memory_utilization;
            let peak = gpu_peak_allocated_bytes(device) as isize;
            let left = (gpu_mem as f64 * frac) as isize - peak;
            if left < 0 {
                panic!("not enough GPU memory for the cache: {gpu_mem} * {frac} < {peak}");
            }
            left as usize
        } else {
            512 << 20 // 512MiB
        };

        let max_cpu = 2 << 30; // 2GiB
        let cpu_cache_size = std::cmp::min(max_cpu, gpu_cache_size);

        let elt_size = CacheEngine::get_cache_block_size(&config);

        let r = CacheSize {
            cpu: cpu_cache_size / elt_size,
            gpu: gpu_cache_size / elt_size,
        };

        let token_kv_size = elt_size / config.cache.block_size;

        const G: f64 = 1024.0 * 1024.0 * 1024.0;
        log::info!(
            "caches: gpu:{:.3}GiB cpu:{:.3}GiB; blocks: {}/{}; tokens: {}/{}; {}KiB/token",
            gpu_cache_size as f64 / G,
            cpu_cache_size as f64 / G,
            r.gpu,
            r.cpu,
            r.gpu * config.cache.block_size,
            r.cpu * config.cache.block_size,
            token_kv_size / 1024,
        );

        r
    }

    pub fn set_aicirt(&mut self, aicirt: AiciRtIface) {
        self.aicirt = Some(aicirt);
    }

    pub fn gen_req_id(&mut self) -> String {
        self.req_id_cnt += 1;
        format!("_{}", self.req_id_cnt)
    }

    pub fn abort_request(&mut self, request_id: &str) {
        self.scheduler.abort_seq_group(request_id);
    }

    pub fn num_pending_requests(&self) -> usize {
        self.scheduler.get_num_unfinished_seq_groups()
    }

    pub fn tokenize(&self, text: &str, add_special_tokens: bool) -> Result<Vec<Token>> {
        let tokens = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(anyhow::Error::msg)?;
        Ok(tokens.get_ids().to_vec())
    }

    pub fn queue_request(&mut self, req: AddRequest) -> Result<()> {
        let mut seq = Sequence::new(
            self.seq_id,
            &req.prompt,
            self.scheduler.config.cache.block_size,
        );
        seq.expected = req.expected;
        seq.pending_fork_ids = (1..req.sampling_params.n)
            .map(|i| self.seq_id + i)
            .collect::<Vec<_>>();
        self.seq_id += req.sampling_params.n;

        let logits_processor = LogitsProcessor::new(&req.sampling_params, self.tok_trie.clone());
        let prompt = self
            .tokenizer
            .decode(&req.prompt, false)
            .map_err(anyhow::Error::msg)?;

        let sg = SequenceGroup {
            request_id: req.request_id,
            prompt,
            seqs: vec![seq],
            sampling_params: req.sampling_params,
            arrival_time: Instant::now(),
            logits_processor,
            max_index: 0,
            usage: TokenUsage::default(),
        };

        self.scheduler.add_seq_group(sg);

        Ok(())
    }

    pub fn add_expected_generation(
        &mut self,
        exp_gen: ExpectedGeneration,
        req_id: Option<String>,
    ) -> Result<()> {
        let request_id = req_id.unwrap_or_else(|| self.gen_req_id());
        self.queue_request(AddRequest {
            request_id,
            prompt: exp_gen.prompt.clone(),
            sampling_params: SamplingParams {
                max_tokens: exp_gen.output.len() + 1,
                ..SamplingParams::default()
            },
            expected: Some(exp_gen),
        })
    }

    pub fn add_request(
        &mut self,
        request_id: String,
        prompt: &str,
        sampling_params: SamplingParams,
    ) -> Result<()> {
        let tokens = self.tokenize(prompt, true)?;
        self.queue_request(AddRequest {
            request_id,
            prompt: tokens,
            sampling_params,
            expected: None,
        })
    }

    fn aici_bias(&mut self, sched_out: &mut SchedulerOutputs) -> Result<Tensor> {
        if self.aicirt.is_none() {
            return Ok(Tensor::zeros(&[0], (DType::Float, self.device)));
        }

        let mid_res = self.aicirt.as_mut().unwrap().finish_mid_process()?;
        let mut idx = 0;

        for sg in sched_out.next_seq_groups.iter_mut() {
            if sg.sampling_params.aici_module.is_none() {
                continue;
            }
            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase != SchedulingPhase::Running {
                    continue;
                }
                assert!(seq.has_aici);
                match self.save_aici_log(seq, &mid_res.seqs) {
                    Some(r) if r.ff_tokens.len() > 0 || r.backtrack > 0 => {
                        // save the computed prefix
                        // we may drop some of it but seq.splice_tokens() takes care of that
                        seq.num_kv_computed = seq.get_len();

                        seq.aici_sampling = AiciSampling::Splice {
                            // backtrack count includes the token that was supposed to be appended
                            // due to current sampling; however we never append it
                            backtrack: r.backtrack.saturating_sub(1),
                            ff_tokens: r.ff_tokens.clone(),
                        }
                    }
                    _ => {
                        seq.aici_sampling = AiciSampling::SampleWithBias { offset: idx };
                    }
                }
                idx += 1;
            }
        }

        assert!(idx == mid_res.num_seqs);

        let shm = &self.aicirt.as_mut().unwrap().bin_shm;
        let num_elts = mid_res.num_seqs * self.tok_trie.vocab_size();
        let slice = shm.slice_at_byte_offset::<f32>(0, num_elts);
        let t = Tensor::from_slice(slice)
            .to(self.device)
            .reshape(&[mid_res.num_seqs as i64, self.tok_trie.vocab_size() as i64]);
        Ok(t)
    }

    fn aici_apply_bias(
        &self,
        seq: &mut Sequence,
        logits: &mut Tensor,
        aici_bias: &Tensor,
    ) -> Option<AiciPostOp> {
        let sid = seq.seq_id;
        match std::mem::take(&mut seq.aici_sampling) {
            AiciSampling::Regular => None,
            AiciSampling::SampleWithBias { offset } => {
                log::trace!("sample *{sid}: bias at {offset}");
                let logits_aici = aici_bias.i((offset as i64, ..));
                *logits = &*logits + logits_aici;
                None
            }
            AiciSampling::Splice {
                backtrack,
                ff_tokens,
            } => {
                log::trace!("sample *{sid}: backtrack:{backtrack} ff_tokens:{ff_tokens:?}",);
                seq.splice_tokens(backtrack as usize, &ff_tokens);
                Some(AiciPostOp {
                    id: seq.seq_id,
                    tokens: ff_tokens,
                    backtrack: backtrack,
                    clone_id: None,
                })
            }
        }
    }

    fn check_expected(&mut self, logits: &Tensor, req_id: &str, seq: &mut Sequence) -> Token {
        let exp = seq.expected.as_ref().unwrap();
        let idx = seq.get_len() - exp.prompt.len();
        let next_token = if idx >= exp.output.len() {
            self.eos_token_id
        } else {
            let out = &exp.output[idx];
            let mut logits = to_vec1::<f32>(&logits.to_kind(Kind::Float));
            let mut max_err = 0.0;
            let mut sum_err = 0.0;
            let mut min_logit = f32::INFINITY;
            for (t, l_exp) in out.logits.iter() {
                let l_act = logits[*t as usize];
                let d = (l_act - l_exp).abs();
                sum_err += d;
                if d > max_err {
                    max_err = d;
                }
                if *l_exp < min_logit {
                    min_logit = *l_exp;
                }

                // log::debug!(
                //     "exp: {t} {tstr} {l_exp:.4} {l_act:.4} {d:.4}",
                //     tstr = self.tok_trie.token_dbg(*t),
                // );

                // zero it out for the "unmentioned" test below
                logits[*t as usize] = 0.0;
            }

            let max_allowed_err = get_setting("test_maxtol") as f32;
            let avg_allowed_err = get_setting("test_avgtol") as f32;
            let avg_err = sum_err / out.logits.len() as f32;
            log::debug!("exp #{idx} in {req_id}: avg_err:{avg_err:.4} max_err:{max_err:.4}");
            if max_err > max_allowed_err {
                log::error!("max error too large: {max_err} > {}", max_allowed_err);
                self.num_errors += 1;
            } else if avg_err > avg_allowed_err {
                log::error!("avg error too large: {avg_err} > {avg_allowed_err}");
                self.num_errors += 1;
            }

            let limit = min_logit + max_allowed_err;
            let l_act = logits.into_iter().max_by(f32::total_cmp).unwrap();
            if l_act > limit {
                log::error!("unmentioned entry too large: {l_act} > {limit}");
                self.num_errors += 1;
            }

            if out.ff_section_len > 1 {
                let mut toks = exp.output[idx..(idx + out.ff_section_len)]
                    .iter()
                    .map(|e| e.sampled)
                    .collect::<Vec<_>>();
                let r = toks.pop().unwrap();
                seq.append_tokens(&toks);
                r
            } else {
                out.sampled
            }
        };

        next_token
    }

    fn dropped_outputs(&mut self, sched_out: &mut SchedulerOutputs) -> Vec<RequestOutput> {
        sched_out
            .dropped_seq_groups
            .iter_mut()
            .map(|sg| self.req_output(sg, true))
            .collect()
    }

    fn empty_outputs(&mut self, sched_out: &mut SchedulerOutputs) -> Result<Vec<RequestOutput>> {
        let _ = self.aici_bias(sched_out)?;
        Ok(self.dropped_outputs(sched_out))
    }

    fn sample(
        &mut self,
        logits: &Tensor,
        sched_out: &mut SchedulerOutputs,
        info: &BatchInfo,
    ) -> Result<Vec<RequestOutput>> {
        let mut seq_id_to_logit_idx = info.seq_id_to_idx.clone();

        {
            let (num_seq, vocab_size) = logits.size2()?;
            let t_vocab = self.tok_trie.vocab_size() as i64;
            if vocab_size != t_vocab {
                panic!("vocab size mismatch: model {vocab_size} != tokenizer {t_vocab}");
            }
            assert!(num_seq == info.seq_id_to_idx.len() as i64);
        }

        for sg in sched_out.next_seq_groups.iter_mut() {
            let mut to_add = Vec::new();
            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase != SchedulingPhase::Running {
                    continue;
                }
                // get it even if no pending - make sure we have them all
                let trg_idx = seq_id_to_logit_idx[&seq.seq_id];
                let pending = std::mem::take(&mut seq.pending_fork_ids);
                for copy_id in pending {
                    seq_id_to_logit_idx.insert(copy_id, trg_idx);
                    let copy = seq.fork_as(copy_id, sg.max_index + 1);
                    sg.max_index += 1;
                    log::debug!("forked: {:?} -> {:?}", seq, copy);
                    to_add.push(copy);
                }
            }
            sg.seqs.extend(to_add);
        }

        let aici_bias = with_timer!(self.tim_aici_bias, self.aici_bias(sched_out)?);

        let mut post_ops = Vec::new();

        let mut pre_sample = HashMap::new();

        for sg in sched_out.next_seq_groups.iter_mut() {
            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase != SchedulingPhase::Running {
                    continue;
                }

                let idx = seq_id_to_logit_idx[&seq.seq_id];
                let mut logits = logits.i((idx as i64, ..));

                if let Some(op) = self.aici_apply_bias(seq, &mut logits, &aici_bias) {
                    post_ops.push(op);
                    continue;
                }

                seq.num_kv_computed = seq.get_len();

                let next_token = if seq.expected.is_some() {
                    let t = self.check_expected(&logits, &sg.request_id, seq);
                    scalar_tensor(t as i64, logits.device())
                } else {
                    with_timer!(self.tim_logit_sample, sg.logits_processor.sample(&logits)?)
                };

                pre_sample.insert(seq.seq_id, next_token);
            }
        }

        for sg in sched_out.next_seq_groups.iter_mut() {
            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase != SchedulingPhase::Running {
                    continue;
                }

                let next_token = match pre_sample.get(&seq.seq_id) {
                    Some(t) => with_timer!(self.tim_logit_sync, t.int64_value(&[]) as u32),
                    None => continue,
                };

                let mut info = "";
                if seq.has_aici && next_token == self.eos_token_id {
                    // replace with space, so the model doesn't get confused
                    // note that aici will still get the real EOS token
                    seq.append_tokens(&[self.space_token_id]);
                    info = " -> space";
                } else {
                    seq.append_tokens(&[next_token]);
                }

                if seq.has_aici {
                    post_ops.push(AiciPostOp {
                        id: seq.seq_id,
                        tokens: vec![next_token],
                        backtrack: 0,
                        clone_id: None,
                    });
                }

                log::trace!(
                    "sample *{}: {}{}",
                    seq.seq_id,
                    self.tok_trie.token_dbg(next_token),
                    info
                );

                if !sg.sampling_params.ignore_eos && next_token == self.eos_token_id {
                    self.scheduler.finish_seq(seq, FinishReason::FoundEos);
                } else if seq.get_gen_len() >= sg.sampling_params.max_tokens {
                    self.scheduler
                        .finish_seq(seq, FinishReason::MaxTokensReached);
                }
            }
        }

        with_timer!(
            self.tim_aici_post,
            self.aici_post(sched_out, AiciPostProcessReq { ops: post_ops })?
        );

        let mut outputs = self.dropped_outputs(sched_out);
        outputs.extend(
            sched_out
                .next_seq_groups
                .iter_mut()
                .map(|sg| self.req_output(sg, false)),
        );

        Ok(outputs)
    }

    fn req_output(&self, sg: &mut SequenceGroup, is_final: bool) -> RequestOutput {
        RequestOutput {
            request_id: sg.request_id.clone(),
            seq_outputs: sg
                .seqs
                .iter_mut()
                .map(|seq| {
                    let mut out = seq.gen_output();
                    out.new_text = self.tok_trie.decode_str(&out.new_output_tokens);
                    out
                })
                .collect(),
            usage: sg.usage.clone(),
            is_final,
        }
    }

    fn run_model(&mut self, sched_out: &mut SchedulerOutputs) -> Result<Vec<RequestOutput>> {
        if sched_out.is_empty() {
            log::debug!("no seqs to run");
            return self.empty_outputs(sched_out);
        }

        let iface = {
            self.cache_engine.new_round();
            if sched_out.blocks_to_swap_in.len() > 0 {
                self.cache_engine.swap_in(&sched_out.blocks_to_swap_in);
            }
            if sched_out.blocks_to_swap_out.len() > 0 {
                self.cache_engine.swap_out(&sched_out.blocks_to_swap_out);
            }
            if sched_out.blocks_to_copy.len() > 0 {
                self.cache_engine.copy(&sched_out.blocks_to_copy);
            }
            self.cache_engine.get_cache_iface()
        };

        let mut info = BatchInfoBuilder::new(self.config.clone())
            .sched_out(sched_out)
            .finish(self.step_no, iface);

        log::trace!("batch_info #{}: {:?}", info.step_no, info);
        // log::trace!("{}", info.positions);
        // log::trace!("{}", info.gather_mapping);
        // log::trace!("{}", info.slot_mapping);

        #[cfg(feature = "cuda")]
        if self.nv_profile {
            cudarc::driver::safe::profiler_start()?;
        }

        let t0 = Instant::now();
        let logits = with_timer!(self.tim_model_fwd, {
            let l = self.model.forward(&mut info);
            if false {
                // without this, the timing is off but we may get better perf
                synchronize(self.device);
            }
            l
        });
        let r = with_timer!(self.tim_sample, { self.sample(&logits, sched_out, &info) });
        let dur = t0.elapsed().as_micros() as f64 / 1000.0;

        log::info!(
            "model forward: step #{} {:.2}ms; {} tok(s); {:.1}tps",
            self.step_no,
            dur,
            info.tokens.numel(),
            info.tokens.numel() as f64 / (dur / 1000.0),
        );

        #[cfg(feature = "cuda")]
        if self.nv_profile {
            cudarc::driver::safe::profiler_stop()?;
        }

        info.save_log(&format!("step-{}.safetensor", self.step_no));
        log::trace!("logits: {:?}", logits);
        r
    }

    pub fn seq_output_text(&self, seq_output: &SeqOutput) -> Result<String> {
        let generated = self
            .tokenizer
            .decode(&seq_output.output_tokens, true)
            .map_err(anyhow::Error::msg)?;
        Ok(generated)
    }

    fn save_aici_log<'a, T>(
        &self,
        seq: &mut Sequence,
        seqs: &'a HashMap<ModuleInstId, SequenceResult<T>>,
    ) -> Option<&'a T> {
        if let Some(r) = seqs.get(&seq.seq_id) {
            seq.aici_logs.push(r.clone_with(None));
            if r.error.len() > 0 {
                self.scheduler.finish_seq(seq, FinishReason::Failed);
            }
            match &r.result {
                Some(r) => Some(r),
                None => None,
            }
        } else {
            None
        }
    }

    fn aici_post(
        &mut self,
        sched_out: &mut SchedulerOutputs,
        req: AiciPostProcessReq,
    ) -> Result<()> {
        if self.aicirt.is_none() {
            return Ok(());
        }

        let post_res = self.aicirt.as_mut().unwrap().post_process(req)?;

        for sg in sched_out.next_seq_groups.iter_mut() {
            if sg.sampling_params.aici_module.is_none() {
                continue;
            }

            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase != SchedulingPhase::Running {
                    continue;
                }

                match self.save_aici_log(seq, &post_res.seqs) {
                    Some(r) if r.stop => {
                        self.scheduler.finish_seq(seq, FinishReason::AiciStop);
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    fn aici_pre(&mut self) -> Result<()> {
        if self.aicirt.is_none() {
            return Ok(());
        }

        let mut max_context_len = 0;
        let mut ops = Vec::new();

        self.scheduler.for_each_sg(|sg| {
            if sg.sampling_params.aici_module.is_none() {
                return;
            }
            if sg.seqs.len() == 1 && !sg.seqs[0].has_aici {
                let seq = &mut sg.seqs[0];
                max_context_len = std::cmp::max(max_context_len, seq.get_len());
                seq.has_aici = true;
                ops.push(AiciPreOp {
                    id: seq.seq_id,
                    req_id: Some(sg.request_id.clone()),
                });
            } else {
                for seq in sg.seqs.iter() {
                    if seq.sched_phase != SchedulingPhase::Running {
                        continue;
                    }
                    assert!(seq.has_aici);
                    max_context_len = std::cmp::max(max_context_len, seq.get_len());
                }
            }
        });

        //  let ids = ops.iter().map(|op| op.id).collect::<Vec<_>>();
        let pre_res = self
            .aicirt
            .as_mut()
            .unwrap()
            .pre_process(AiciPreProcessReq {
                max_context_len,
                freed: self.scheduler.get_freed_seq_ids(),
                ops,
            })?;

        let mut curr_seq_id = self.seq_id;
        self.scheduler.for_each_sg(|sg| {
            if sg.sampling_params.aici_module.is_none() {
                return;
            }

            for seq in sg.seqs.iter_mut() {
                let seq_id = seq.seq_id;
                let res = pre_res.seqs.get(&seq_id);
                if res.is_none() {
                    continue;
                }
                let res = res.unwrap();

                assert!(seq.has_aici);
                self.save_aici_log(seq, &pre_res.seqs);

                match &res.result {
                    Some(r) => {
                        if r.suspend {
                            if seq.sched_phase == SchedulingPhase::Running {
                                seq.sched_phase = SchedulingPhase::Suspended;
                            }
                            continue;
                        }
                        if r.num_forks == 0 {
                            self.scheduler.finish_seq(seq, FinishReason::AiciStop);
                            continue;
                        }
                        if r.ff_tokens.len() > 0 {
                            seq.append_tokens(&r.ff_tokens);
                        }

                        while seq.pending_fork_ids.len() < r.num_forks - 1 {
                            seq.pending_fork_ids.push(curr_seq_id);
                            curr_seq_id += 1;
                        }
                    }
                    None => {}
                }
            }

            let mut num_susp = 0;
            let mut num_running = 0;

            for seq in sg.seqs.iter() {
                match seq.sched_phase {
                    SchedulingPhase::Waiting
                    | SchedulingPhase::Running
                    | SchedulingPhase::Swapped => {
                        num_running += 1;
                    }
                    SchedulingPhase::Suspended => {
                        num_susp += 1;
                    }
                    SchedulingPhase::Finished(_) => {}
                }
            }

            if num_running == 0 && num_susp > 0 {
                for seq in sg.seqs.iter_mut() {
                    self.scheduler.finish_seq(seq, FinishReason::Deadlock);
                }
            }
        });
        self.seq_id = curr_seq_id;

        Ok(())
    }

    fn aici_mid(&mut self, sched_out: &mut SchedulerOutputs) -> Result<()> {
        if self.aicirt.is_none() {
            return Ok(());
        }

        let mut mid_ops = Vec::new();

        for sg in sched_out.next_seq_groups.iter_mut() {
            if sg.sampling_params.aici_module.is_none() {
                continue;
            }

            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase != SchedulingPhase::Running {
                    continue;
                }

                assert!(seq.has_aici);

                mid_ops.push(AiciMidOp {
                    id: seq.seq_id,
                    clone_id: None,
                });

                for copy_id in &seq.pending_fork_ids {
                    mid_ops.push(AiciMidOp {
                        id: *copy_id,
                        clone_id: Some(seq.seq_id),
                    });
                }
            }
        }

        self.aicirt
            .as_mut()
            .unwrap()
            .start_mid_process(AiciMidProcessReq { ops: mid_ops })?;

        Ok(())
    }

    pub fn run_to_completion(&mut self) {
        while self.num_pending_requests() > 0 {
            self.step().expect("step failed");
        }
    }

    pub fn step(&mut self) -> Result<Vec<RequestOutput>> {
        let _no_grad = tch::no_grad_guard();
        let r = with_timer!(self.tim_step, self.step_inner());

        if self.step_no % 20 == 0 {
            log::debug!("timers\n{}", self.timers.pp());
            self.timers.reset();
        }

        r
    }

    fn step_inner(&mut self) -> Result<Vec<RequestOutput>> {
        self.step_no += 1;

        #[cfg(feature = "cuda")]
        if self.step_no == self.profile_step_no {
            cudarc::driver::safe::profiler_start()?;
        }

        with_timer!(self.tim_aici_pre, self.aici_pre()?);

        self.scheduler.for_each_waiting_sg(|sg| {
            if sg.only_seq().get_len() == 0 {
                // this happens when we fork right away, and there is no start token
                // for the current model
                sg.seqs[0].append_tokens(&[self.space_token_id]);
            }
        });

        let mut sched_out = with_timer!(self.tim_schedule, self.scheduler.schedule());

        with_timer!(self.tim_aici_mid, self.aici_mid(&mut sched_out)?);

        log::trace!(
            "scheduled: {} groups, dropped: {}",
            sched_out.next_seq_groups.len(),
            sched_out.dropped_seq_groups.len()
        );
        let outputs = with_timer!(self.tim_run_model, self.run_model(&mut sched_out));
        // we run step_finished() regardless if model failed
        self.scheduler.step_finished(sched_out);

        #[cfg(feature = "cuda")]
        if self.step_no == self.profile_step_no {
            cudarc::driver::safe::profiler_stop()?;
        }

        let outputs = outputs?;
        if outputs.is_empty() {
            assert!(!self.scheduler.has_unfinished_seqs());
        }
        Ok(outputs)
    }

    fn decode_seq(&self, tokens: &Vec<Token>) -> Result<String> {
        let generated = self
            .tokenizer
            .decode(tokens, true)
            .map_err(anyhow::Error::msg)?;
        Ok(generated)
    }

    pub fn generate(&mut self, prompt: &str, sampling_params: SamplingParams) -> Result<String> {
        let req_id = self.gen_req_id();
        self.add_request(req_id, prompt, sampling_params)?;

        let mut outputs = Vec::new();
        let t0 = Instant::now();

        while self.scheduler.has_unfinished_seqs() {
            let outp = self.step()?;
            if !outp.is_empty() {
                assert!(outp.len() == 1);
                assert!(outp[0].seq_outputs.len() == 1);
                outputs = outp[0].seq_outputs[0].output_tokens.clone();
            }
        }

        let dur = Instant::now().duration_since(t0);
        log::debug!(
            "generated {} tokens in {:?}; {:.2} t/s",
            outputs.len(),
            dur,
            outputs.len() as f64 / (dur.as_millis() as f64 / 1000.0)
        );

        Ok(self.decode_seq(&outputs)?)
    }

    pub fn get_stats(&self) -> Stats {
        Stats {
            free_gpu_blocks: self.scheduler.block_manager.get_num_free_gpu_blocks(),
            free_cpu_blocks: self.scheduler.block_manager.get_num_free_cpu_blocks(),
        }
    }
}
