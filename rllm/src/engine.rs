use crate::{DType, Device, IndexOp, Tensor};
use aici_abi::toktree::TokTrie;
use aicirt::api::{
    AiciMidOp, AiciMidProcessReq, AiciPostOp, AiciPostProcessReq, AiciPreOp, AiciPreProcessReq,
    ModuleInstId, SequenceResult,
};
use anyhow::{anyhow, bail, Error as E, Result};
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
    sync::{Arc, Mutex},
    time::Instant,
};
use tch::{nn::VarStore, Kind};
use tokenizers::Tokenizer;

use crate::{
    cache_engine::CacheEngine,
    config::{
        CacheConfig, ModelConfig, ParallelConfig, RllmConfig, SamplingParams, SchedulerConfig,
    },
    iface::AiciRtIface,
    scheduler::SchedulerOutputs,
    seq::{AiciSampling, FinishReason, RequestOutput, SchedulingPhase, SequenceGroup, Token},
    to_offsets,
};
use crate::{
    llama::{Llama, LlamaConfig},
    LoaderArgs,
};
use crate::{
    scheduler::Scheduler,
    seq::{BatchInfo, SeqId, Sequence},
};
use crate::{seq::SeqOutput, LogitsProcessor};

pub struct AddRequest {
    pub request_id: String,
    pub prompt: Vec<Token>,
    pub sampling_params: SamplingParams,
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

pub enum Model {
    Llama(Llama),
}

impl Model {
    pub fn forward(&self, info: &mut BatchInfo) -> Result<Tensor> {
        match self {
            Model::Llama(llama) => Ok(llama.forward(info)?),
        }
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
    pub tokenizer: Arc<Tokenizer>,
    pub tok_trie: Arc<TokTrie>,
    pub model_id: String,
    pub model: Model,
    seq_id: SeqId,
    step_no: usize,
    pub profile_step_no: usize,
    req_id_cnt: usize,
    #[allow(dead_code)]
    pub alt: usize,
    pub device: Device,
    pub eos_token_id: u32,
    pub nv_profile: bool,

    aicirt: Option<AiciRtIface>,

    cache_engine: CacheEngine,
    scheduler: Scheduler,
}

impl RllmEngine {
    pub fn load_tokenizer(args: &LoaderArgs) -> Result<(Tokenizer, TokTrie)> {
        let byte_tokenizer = aici_tokenizers::find_tokenizer(&args.tokenizer)?;
        let tokenizer =
            Tokenizer::from_bytes(byte_tokenizer.hf_bytes).map_err(anyhow::Error::msg)?;
        let tokens = byte_tokenizer.token_bytes();
        let trie = TokTrie::from(&byte_tokenizer.tokrx_info(), &tokens);
        trie.check_against(&tokens);
        Ok((tokenizer, trie))
    }

    pub fn load_model_config(args: &LoaderArgs) -> Result<ModelConfig> {
        let repo = Repo::from(args)?;
        log::info!("loading the model from {}", repo);
        let json_config: LlamaConfig = serde_json::from_slice(&repo.read("config.json")?)?;
        Ok(json_config.into_config())
    }

    pub fn load(args: LoaderArgs) -> Result<RllmEngine> {
        let _no_grad = tch::no_grad_guard();

        let device = Device::Cuda(0);
        let dtype = DType::BFloat16;

        let repo = Repo::from(&args)?;

        let (tokenizer, tok_trie) = Self::load_tokenizer(&args)?;
        let model_config = Self::load_model_config(&args)?;

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
        let eos_token_id = tok_trie.info().tok_eos;

        let mut vs = VarStore::new(device.clone());

        let model = {
            let llama = Llama::load(vs.root(), &model_config)?;
            Model::Llama(llama)
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
                if !vars.contains_key(vname) {
                    log::warn!("variable {} not found in the model", vname);
                    continue;
                }

                let view = safetensors.tensor(vname)?;
                let size: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();
                let kind: DType = kind_from_dt(view.dtype());
                // Using from_blob here instead of from_data_size avoids some unnecessary copy.
                let src_tensor = unsafe {
                    Tensor::from_blob(view.data().as_ptr(), &size, &[], kind, Device::Cpu)
                };
                let mut var = vars.remove(vname).unwrap();
                // println!("copying to {var:?} from {src_tensor:?}");
                var.f_copy_(&src_tensor)?;

                bar.inc(1);
                if bar.is_hidden() {
                    eprint!(".");
                }
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
        }

        if vars.len() > 0 {
            bail!("{} variables not found in the model: {vars:?}", vars.len());
        }

        if bar.is_hidden() {
            eprintln!(" done");
        }
        bar.finish();

        log::info!("model loaded");

        let rllm_config = Arc::new(rllm_config);
        let scheduler = Scheduler::new(rllm_config.clone());
        let cache_engine = CacheEngine::new(rllm_config.clone());

        Ok(RllmEngine {
            tokenizer: Arc::new(tokenizer),
            tok_trie: Arc::new(tok_trie),
            model_id: format!("{}", repo),
            model,
            seq_id: 1,
            step_no: 0,
            profile_step_no: 0,
            req_id_cnt: 0,
            device,
            eos_token_id,
            alt: args.alt,
            scheduler,
            cache_engine,
            nv_profile: false,
            aicirt: None,
        })
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
        };

        self.scheduler.add_seq_group(sg);

        Ok(())
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
        })
    }

    fn splice_seq(&self, seq: &mut Sequence, backtrack: usize, tokens: &[Token]) {
        seq.tokens.truncate(seq.tokens.len() - backtrack);
        seq.output_ptr = std::cmp::min(seq.output_ptr, seq.tokens.len());
        self.scheduler.block_manager.trim_physical_blocks(seq);
        seq.tokens.extend_from_slice(tokens);
    }

    #[allow(dead_code)]
    fn splice_seq_id(&mut self, seq_id: SeqId, backtrack: usize, tokens: &[Token]) {
        self.scheduler.for_each_sg(|sg| {
            sg.seqs.iter_mut().for_each(|seq| {
                if seq.seq_id == seq_id {
                    self.splice_seq(seq, backtrack, tokens)
                }
            })
        })
    }

    fn aici_bias(&mut self, sched_out: &mut SchedulerOutputs) -> Result<Tensor> {
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
                        // we may drop some of it but self.splice_seq() takes care of that
                        seq.num_kv_computed = seq.tokens.len();

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
                self.splice_seq(seq, backtrack as usize, &ff_tokens);
                Some(AiciPostOp {
                    id: seq.seq_id,
                    tokens: ff_tokens,
                    backtrack: backtrack,
                    clone_id: None,
                })
            }
        }
    }

    fn generate_outputs(
        &mut self,
        logits: &Tensor,
        sched_out: &mut SchedulerOutputs,
    ) -> Result<Vec<RequestOutput>> {
        let mut idx = 0;

        let mut seq_id_to_logit_idx = HashMap::new();
        for sg in sched_out.next_seq_groups.iter_mut() {
            let mut to_add = Vec::new();
            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase != SchedulingPhase::Running {
                    continue;
                }
                seq_id_to_logit_idx.insert(seq.seq_id, idx);
                let pending = std::mem::take(&mut seq.pending_fork_ids);
                for copy_id in pending {
                    seq_id_to_logit_idx.insert(copy_id, idx);
                    let copy = seq.fork_as(copy_id, sg.max_index + 1);
                    sg.max_index += 1;
                    log::debug!("forked: {:?} -> {:?}", seq, copy);
                    to_add.push(copy);
                }
                idx += 1;
            }
            sg.seqs.extend(to_add);
        }

        let aici_bias = self.aici_bias(sched_out)?;

        if idx > 0 {
            let (num_seq, vocab_size) = logits.size2()?;
            assert!(vocab_size == self.tok_trie.vocab_size() as i64);
            assert!(num_seq == idx);
        }

        let mut outputs = Vec::new();
        for sg in sched_out.dropped_seq_groups.iter_mut() {
            outputs.push(self.req_output(sg, true));
        }

        let mut post_ops = Vec::new();

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

                seq.num_kv_computed = seq.tokens.len();

                let mut info = "";
                let next_token = sg.logits_processor.sample(&logits)?;
                if seq.has_aici && next_token == self.tok_trie.info().tok_eos {
                    // replace with space, so the model doesn't get confused
                    // note that aici will still get the real EOS token
                    let space = self.tok_trie.greedy_tokenize(b" ")[0];
                    seq.tokens.push(space);
                    info = " -> space";
                } else {
                    seq.tokens.push(next_token);
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

        self.aici_post(sched_out, AiciPostProcessReq { ops: post_ops })?;

        for sg in sched_out.next_seq_groups.iter_mut() {
            outputs.push(self.req_output(sg, false));
        }

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
            is_ambiguous: sg.logits_processor.num_ambiguous > 0,
            is_final,
        }
    }

    fn run_model(&mut self, sched_out: &mut SchedulerOutputs) -> Result<Vec<RequestOutput>> {
        if sched_out.is_empty() {
            log::debug!("no seqs to run");
            let logits = Tensor::from_slice(&[0u8]).to(self.device);
            // still run generate_outputs() to finish the dropped seqs
            return self.generate_outputs(&logits, sched_out);
        }

        let mut issued_cache_op = false;
        if sched_out.blocks_to_swap_in.len() > 0 {
            self.cache_engine.swap_in(&sched_out.blocks_to_swap_in);
            issued_cache_op = true;
        }
        if sched_out.blocks_to_swap_out.len() > 0 {
            self.cache_engine.swap_out(&sched_out.blocks_to_swap_out);
            issued_cache_op = true;
        }
        if sched_out.blocks_to_copy.len() > 0 {
            self.cache_engine.copy(&sched_out.blocks_to_copy);
            issued_cache_op = true;
        }

        if issued_cache_op {
            self.cache_engine.wait_for_copy();
        }

        let mut info = self.build_batch_info(sched_out)?;

        log::trace!("batch_info #{}: {:?}", info.step_no, info);
        // log::trace!("{}", info.positions);
        // log::trace!("{}", info.gather_mapping);
        // log::trace!("{}", info.slot_mapping);

        if self.nv_profile {
            cudarc::driver::safe::profiler_start()?;
        }

        let t0 = Instant::now();
        let logits = self.model.forward(&mut info)?;
        let r = self.generate_outputs(&logits, sched_out);
        log::info!(
            "model forward: step #{} {:?}; {} toks; {:?}/tok",
            self.step_no,
            t0.elapsed(),
            info.tokens.numel(),
            t0.elapsed() / info.tokens.numel() as u32
        );

        if self.nv_profile {
            cudarc::driver::safe::profiler_stop()?;
        }

        info.save_log(&format!("step-{}.safetensor", self.step_no));
        log::trace!("logits: {:?}", logits);
        r
    }

    fn build_batch_info(&self, sched_out: &mut SchedulerOutputs) -> Result<BatchInfo> {
        let mut positions: Vec<i64> = Vec::new();
        let mut tokens: Vec<i32> = Vec::new();
        let mut seqlens_q = Vec::new();
        let mut seqlens_k = Vec::new();
        let mut gather_mapping: Vec<i32> = Vec::new();
        let mut slot_mapping: Vec<i32> = Vec::new();

        let max_seq = self.scheduler.config.model.max_sequence_length;

        for sg in sched_out.next_seq_groups.iter_mut() {
            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase != SchedulingPhase::Running {
                    continue;
                }

                let seq_len = seq.tokens.len();
                let k_len = seq_len;
                log::trace!("seq: {seq:?}");
                let q_len = seq.tokens.len() - seq.num_kv_computed;
                assert!(q_len > 0); // TODO if it's 0, we can probably bump it up to 1 (re-compute)
                let off = k_len - q_len;
                for idx in off..off + q_len {
                    assert!(idx < max_seq);
                    positions.push(idx as i64);
                    tokens.push(seq.tokens[idx] as i32);
                    slot_mapping.push(seq.get_gpu_slot(idx) as i32);
                }
                for idx in 0..k_len {
                    gather_mapping.push(seq.get_gpu_slot(idx) as i32);
                }
                seqlens_q.push(q_len);
                seqlens_k.push(k_len);
            }
        }

        let device = self.device;
        let (max_seqlen_q, seqlens_q) = to_offsets(&seqlens_q, device);
        let (max_seqlen_k, seqlens_k) = to_offsets(&seqlens_k, device);

        // TODO positions, tokens should be padded to 8? see worker.py, search for multiple_of=8
        let positions = Tensor::from_slice(positions.as_slice()).to(device);
        let tokens = Tensor::from_slice(tokens.as_slice()).to(device);
        let slot_mapping = Tensor::from_slice(slot_mapping.as_slice()).to(device);
        let gather_mapping = Tensor::from_slice(gather_mapping.as_slice()).to(device);

        let kv_cache = self.cache_engine.get_gpu_cache();

        Ok(BatchInfo {
            tokens,
            positions,
            seqlens_q,
            seqlens_k,
            slot_mapping,
            gather_mapping,
            max_seqlen_q,
            max_seqlen_k,
            kv_cache,
            infer_log: Mutex::new(Vec::new()),
            step_no: self.step_no,
        })
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

    fn aici_pre(&mut self, sched_out: &mut SchedulerOutputs) -> Result<()> {
        let mut max_context_len = 0;
        let mut ops = Vec::new();
        let mut seq_ptr = HashMap::new();

        for (sg_idx, sg) in sched_out.next_seq_groups.iter_mut().enumerate() {
            if sg.sampling_params.aici_module.is_none() {
                continue;
            }
            if sg.seqs.len() == 1 && !sg.seqs[0].has_aici {
                let seq = &mut sg.seqs[0];
                max_context_len = std::cmp::max(max_context_len, seq.tokens.len());
                seq.has_aici = true;
                seq_ptr.insert(seq.seq_id, (sg_idx, 0));
                ops.push(AiciPreOp {
                    id: seq.seq_id,
                    req_id: Some(sg.request_id.clone()),
                });
            } else {
                for (seq_idx, seq) in sg.seqs.iter_mut().enumerate() {
                    if seq.sched_phase != SchedulingPhase::Running {
                        continue;
                    }
                    assert!(seq.has_aici);
                    ops.push(AiciPreOp {
                        id: seq.seq_id,
                        req_id: None,
                    });
                    seq_ptr.insert(seq.seq_id, (sg_idx, seq_idx));
                    max_context_len = std::cmp::max(max_context_len, seq.tokens.len());
                }
            }
        }

        let ids = ops.iter().map(|op| op.id).collect::<Vec<_>>();
        let pre_res = self
            .aicirt
            .as_mut()
            .unwrap()
            .pre_process(AiciPreProcessReq {
                max_context_len,
                freed: self.scheduler.get_freed_seq_ids(),
                ops,
            })?;

        let mut mid_ops = Vec::new();

        for seq_id in ids {
            let res = pre_res
                .seqs
                .get(&seq_id)
                .ok_or(anyhow!("missing seq {}", seq_id))?;
            let (sg_idx, seq_idx) = seq_ptr[&seq_id];
            let sg = &mut sched_out.next_seq_groups[sg_idx];
            let seq = &mut sg.seqs[seq_idx];

            assert!(seq.has_aici);
            self.save_aici_log(seq, &pre_res.seqs);

            let suspend = res.result.as_ref().map(|r| r.suspend).unwrap_or(false);
            if suspend {
                seq.sched_phase = SchedulingPhase::Suspended;
                continue;
            }

            let num_forks = res.result.as_ref().map(|r| r.num_forks).unwrap_or(0);
            if num_forks == 0 {
                self.scheduler.finish_seq(seq, FinishReason::AiciStop);
                continue;
            }

            mid_ops.push(AiciMidOp {
                id: seq_id,
                clone_id: None,
            });

            while seq.pending_fork_ids.len() < num_forks - 1 {
                seq.pending_fork_ids.push(self.seq_id);
                self.seq_id += 1;
            }

            for copy_id in &seq.pending_fork_ids {
                mid_ops.push(AiciMidOp {
                    id: *copy_id,
                    clone_id: Some(seq.seq_id),
                });
            }
        }

        self.aicirt
            .as_mut()
            .unwrap()
            .start_mid_process(AiciMidProcessReq { ops: mid_ops })?;

        Ok(())
    }

    pub fn step(&mut self) -> Result<Vec<RequestOutput>> {
        let _no_grad = tch::no_grad_guard();

        self.step_no += 1;

        if self.step_no == self.profile_step_no {
            cudarc::driver::safe::profiler_start()?;
        }

        let mut sched_out = self.scheduler.schedule();

        self.aici_pre(&mut sched_out)?;

        log::trace!(
            "scheduled: {} groups, dropped: {}",
            sched_out.next_seq_groups.len(),
            sched_out.dropped_seq_groups.len()
        );
        let outputs = self.run_model(&mut sched_out);
        // we run step_finished() regardless if model failed
        self.scheduler.step_finished(sched_out);

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
            "generted {} tokens in {:?}; {:.2} t/s",
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
