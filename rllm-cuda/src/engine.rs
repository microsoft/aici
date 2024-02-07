use crate::{
    config::{CacheConfig, ParallelConfig, RllmConfig, SamplingParams, SchedulerConfig},
    iface::AiciRtIface,
    seq::{
        AiciSampling, FinishReason, RequestOutput, SchedulingPhase, SeqOutput, Sequence,
        SequenceGroup, Token, TokenUsage,
    },
    util::get_setting,
    AiciBias as _, HashMap, LoaderArgs, LogitsProcessor, ModelExec, Scheduler, SchedulerOutputs,
    SequenceManager, TBlockSpaceManager as _, TensorOps,
};
use aici_abi::toktree::TokTrie;
use aicirt::{
    api::{
        AiciMidOp, AiciMidProcessReq, AiciPostOp, AiciPostPreProcessReq, AiciPreOp, ModuleInstId,
        SequenceResult,
    },
    with_timer, TimerRef, TimerSet,
};
use anyhow::{bail, Error as E, Result};
use hf_hub::{
    api::sync::{Api, ApiRepo},
    RepoType,
};
use serde::{Deserialize, Serialize};
use std::{fmt::Display, ops::Deref, path::PathBuf, sync::Arc, time::Instant};
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

pub struct AddRequest {
    pub request_id: String,
    pub prompt: Vec<Token>,
    pub sampling_params: SamplingParams,
    pub expected: Option<ExpectedGeneration>,
}

pub(crate) enum Repo {
    Api(ApiRepo),
    Local(String),
}

impl Repo {
    pub fn from(args: &LoaderArgs) -> Result<Repo> {
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

    #[allow(dead_code)]
    pub fn is_local(&self) -> bool {
        match self {
            Repo::Api(_) => false,
            Repo::Local(_) => true,
        }
    }

    pub fn get(&self, filename: &str) -> Result<PathBuf> {
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

    #[allow(dead_code)]
    pub fn read(&self, filename: &str) -> Result<Vec<u8>> {
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

pub struct RllmEngine<ME: ModelExec> {
    pub config: Arc<RllmConfig<ME>>,
    pub tokenizer: Arc<Tokenizer>,
    pub tok_trie: Arc<TokTrie>,
    pub model_id: String,
    pub tmodel: ME,
    pub(crate) step_no: usize,
    pub profile_step_no: usize,
    req_id_cnt: usize,
    #[allow(dead_code)]
    pub alt: usize,
    pub eos_token_id: Token,
    pub space_token_id: Token,
    pub num_errors: usize,

    post_ops: Vec<AiciPostOp>,

    pub timers: TimerSet,

    tim_step: TimerRef,

    tim_schedule: TimerRef,
    tim_aici_mid: TimerRef,
    tim_run_model: TimerRef,

    pub(crate) tim_model_fwd: TimerRef,
    tim_sample: TimerRef,

    tim_aici_bias: TimerRef,
    tim_logit_sample: TimerRef,
    tim_aici_post: TimerRef,

    aicirt: Option<AiciRtIface>,

    scheduler: Scheduler<ME>,
    seq_mgr: Arc<ME::SequenceManager>,
}

impl<ME: ModelExec> RllmEngine<ME> {
    pub(crate) fn build_config(
        args: &LoaderArgs,
        model_args: &mut ME::ModelLoaderArgs,
    ) -> Result<RllmConfig<ME>> {
        let (model_meta, model_config) = ME::load_model_config(args, model_args)?;
        let model_len = model_meta.max_sequence_length;

        let mut aici = args.aici.clone();
        if aici.max_fuel == 0 {
            aici.max_fuel = model_len * 10;
        }

        let rllm_config = RllmConfig {
            model: model_config,
            meta: model_meta,
            parallel: ParallelConfig::single(),
            cache: CacheConfig::default(),
            scheduler: SchedulerConfig {
                max_num_batched_tokens: model_len,
                max_num_kv_tokens: model_len * 10,
                max_num_seqs: 100,
                max_model_len: model_len,
            },
            aici,
        };

        ME::verify_args(&rllm_config)?;

        Ok(rllm_config)
    }

    pub(crate) fn build(
        mut args: LoaderArgs,
        tmodel: ME,
        block_space_manager: ME::BlockSpaceManager,
        rllm_config: Arc<RllmConfig<ME>>,
    ) -> Result<Self> {
        let (tokenizer, tok_trie) = RllmEngine::<ME>::load_tokenizer(&mut args)?;
        let eos_token_id = tok_trie.info().tok_eos;
        let space_token_id = tok_trie.greedy_tokenize(b" ")[0];
        let repo = Repo::from(&args)?;

        let scheduler = Scheduler::new(
            tmodel.sequence_manager(),
            block_space_manager,
            rllm_config.clone(),
        );

        let timers = TimerSet::new();
        let mut model_id = format!("{}", repo);
        match &args.revision {
            Some(r) => model_id += &format!("@{}", r),
            None => {}
        }
        match &args.file {
            Some(r) => model_id += &format!("::{}", r),
            None => {}
        }

        Ok(RllmEngine {
            config: rllm_config,
            tokenizer: Arc::new(tokenizer),
            tok_trie: Arc::new(tok_trie),
            model_id,
            seq_mgr: tmodel.sequence_manager(),
            tmodel,
            step_no: 0,
            profile_step_no: 0,
            req_id_cnt: 0,
            num_errors: 0,
            eos_token_id,
            space_token_id,
            alt: args.alt,
            scheduler,
            aicirt: None,
            post_ops: Vec::new(),
            tim_step: timers.new_timer("step"),
            tim_schedule: timers.new_timer("step.schedule"),
            tim_aici_mid: timers.new_timer("step.aici_mid"),
            tim_run_model: timers.new_timer("step.run_model"),
            tim_model_fwd: timers.new_timer("step.run_model.model_fwd"),
            tim_sample: timers.new_timer("step.run_model.sample"),
            tim_aici_bias: timers.new_timer("step.run_model.sample.aici_bias"),
            tim_logit_sample: timers.new_timer("step.run_model.sample.sample"),
            tim_aici_post: timers.new_timer("step.run_model.sample.aici_post"),
            timers,
        })
    }

    pub fn load_tokenizer(args: &mut LoaderArgs) -> Result<(Tokenizer, TokTrie)> {
        let byte_tokenizer = aicirt::bintokens::find_tokenizer(&args.tokenizer)?;
        let tokens = byte_tokenizer.token_bytes();
        log::info!(
            "TokTrie building: {:?} wl={}",
            byte_tokenizer.tokrx_info(),
            tokens.len()
        );
        let trie = TokTrie::from(&byte_tokenizer.tokrx_info(), &tokens);
        trie.check_against(&tokens);
        Ok((byte_tokenizer.hf_tokenizer, trie))
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
            self.seq_mgr.new_sequence(),
            &req.prompt,
            self.scheduler.config.cache.block_size,
        );
        seq.expected = req.expected;
        seq.pending_fork_ids = (1..req.sampling_params.n)
            .map(|_| self.seq_mgr.new_sequence())
            .collect::<Vec<_>>();

        let logits_processor = LogitsProcessor::new(&req.sampling_params);
        let prompt = self
            .tokenizer
            .decode(&req.prompt, false)
            .map_err(anyhow::Error::msg)?;

        let sg = SequenceGroup {
            request_id: req.request_id,
            prompt,
            seqs: vec![seq],
            sampling_params: req.sampling_params,
            deadlock_steps: 0,
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

    fn aici_bias(&mut self, sched_out: &mut SchedulerOutputs) -> Result<ME::AiciBias> {
        let vocab_size = self.tok_trie.vocab_size();
        if self.aicirt.is_none() {
            return Ok(self.tmodel.empty_bias(vocab_size));
        }

        let mid_res = self.aicirt.as_mut().unwrap().finish_mid_process()?;
        let mut idx = 0;

        for sg in sched_out.next_seq_groups.iter_mut() {
            if sg.sampling_params.controller.is_none() {
                continue;
            }
            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase != SchedulingPhase::Running {
                    continue;
                }
                assert!(seq.has_aici);
                match self.save_aici_log(seq, &mid_res.seqs) {
                    Some(r) if r.ff_tokens.len() > 0 || r.backtrack > 0 => {
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
        let slice = shm.slice_at_byte_offset::<f32>(0, mid_res.num_seqs * vocab_size);
        Ok(self.tmodel.new_bias(
            //    self.device,
            slice,
            mid_res.num_seqs,
            vocab_size,
        ))
    }

    fn aici_apply_bias(
        &self,
        seq: &mut Sequence,
        logits: &mut ME::Tensor,
        aici_bias: &ME::AiciBias,
    ) -> Option<AiciPostOp> {
        match std::mem::take(&mut seq.aici_sampling) {
            AiciSampling::Regular => None,
            AiciSampling::SampleWithBias { offset } => {
                log::trace!("sample *{}: bias at {offset}", seq.seq_id);
                aici_bias.apply(logits, offset);
                None
            }
            AiciSampling::Splice {
                backtrack,
                ff_tokens,
            } => {
                log::trace!(
                    "sample *{}: backtrack:{backtrack} ff_tokens:{ff_tokens:?}",
                    seq.seq_id
                );
                seq.splice_tokens(self.seq_mgr.deref(), backtrack as usize, &ff_tokens);
                Some(AiciPostOp {
                    id: seq.seq_id.to_num(),
                    tokens: ff_tokens,
                    backtrack: backtrack,
                })
            }
        }
    }

    fn check_expected(&mut self, mut logits: Vec<f32>, req_id: &str, seq: &mut Sequence) -> Token {
        let exp = seq.expected.as_ref().unwrap();
        let idx = seq.get_len() - exp.prompt.len();
        let next_token = if idx >= exp.output.len() {
            self.eos_token_id
        } else {
            let out = &exp.output[idx];
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
        let mut res = Vec::new();

        // in addition to dropped, we also add suspended/deadlocked
        self.scheduler.for_each_ongpu_sg(|sg| {
            if sg.deadlock_steps > 0 {
                res.push(self.req_output(sg, false));
            }
        });

        sched_out
            .dropped_seq_groups
            .iter_mut()
            .for_each(|sg| res.push(self.req_output(sg, true)));

        res
    }

    fn empty_outputs(&mut self, sched_out: &mut SchedulerOutputs) -> Result<Vec<RequestOutput>> {
        let _ = self.aici_bias(sched_out)?;
        Ok(self.dropped_outputs(sched_out))
    }

    fn sample(
        &mut self,
        sched_out: &mut SchedulerOutputs,
    ) -> Result<(Vec<RequestOutput>, Vec<AiciPostOp>)> {
        let mut seq_id_mapping = HashMap::default();

        for sg in sched_out.next_seq_groups.iter_mut() {
            let mut to_add = Vec::new();
            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase != SchedulingPhase::Running {
                    continue;
                }
                // get it even if no pending - make sure we have them all
                let pending = std::mem::take(&mut seq.pending_fork_ids);
                for copy_id in pending {
                    seq_id_mapping.insert(copy_id.to_num(), seq.seq_id.to_num());
                    let copy = seq.fork_as(self.seq_mgr.deref(), copy_id, sg.max_index + 1);
                    sg.max_index += 1;
                    log::debug!("forked: {:?} -> {:?}", seq, copy);
                    to_add.push(copy);
                }
            }
            sg.seqs.extend(to_add);
        }

        let aici_bias = with_timer!(self.tim_aici_bias, self.aici_bias(sched_out)?);

        let mut post_ops = Vec::new();

        for sg in sched_out.next_seq_groups.iter_mut() {
            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase != SchedulingPhase::Running {
                    continue;
                }

                let sidx = seq.seq_id.to_num();
                let sidx = seq_id_mapping.get(&sidx).unwrap_or(&sidx);
                let mut logits = self.tmodel.get_logits(*sidx);

                if let Some(op) = self.aici_apply_bias(seq, &mut logits, &aici_bias) {
                    post_ops.push(op);
                    continue;
                }

                let next_token = if seq.expected.is_some() {
                    let logits = logits.to_vec1();
                    self.check_expected(logits, &sg.request_id, seq)
                } else {
                    with_timer!(
                        self.tim_logit_sample,
                        self.tmodel.sample(&mut sg.logits_processor, &logits)?
                    )
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
                        id: seq.seq_id.to_num(),
                        tokens: vec![next_token],
                        backtrack: 0,
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

        let mut outputs = self.dropped_outputs(sched_out);
        outputs.extend(
            sched_out
                .next_seq_groups
                .iter_mut()
                .map(|sg| self.req_output(sg, false)),
        );

        Ok((outputs, post_ops))
    }

    fn req_output(&self, sg: &mut SequenceGroup, is_final: bool) -> RequestOutput {
        RequestOutput {
            request_id: sg.request_id.clone(),
            seq_outputs: sg
                .seqs
                .iter_mut()
                .map(|seq| seq.gen_output(&self.tok_trie))
                .collect(),
            usage: sg.usage.clone(),
            is_final,
        }
    }

    fn run_model(
        &mut self,
        sched_out: &mut SchedulerOutputs,
    ) -> Result<(Vec<RequestOutput>, Vec<AiciPostOp>)> {
        if sched_out.is_empty() {
            log::debug!("no seqs to run");
            return Ok((self.empty_outputs(sched_out)?, vec![]));
        }

        self.tmodel.run(
            self.tok_trie.vocab_size(),
            &self.tim_model_fwd,
            self.step_no,
            sched_out,
        )?;

        let r = with_timer!(self.tim_sample, { self.sample(sched_out) });

        self.tmodel.finalize_run()?;

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
        if let Some(r) = seqs.get(&seq.seq_id.to_num()) {
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

    fn aici_post_pre(&mut self, post_ops: Vec<AiciPostOp>) -> Result<()> {
        if self.aicirt.is_none() {
            return Ok(());
        }

        let mut pre_ops = Vec::new();

        self.scheduler.for_each_sg(|sg| {
            if sg.sampling_params.controller.is_none() {
                return;
            }
            if sg.seqs.len() == 1 && !sg.seqs[0].has_aici {
                let seq = &mut sg.seqs[0];
                seq.has_aici = true;
                pre_ops.push(AiciPreOp {
                    id: seq.seq_id.to_num(),
                    req_id: sg.request_id.clone(),
                });
            } else {
                for seq in sg.seqs.iter() {
                    if seq.sched_phase != SchedulingPhase::Running {
                        continue;
                    }
                    assert!(seq.has_aici);
                }
            }
        });

        //  let ids = ops.iter().map(|op| op.id).collect::<Vec<_>>();
        let aici_res = self
            .aicirt
            .as_mut()
            .unwrap()
            .post_pre_process(AiciPostPreProcessReq {
                pre_ops,
                post_ops,
                freed: self.scheduler.get_freed_seq_ids(),
            })?;

        // let gen = &mut self.seq_gen;
        self.scheduler.for_each_sg(|sg| {
            if sg.sampling_params.controller.is_none() {
                return;
            }

            for seq in sg.seqs.iter_mut() {
                let sid = seq.seq_id.to_num();

                match self.save_aici_log(seq, &aici_res.post_seqs) {
                    Some(r) if r.stop => {
                        self.scheduler.finish_seq(seq, FinishReason::AiciStop);
                    }
                    _ => {}
                }

                let res = aici_res.pre_seqs.get(&sid);
                if res.is_none() {
                    continue;
                }
                let res = res.unwrap();

                assert!(seq.has_aici);
                self.save_aici_log(seq, &aici_res.pre_seqs);

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
                            seq.pending_fork_ids.push(self.seq_mgr.new_sequence());
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
                sg.deadlock_steps += 1;
                if sg.deadlock_steps > 3 {
                    for seq in sg.seqs.iter_mut() {
                        self.scheduler.finish_seq(seq, FinishReason::Deadlock);
                    }
                }
            } else {
                sg.deadlock_steps = 0;
            }
        });

        Ok(())
    }

    fn aici_mid(&mut self, sched_out: &mut SchedulerOutputs) -> Result<()> {
        if self.aicirt.is_none() {
            return Ok(());
        }

        let mut mid_ops = Vec::new();

        for sg in sched_out.next_seq_groups.iter_mut() {
            if sg.sampling_params.controller.is_none() {
                continue;
            }

            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase != SchedulingPhase::Running {
                    continue;
                }

                assert!(seq.has_aici);

                mid_ops.push(AiciMidOp {
                    id: seq.seq_id.to_num(),
                    clone_id: None,
                });

                for copy_id in &seq.pending_fork_ids {
                    mid_ops.push(AiciMidOp {
                        id: copy_id.to_num(),
                        clone_id: Some(seq.seq_id.to_num()),
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
        #[cfg(feature = "tch")]
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

        let post_ops = std::mem::take(&mut self.post_ops);
        with_timer!(self.tim_aici_post, self.aici_post_pre(post_ops)?);

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

        let (outputs, post_ops) = outputs?;
        if outputs.is_empty() {
            assert!(!self.scheduler.has_unfinished_seqs());
        }

        assert!(self.post_ops.is_empty());
        self.post_ops = post_ops;

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
