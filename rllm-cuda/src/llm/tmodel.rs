use super::{
    config::{self, TchRllmConfig},
    loader::{load_model_config, load_rllm_engine},
    paged::{BatchInfo, BatchInfoBuilder, BlockSpaceManager, CacheEngine, CacheIface, TchSeqMgr},
    util::synchronize,
    DType,
};
use crate::{
    config::RllmConfig, AiciBias, LogitsProcessor, ModelExec, SchedulerOutputs, TensorOps,
};
use aicirt::{with_timer, TimerRef};
use anyhow::Result;
use rand::distributions::Distribution as _;
use std::{sync::Arc, time::Instant};
use tch::{Device, IndexOp, Tensor};

pub trait TModelInner {
    fn forward(&self, batch_info: &mut BatchInfo) -> Tensor;
    fn finalize(&mut self) {}
}

pub struct TModel {
    config: Arc<RllmConfig<TModel>>,
    model: Box<dyn TModelInner>,
    cache_engine: CacheEngine,
    batch_info: Option<BatchInfo>,
    logits: Option<Tensor>,
    t0: Instant,
    seq_mgr: Arc<TchSeqMgr>,
    pub nv_profile: bool,
}

pub struct TchLoaderArgs {
    pub profile_step_no: usize,
    pub device: Device,
    pub dtype: Option<DType>,
}

impl ModelExec for TModel {
    type Tensor = Tensor;
    type BlockSpaceManager = BlockSpaceManager;
    type AiciBias = TchAiciBias;
    type ModelConfig = config::ModelConfig;
    type ModelLoaderArgs = TchLoaderArgs;
    type SequenceManager = TchSeqMgr;

    fn load_model_config(
        args: &crate::LoaderArgs,
        model_args: &mut Self::ModelLoaderArgs,
    ) -> Result<(crate::config::ModelMeta, Self::ModelConfig)> {
        let m = load_model_config(args, model_args)?;
        Ok((m.meta.clone(), m))
    }

    fn verify_args(args: &RllmConfig<Self>) -> Result<()> {
        args.verify_args()
    }

    fn load_rllm_engine(
        args: crate::LoaderArgs,
        model_args: Self::ModelLoaderArgs,
    ) -> Result<crate::RllmEngine<Self>> {
        load_rllm_engine(args, model_args)
    }

    fn sequence_manager(&self) -> Arc<Self::SequenceManager> {
        self.seq_mgr.clone()
    }

    fn run(
        &mut self,
        vocab_size: usize,
        tim: &TimerRef,
        step_no: usize,
        sched_out: &mut SchedulerOutputs,
    ) -> Result<()> {
        let _no_grad = tch::no_grad_guard();

        if step_no == self.config.model.profile_step_no {
            self.nv_profile = true;
        }

        let mut info = BatchInfoBuilder::new(self.config.clone())
            .sched_out(sched_out, self.seq_mgr.get_gpu_allocator())
            .finish(step_no, self.cache_iface(sched_out));
        log::trace!("batch_info #{}: {:?}", info.step_no, info);

        #[cfg(feature = "cuda")]
        if self.nv_profile {
            cudarc::driver::safe::profiler_start()?;
        }

        self.t0 = Instant::now();

        let logits = with_timer!(tim, {
            let l = self.model.forward(&mut info);
            if false {
                // without this, the timing is off but we may get better perf
                synchronize(self.config.model.device.clone());
            }
            l
        });

        {
            let (num_seq, logit_vocab_size) = logits.size2()?;
            let t_vocab = vocab_size as i64;
            if logit_vocab_size != t_vocab {
                panic!("vocab size mismatch: model {logit_vocab_size} != tokenizer {t_vocab}");
            }
            assert!(num_seq == info.seq_id_to_idx.len() as i64);
        }

        self.batch_info = Some(info);
        self.logits = Some(logits);

        Ok(())
    }

    fn get_logits(&self, seq_id: usize) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        let idx = self.batch_info.as_ref().unwrap().seq_id_to_idx[&seq_id];
        self.logits.as_ref().unwrap().i((idx as i64, ..))
    }

    fn finalize_run(&mut self) -> Result<()> {
        let _no_grad = tch::no_grad_guard();

        let dur = self.t0.elapsed().as_micros() as f64 / 1000.0;
        let info = self.batch_info.as_ref().unwrap();

        log::info!(
            "model forward: step #{} {:.2}ms; {} tok(s); {:.1}tps",
            info.step_no,
            dur,
            info.tokens.numel(),
            info.tokens.numel() as f64 / (dur / 1000.0),
        );

        #[cfg(feature = "cuda")]
        if self.nv_profile {
            cudarc::driver::safe::profiler_stop()?;
        }

        info.save_log(&format!("step-{}.safetensor", info.step_no));
        log::trace!("logits: {:?}", self.logits.as_ref().unwrap());

        Ok(())
    }

    fn empty_bias(&self, vocab_size: usize) -> Self::AiciBias {
        TchAiciBias {
            vocab_size,
            bias: None,
        }
    }

    fn new_bias(
        &self,
        slice: &'static [f32],
        num_seqs: usize,
        vocab_size: usize,
    ) -> Self::AiciBias {
        let _no_grad = tch::no_grad_guard();

        let tensor = Tensor::from_slice(slice)
            .to(self.config.model.device)
            .reshape(&[num_seqs as i64, vocab_size as i64]);
        TchAiciBias {
            vocab_size,
            bias: Some(tensor),
        }
    }

    fn sample(&self, state: &mut LogitsProcessor, logits: &Tensor) -> Result<u32> {
        let _no_grad = tch::no_grad_guard();

        let next_token = match state.temperature {
            None => self.sample_argmax(&logits),
            Some(temperature) => {
                let logits = logits.to_kind(DType::Float);
                let logits = logits / (temperature as f64);
                let prs = logits.softmax(-1, DType::Float);

                let top_p = state.top_p;
                if top_p <= 0.0 || top_p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    prs.multinomial(1, false).int64_value(&[]) as u32
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    let mut prs: Vec<f32> = prs.to_vec1();
                    self.sample_topp(state, &mut prs, top_p as f32)?
                }
            }
        };
        Ok(next_token)
    }
}

impl TensorOps for Tensor {
    fn to_vec1(&self) -> Vec<f32> {
        super::util::to_vec1(&self.to_kind(DType::Float))
    }
}

impl TModel {
    pub fn new(
        config: Arc<RllmConfig<TModel>>,
        cache_engine: CacheEngine,
        seq_mgr: Arc<TchSeqMgr>,
        model: Box<dyn TModelInner>,
    ) -> Self {
        Self {
            config,
            cache_engine,
            nv_profile: false,
            model,
            batch_info: None,
            logits: None,
            seq_mgr,
            t0: Instant::now(),
        }
    }

    fn cache_iface(&mut self, sched_out: &mut SchedulerOutputs) -> Box<dyn CacheIface> {
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
    }

    fn sample_argmax(&self, logits: &Tensor) -> u32 {
        logits.argmax(0, false).int64_value(&[]) as u32
    }

    fn sample_multinomial(&self, state: &mut LogitsProcessor, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(prs)?;
        let next_token = distr.sample(&mut state.rng) as u32;
        Ok(next_token)
    }

    fn sample_topp(
        &self,
        state: &mut LogitsProcessor,
        prs: &mut Vec<f32>,
        top_p: f32,
    ) -> Result<u32> {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability top_p. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].partial_cmp(&prs[i]).unwrap());

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        // Sample with clamped probabilities.
        self.sample_multinomial(state, prs)
    }
}

pub struct TchAiciBias {
    pub vocab_size: usize,
    pub bias: Option<Tensor>,
}

impl AiciBias<Tensor> for TchAiciBias {
    fn apply(&self, logits: &mut Tensor, seq_id: usize) {
        let bias = self.bias.as_ref().unwrap();
        use tch::IndexOp;
        let bias = bias.i((seq_id as i64, ..));
        *logits = &*logits + bias;
    }
}
