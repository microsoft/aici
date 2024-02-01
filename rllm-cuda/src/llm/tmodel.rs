use std::{sync::Arc, time::Instant};

use aicirt::{with_timer, TimerRef};
use anyhow::Result;
use tch::{IndexOp, Tensor};

use crate::{
    config::RllmConfig,
    llm::util::synchronize,
    paged::{BatchInfo, BatchInfoBuilder, CacheEngine, CacheIface, SchedulerOutputs},
};

pub trait TModelInner {
    fn forward(&self, batch_info: &mut BatchInfo) -> Tensor;
    fn finalize(&mut self) {}
}

pub struct TModel {
    config: Arc<RllmConfig>,
    model: Box<dyn TModelInner>,
    cache_engine: CacheEngine,
    batch_info: Option<BatchInfo>,
    logits: Option<Tensor>,
    t0: Instant,
    pub nv_profile: bool,
}

impl TModel {
    pub fn new(
        config: Arc<RllmConfig>,
        cache_engine: CacheEngine,
        model: Box<dyn TModelInner>,
    ) -> Self {
        Self {
            config,
            cache_engine,
            nv_profile: false,
            model,
            batch_info: None,
            logits: None,
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

    pub fn run(
        &mut self,
        vocab_size: usize,
        tim: &TimerRef,
        step_no: usize,
        sched_out: &mut SchedulerOutputs,
    ) -> Result<()> {
        let mut info = BatchInfoBuilder::new(self.config.clone())
            .sched_out(sched_out)
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
                synchronize(self.config.device.clone());
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

    pub fn get_logits(&self, seq_id: usize) -> Tensor {
        let idx = self.batch_info.as_ref().unwrap().seq_id_to_idx[&seq_id];
        self.logits.as_ref().unwrap().i((idx as i64, ..))
    }

    pub fn finalize_run(&mut self) -> Result<()> {
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
}
