use crate::{config::RllmConfig, paged::SchedulerOutputs, Tensor};
use aicirt::TimerRef;
use anyhow::Result;
use std::{sync::Arc, time::Instant};

pub struct TModel {
    config: Arc<RllmConfig>,
    logits: Option<Tensor>,
    t0: Instant,
    pub nv_profile: bool,
}

impl TModel {
    pub fn new(config: Arc<RllmConfig>) -> Self {
        Self {
            config,
            nv_profile: false,
            logits: None,
            t0: Instant::now(),
        }
    }

    pub fn run(
        &mut self,
        vocab_size: usize,
        tim: &TimerRef,
        step_no: usize,
        sched_out: &mut SchedulerOutputs,
    ) -> Result<()> {
        // log::trace!("batch_info #{}: {:?}", info.step_no, info);

        self.t0 = Instant::now();

        // let logits = with_timer!(tim, {
        //     let l = self.model.forward(&mut info);
        //     if false {
        //         // without this, the timing is off but we may get better perf
        //         synchronize(self.config.device.clone());
        //     }
        //     l
        // });

        // self.logits = Some(logits);

        todo!()
    }

    pub fn get_logits(&self, seq_id: usize) -> Tensor {
        todo!()
    }

    pub fn finalize_run(&mut self) -> Result<()> {
        let dur = self.t0.elapsed().as_micros() as f64 / 1000.0;

        // log::info!(
        //     "model forward: step #{} {:.2}ms; {} tok(s); {:.1}tps",
        //     info.step_no,
        //     dur,
        //     info.tokens.numel(),
        //     info.tokens.numel() as f64 / (dur / 1000.0),
        // );

        todo!()
    }
}
