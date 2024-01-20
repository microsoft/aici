use crate::{config::RllmConfig, paged::SchedulerOutputs, seq::SchedulingPhase, HashMap, Tensor};
use aicirt::{with_timer, TimerRef};
use anyhow::Result;
use llama_cpp_low as cpp;
use std::{sync::Arc, time::Instant};

pub struct TModel {
    pub(super) model: cpp::Model,
    batch: cpp::Batch,
    seq_id_to_idx: HashMap<usize, usize>,
    t0: Instant,
    step_no: usize,
    pub nv_profile: bool,
}

impl TModel {
    pub fn new(config: Arc<RllmConfig>, model: cpp::Model) -> Self {
        let batch = cpp::Batch::new(config.scheduler.max_num_batched_tokens);
        Self {
            model,
            batch,
            nv_profile: false,
            seq_id_to_idx: HashMap::default(),
            step_no: 0,
            t0: Instant::now(),
        }
    }

    pub fn run(
        &mut self,
        _vocab_size: usize,
        tim: &TimerRef,
        step_no: usize,
        sched_out: &mut SchedulerOutputs,
    ) -> Result<()> {
        self.step_no = step_no;
        self.batch.clear();
        self.seq_id_to_idx.clear();

        for sg in sched_out.next_seq_groups.iter_mut() {
            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase != SchedulingPhase::Running {
                    continue;
                }

                let seq_len = seq.get_len();
                let k_len = seq_len;
                log::trace!("seq: {seq:?}");
                let mut q_len = seq.get_len() - seq.num_kv_computed;
                if q_len == 0 {
                    // just re-compute the last token
                    q_len = 1;
                }
                sg.usage.gen_tokens += 1;
                sg.usage.prompt_tokens += q_len;

                let off = k_len - q_len;
                for idx in off..off + q_len {
                    let logits = idx + 1 == off + q_len;
                    if logits {
                        self.seq_id_to_idx
                            .insert(seq.seq_id.to_num(), self.batch.len());
                    }
                    seq.seq_id.cpp.assert_model(&self.model);
                    self.batch
                        .add_token(seq.get_token(idx), idx, &seq.seq_id.cpp, logits);
                }

                seq.sync_computed_kv();
            }
        }

        log::trace!("batch_info #{}; {:?}", self.step_no, self.batch);

        self.t0 = Instant::now();

        with_timer!(tim, { self.model.decode(&mut self.batch)? });

        Ok(())
    }

    pub fn get_logits(&self, seq_id: usize) -> Tensor {
        let l = self.model.get_logits(self.seq_id_to_idx[&seq_id]);
        Tensor::from_slice(l)
    }

    pub fn finalize_run(&mut self) -> Result<()> {
        let dur = self.t0.elapsed().as_micros() as f64 / 1000.0;

        let ntok = self.batch.len();

        log::info!(
            "model forward: step #{} {:.2}ms; {} tok(s); {:.1}tps",
            self.step_no,
            dur,
            ntok,
            ntok as f64 / (dur / 1000.0),
        );

        Ok(())
    }
}
