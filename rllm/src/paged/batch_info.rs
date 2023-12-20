use super::{cache_engine::CacheEngine, scheduler::SchedulerOutputs};
use crate::{config::RllmConfig, llm::kernels::to_offsets, seq::SchedulingPhase};
use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};
use tch::Tensor;

pub struct BatchInfo {
    pub tokens: Tensor,         // u32, [num_tokens]
    pub positions: Tensor,      // i64, [num_tokens]
    pub seqlens_q: Tensor,      // u32, [batch_size + 1]; points to tokens/positions
    pub seqlens_k: Tensor,      // u32, [batch_size + 1]; can go outside tokens/positions
    pub gather_mapping: Tensor, // u32, [sum(context_len + prompt_len)]
    pub slot_mapping: Tensor,   // u32, [num_tokens]
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub kv_cache: Vec<(Tensor, Tensor)>,

    pub infer_log: Mutex<Vec<(String, Tensor)>>,
    pub step_no: usize,
}

impl BatchInfo {
    pub fn log_tensor(&self, key: &str, value: &Tensor) {
        if false {
            self.infer_log
                .lock()
                .unwrap()
                .push((key.to_string(), value.copy()));
        }
    }

    pub fn save_log(&self, filename: &str) {
        let mut lck = self.infer_log.lock().unwrap();
        if lck.len() == 0 {
            return;
        }
        let tensors = lck
            .iter()
            .enumerate()
            .map(|(i, (k, v))| (format!("{:0>4}_{}", i, k), v.copy()))
            .collect::<Vec<_>>();
        lck.clear();
        Tensor::write_safetensors(&tensors, filename).unwrap();
    }
}

impl Debug for BatchInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchInfo")
            .field("tokens", &self.tokens)
            .field("positions", &self.positions)
            .field("seqlens_q", &self.seqlens_q)
            .field("seqlens_k", &self.seqlens_k)
            // .field("gather_mapping", &self.gather_mapping)
            // .field("slot_mapping", &self.slot_mapping)
            .field("max_seqlen_q", &self.max_seqlen_q)
            .field("max_seqlen_k", &self.max_seqlen_k)
            .finish()
    }
}

pub struct BatchInfoBuilder {
    positions: Vec<i64>,
    tokens: Vec<i32>,
    seqlens: Vec<(usize, usize)>,
    gather_mapping: Vec<i32>,
    slot_mapping: Vec<i32>,
    config: Arc<RllmConfig>,
}

impl BatchInfoBuilder {
    pub fn new(config: Arc<RllmConfig>) -> Self {
        Self {
            positions: Vec::new(),
            tokens: Vec::new(),
            seqlens: Vec::new(),
            gather_mapping: Vec::new(),
            slot_mapping: Vec::new(),
            config,
        }
    }

    pub fn sched_out(&mut self, sched_out: &mut SchedulerOutputs) -> &mut Self {
        let sch_cfg = &self.config.scheduler;
        let max_seq = sch_cfg.max_model_len;

        assert!(sched_out.next_seq_groups.len() > 0);
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
                let off = k_len - q_len;
                sg.usage.gen_tokens += 1;
                sg.usage.prompt_tokens += q_len;
                for idx in off..off + q_len {
                    assert!(idx < max_seq);
                    self.positions.push(idx as i64);
                    self.tokens.push(seq.get_token(idx) as i32);
                    self.slot_mapping.push(seq.get_gpu_slot(idx) as i32);
                }
                for idx in 0..k_len {
                    self.gather_mapping.push(seq.get_gpu_slot(idx) as i32);
                }
                self.seqlens.push((q_len, k_len));
            }
        }

        self
    }

    pub fn profile_run(&mut self) -> BatchInfo {
        let sch_cfg = &self.config.scheduler;
        let seq_len = sch_cfg.max_model_len;
        for idx in 0..sch_cfg.max_num_seqs {
            self.positions.push((idx % 200) as i64);
            self.tokens.push(1);
            self.slot_mapping.push(0);
            for _ in 0..seq_len {
                self.gather_mapping.push(0);
            }
            self.seqlens.push((1, seq_len));
        }
        let mut left = sch_cfg.max_num_batched_tokens - sch_cfg.max_num_seqs;
        while left > 0 {
            let seq_len = std::cmp::min(seq_len, left);
            left -= seq_len;
            for idx in 0..seq_len {
                self.positions.push(idx as i64);
                self.tokens.push(2);
                self.slot_mapping.push(0);
                self.gather_mapping.push(0);
            }
            self.seqlens.push((seq_len, seq_len));
        }

        self.fake_finish()
    }

    fn fake_finish(&self) -> BatchInfo {
        let (k, v) = CacheEngine::alloc_gpu_cache_layer(&self.config, 1);
        let num_layers = self.config.get_num_layers_parallel();
        let kv_cache = (0..num_layers)
            .map(|_| (k.shallow_clone(), v.shallow_clone()))
            .collect();
        self.finish(0, kv_cache)
    }

    pub fn finish(&self, step_no: usize, kv_cache: Vec<(Tensor, Tensor)>) -> BatchInfo {
        let seqlens = &self.seqlens;
        assert!(seqlens.len() > 0);
        assert!(seqlens.iter().all(|(q, k)| *q <= *k));

        let device = self.config.device;
        let (max_seqlen_q, seqlens_q) = to_offsets(seqlens.iter().map(|(q, _)| *q), device);
        let (max_seqlen_k, seqlens_k) = to_offsets(seqlens.iter().map(|(_, k)| *k), device);

        // TODO positions, tokens should be padded to 8? see worker.py, search for multiple_of=8
        let positions = Tensor::from_slice(self.positions.as_slice()).to(device);
        let tokens = Tensor::from_slice(self.tokens.as_slice()).to(device);
        let slot_mapping = Tensor::from_slice(self.slot_mapping.as_slice()).to(device);
        let gather_mapping = Tensor::from_slice(self.gather_mapping.as_slice()).to(device);

        BatchInfo {
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
            step_no,
        }
    }
}
