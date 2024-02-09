use super::super::{kernels::to_offsets, tmodel::TModel};
use super::cache_engine::CacheEngine;
use super::BlockAllocator;
use rllm::{
    config::RllmConfig, seq::SchedulingPhase, util::pad_to_multiple, HashMap, SchedulerOutputs,
};
use aicirt::api::Token;
use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};
use tch::{IndexOp, Tensor};

pub trait CacheIface {
    fn get(&self, layer_no: usize) -> (Tensor, Tensor);
}

pub struct BatchInfo {
    pub tokens: Tensor,         // u32, [num_tokens]
    pub positions: Tensor,      // i64, [num_tokens]
    pub seqlens_q: Tensor,      // u32, [batch_size + 1]; points to tokens/positions
    pub seqlens_k: Tensor,      // u32, [batch_size + 1]; can go outside tokens/positions
    pub gather_mapping: Tensor, // u32, [sum(context_len + prompt_len)]
    pub slot_mapping: Tensor,   // u32, [num_tokens]
    pub logit_idxs: Tensor,     // u32, [batch_size]
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub seq_id_to_idx: HashMap<usize, usize>, // seq_id -> index into seqlens_*

    pub infer_log: Mutex<Vec<(String, Tensor)>>,
    pub step_no: usize,

    pub kv_cache: Box<dyn CacheIface>,

    // for paged attn
    pub paged_block_tables: Tensor, // [num_seqs, max_num_blocks_per_seq]
    pub paged_context_lens: Tensor, // [num_seqs]
    pub paged_block_size: usize,
    pub paged_max_context_len: usize,

    pub seqlen_multi: i64,
    pub q_multi: i64,
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

    pub fn extract_positions(&self, x: &Tensor) -> Tensor {
        x.i((&self.logit_idxs, ..))
    }
}

impl Debug for BatchInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchInfo")
            .field("step_no", &self.step_no)
            .field("tokens", &self.tokens)
            .field("positions", &self.positions)
            .field("seqlens_q", &self.seqlens_q)
            .field("seqlens_k", &self.seqlens_k)
            .field("gather_mapping", &self.gather_mapping.numel())
            .field("slot_mapping", &self.slot_mapping.numel())
            .field("max_seqlen_q", &self.max_seqlen_q)
            .field("max_seqlen_k", &self.max_seqlen_k)
            .field("paged_block_tables", &self.paged_block_tables)
            .field("paged_context_lens", &self.paged_context_lens)
            .field("paged_block_size", &self.paged_block_size)
            .field("paged_max_context_len", &self.paged_max_context_len)
            .field("seqlen_multi", &self.seqlen_multi)
            .field("q_multi", &self.q_multi)
            .finish()
    }
}

pub struct BatchInfoBuilder {
    entries: Vec<BatchEntry>,
    config: Arc<RllmConfig<TModel>>,
}

struct BatchEntry {
    seq_id: usize,
    query_pos_token: Vec<(usize, Token)>,
    kv_slots: Vec<usize>,
}

impl BatchInfoBuilder {
    pub fn new(config: Arc<RllmConfig<TModel>>) -> Self {
        Self {
            entries: Vec::new(),
            config,
        }
    }

    pub fn sched_out(
        &mut self,
        sched_out: &mut SchedulerOutputs,
        alloc: &BlockAllocator,
    ) -> &mut Self {
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
                sg.usage.gen_tokens += 1;
                sg.usage.prompt_tokens += q_len;

                let off = k_len - q_len;
                self.entries.push(BatchEntry {
                    seq_id: seq.seq_id.to_num(),
                    query_pos_token: (off..off + q_len)
                        .map(|idx| (idx, seq.get_token(idx)))
                        .collect(),
                    kv_slots: alloc.get_block_idxes(seq.seq_id, k_len),
                });

                seq.sync_computed_kv();
            }
        }

        self
    }

    pub fn profile_run(&mut self) -> BatchInfo {
        let sch_cfg = &self.config.clone().scheduler;
        let seq_len = sch_cfg.max_model_len;
        let max_num_seqs = sch_cfg.max_num_seqs;
        let avg_len = sch_cfg.max_num_kv_tokens / max_num_seqs;

        let fake_token = 12;
        let fake_slot = 0; // has to be 0 - we only have 1 slot in our fake kv cache
        let seq_id = 424242;

        for idx in 0..max_num_seqs {
            self.entries.push(BatchEntry {
                seq_id,
                query_pos_token: (0..1).map(|_| (idx, fake_token)).collect(),
                kv_slots: (0..avg_len).map(|_| fake_slot).collect(),
            });
        }

        let mut left = sch_cfg.max_num_batched_tokens - max_num_seqs;
        while left > 0 {
            let seq_len = std::cmp::min(seq_len, left);
            left -= seq_len;
            self.entries.push(BatchEntry {
                seq_id,
                query_pos_token: (0..seq_len).map(|idx| (idx, fake_token)).collect(),
                kv_slots: (0..seq_len).map(|_| fake_slot).collect(),
            });
        }

        let res = self.fake_finish();

        log::info!("profile: {res:?}");

        res
    }

    fn fake_finish(&mut self) -> BatchInfo {
        let (k, v) = CacheEngine::alloc_gpu_cache_layer(&self.config, 1);
        let kv_cache = Box::new(FakeKVCache { k, v });
        self.finish(0, kv_cache)
    }

    pub fn finish(&mut self, step_no: usize, kv_cache: Box<dyn CacheIface>) -> BatchInfo {
        let mut positions: Vec<i64> = Vec::new();
        let mut tokens: Vec<i32> = Vec::new();
        let mut logit_idxs: Vec<i32> = Vec::new();
        let mut seqlens_q: Vec<usize> = Vec::new();
        let mut seqlens_k: Vec<usize> = Vec::new();
        let mut gather_mapping: Vec<i32> = Vec::new();
        let mut slot_mapping: Vec<i32> = Vec::new();
        let mut seq_id_to_idx: HashMap<usize, usize> = HashMap::default();

        let mut paged_block_tables: Vec<Vec<i32>> = Vec::new();
        let mut paged_context_lens: Vec<i32> = Vec::new();

        let num_multitoken = if self.config.model.cache.paged_attn_kernel_v > 0 {
            // sort single-token entries to the back
            let (single, multi) = std::mem::take(&mut self.entries)
                .into_iter()
                .partition::<Vec<_>, _>(|e| e.query_pos_token.len() == 1);
            let multi_len = multi.len();
            self.entries = multi;
            self.entries.extend(single);
            multi_len
        } else {
            self.entries.len()
        };

        let mut first_single_token = 0;

        let max_seq = self.config.scheduler.max_model_len;
        let mut idx = 0;
        for e in &self.entries {
            seq_id_to_idx.insert(e.seq_id, idx);
            let query = &e.query_pos_token;
            let off = e.kv_slots.len() - query.len();
            for (qidx, (tpos, token)) in query.iter().enumerate() {
                assert!(*tpos < max_seq);
                positions.push(*tpos as i64);
                tokens.push(*token as i32);
                slot_mapping.push(e.kv_slots[off + qidx] as i32);
            }
            logit_idxs.push((tokens.len() - 1) as i32);
            if idx < num_multitoken {
                for slot in e.kv_slots.iter() {
                    gather_mapping.push(*slot as i32);
                }
                first_single_token = tokens.len();
                seqlens_q.push(query.len());
                seqlens_k.push(e.kv_slots.len());
            } else {
                let ctx_size = e.kv_slots.len();
                paged_context_lens.push(ctx_size as i32);
                let bl_size = self.config.model.cache.block_size;
                paged_block_tables.push(
                    (0..ctx_size)
                        .step_by(bl_size)
                        .map(|idx| {
                            let bl = e.kv_slots[idx];
                            assert!(bl % bl_size == 0);
                            (bl / bl_size) as i32
                        })
                        .collect(),
                );
            }
            idx += 1;
        }

        assert!(seqlens_q.len() + paged_context_lens.len() > 0);

        let device = self.config.model.device;
        let (max_seqlen_q, seqlens_q) = to_offsets(seqlens_q.into_iter(), device);
        let (max_seqlen_k, seqlens_k) = to_offsets(seqlens_k.into_iter(), device);

        // TODO positions, tokens should be padded to 8? see worker.py, search for multiple_of=8
        let positions = Tensor::from_slice(positions.as_slice()).to(device);
        let tokens = Tensor::from_slice(tokens.as_slice()).to(device);
        let slot_mapping = Tensor::from_slice(slot_mapping.as_slice()).to(device);
        let gather_mapping = Tensor::from_slice(gather_mapping.as_slice()).to(device);
        let logit_idxs = Tensor::from_slice(logit_idxs.as_slice()).to(device);

        let num_paged = paged_context_lens.len() as i64;
        let paged_max_context_len = *paged_context_lens.iter().max().unwrap_or(&0) as usize;
        let paged_block_tables_max_len = paged_block_tables
            .iter()
            .map(|v| v.len())
            .max()
            .unwrap_or(0);
        let flat_block_tables = paged_block_tables
            .into_iter()
            .flat_map(|mut v| {
                pad_to_multiple(&mut v, paged_block_tables_max_len);
                v.into_iter()
            })
            .collect::<Vec<_>>();
        let paged_block_tables = Tensor::from_slice(&flat_block_tables)
            .to(device)
            .reshape(&[num_paged, paged_block_tables_max_len as i64]);
        let paged_context_lens = Tensor::from_slice(paged_context_lens.as_slice()).to(device);

        BatchInfo {
            tokens,
            positions,
            seqlens_q,
            seqlens_k,
            logit_idxs,
            slot_mapping,
            gather_mapping,
            seqlen_multi: num_multitoken as i64,
            q_multi: first_single_token as i64,
            max_seqlen_q,
            max_seqlen_k,
            kv_cache,
            seq_id_to_idx,
            infer_log: Mutex::new(Vec::new()),
            step_no,
            paged_block_size: self.config.model.cache.block_size,
            paged_max_context_len,
            paged_block_tables,
            paged_context_lens,
        }
    }
}

struct FakeKVCache {
    k: Tensor,
    v: Tensor,
}

impl CacheIface for FakeKVCache {
    fn get(&self, _layer_no: usize) -> (Tensor, Tensor) {
        (self.k.shallow_clone(), self.v.shallow_clone())
    }
}
