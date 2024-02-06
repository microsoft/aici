use std::{fmt::Display, sync::Arc};

use aicirt::TimerRef;
use anyhow::Result;

use crate::{
    config::{ModelMeta, RllmConfig},
    scheduler::{CacheSize, SchedulerOutputs},
    seq::{Sequence, SequenceGroup},
    HashMap, LoaderArgs, LogitsProcessor, RllmEngine,
};

#[derive(Debug, Clone, Copy)]
pub enum BlockLocation {
    GPU,
    CPU,
}

pub trait TensorOps {
    fn to_vec1(&self) -> Vec<f32>;
}

pub trait AiciBias<T: TensorOps> {
    fn apply(&self, logits: &mut T, seq_id: usize);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqId(pub usize);

impl SeqId {
    pub fn to_num(&self) -> usize {
        self.0
    }
}

impl Display for SeqId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub trait SequenceManager {
    fn new_sequence(&self) -> SeqId;
    fn copy(&self, src: SeqId, dst: SeqId, length: usize);
    fn trim(&self, seq: SeqId, length: usize);
    fn delete(&self, seq: SeqId);
}

pub trait ModelExec: Sized {
    type Tensor: TensorOps;
    type BlockSpaceManager: TBlockSpaceManager<Self>;
    type AiciBias: AiciBias<Self::Tensor>;
    type ModelConfig;
    type ModelLoaderArgs: Send + 'static;
    type SequenceManager: SequenceManager;

    fn load_model_config(
        args: &LoaderArgs,
        model_args: &mut Self::ModelLoaderArgs,
    ) -> Result<(ModelMeta, Self::ModelConfig)>;
    fn verify_args(args: &RllmConfig<Self>) -> Result<()>;
    fn load_rllm_engine(
        args: LoaderArgs,
        model_args: Self::ModelLoaderArgs,
    ) -> Result<RllmEngine<Self>>;

    fn sequence_manager(&self) -> Arc<Self::SequenceManager>;

    fn run(
        &mut self,
        _vocab_size: usize,
        tim: &TimerRef,
        step_no: usize,
        sched_out: &mut SchedulerOutputs,
    ) -> Result<()>;
    fn get_logits(&self, seq_id: usize) -> Self::Tensor;
    fn finalize_run(&mut self) -> Result<()>;

    fn empty_bias(&self, vocab_size: usize) -> Self::AiciBias;
    fn new_bias(&self, slice: &'static [f32], num_seqs: usize, vocab_size: usize)
        -> Self::AiciBias;

    fn sample(&self, processor: &mut LogitsProcessor, logits: &Self::Tensor) -> Result<u32>;
}

pub trait TBlockSpaceManager<ME: ModelExec> {
    fn new(
        _block_size: usize,
        _cache_size: &CacheSize,
        _watermark: f32,
        _config: &RllmConfig<ME>,
    ) -> Self;

    fn can_allocate(&self, _seq_group: &SequenceGroup) -> bool;
    fn allocate(&mut self, seq_group: &mut SequenceGroup);

    fn can_append_slot(&self, _seq_group: &SequenceGroup) -> bool;
    fn append_slots(&mut self, _seq: &mut Sequence, _outputs: &mut SchedulerOutputs);
    fn get_num_free_gpu_blocks(&self) -> usize;
    fn get_num_free_cpu_blocks(&self) -> usize;

    fn can_swap_in(&self, _seq_group: &SequenceGroup) -> bool {
        false
    }

    fn swap_in(&mut self, _seq_group: &mut SequenceGroup) -> HashMap<usize, usize> {
        panic!("no swap_in")
    }

    fn swap_out(&mut self, _seq_group: &mut SequenceGroup) -> HashMap<usize, usize> {
        panic!("no swap_out")
    }

    fn can_swap_out(&self, _seq_group: &SequenceGroup) -> bool {
        false
    }
}
