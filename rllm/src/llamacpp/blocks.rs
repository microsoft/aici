use crate::{
    config::RllmConfig,
    paged::{SchedulerOutputs, CacheSize},
    seq::{Sequence, SequenceGroup},
    HashMap,
};

#[derive(Debug, Clone, Copy)]
pub enum BlockLocation {
    GPU,
    CPU,
}

/// Manages the mapping between logical and physical token blocks.
pub struct BlockSpaceManager {}

impl BlockSpaceManager {
    pub fn new(
        _block_size: usize,
        _cache_size: &CacheSize,
        _watermark: f32,
        _config: &RllmConfig,
    ) -> Self {
        Self {}
    }

    pub fn can_allocate(&self, seq_group: &SequenceGroup) -> bool {
        true
    }

    pub fn allocate(&mut self, seq_group: &mut SequenceGroup) {
        let seq = seq_group.only_seq();
        assert!(seq.num_kv_computed == 0);
        assert!(seq.gpu_blocks.is_empty());
        // seq_group.seqs[0].gpu_blocks = (0..seq.num_logical_blocks())
        //     .map(|_| self.alloc_gpu())
        //     .collect();
    }

    pub fn can_append_slot(&self, _seq_group: &SequenceGroup) -> bool {
        true
    }

    pub fn append_slots(&mut self, _seq: &mut Sequence, _outputs: &mut SchedulerOutputs) {}

    pub fn can_swap_in(&self, seq_group: &SequenceGroup) -> bool {
        false
    }

    pub fn swap_in(&mut self, seq_group: &mut SequenceGroup) -> HashMap<usize, usize> {
        panic!("llama.cpp swap_in")
    }

    pub fn swap_out(&mut self, seq_group: &mut SequenceGroup) -> HashMap<usize, usize> {
        panic!("llama.cpp swap_out")
    }

    pub fn can_swap_out(&self, seq_group: &SequenceGroup) -> bool {
        false
    }

    pub fn get_num_free_gpu_blocks(&self) -> usize {
        0
    }

    pub fn get_num_free_cpu_blocks(&self) -> usize {
        0
    }
}
