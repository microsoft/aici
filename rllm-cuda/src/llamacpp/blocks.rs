use crate::{
    config::RllmConfig,
    paged::{CacheSize, SchedulerOutputs},
    seq::{Sequence, SequenceGroup},
    TBlockSpaceManager,
};

use super::tmodel::TModel;

/// Manages the mapping between logical and physical token blocks.
pub struct CppBlockSpaceManager {}

impl TBlockSpaceManager<TModel> for CppBlockSpaceManager {
    fn new(
        _block_size: usize,
        _cache_size: &CacheSize,
        _watermark: f32,
        _config: &RllmConfig<TModel>,
    ) -> Self {
        Self {}
    }

    fn can_allocate(&self, _seq_group: &SequenceGroup) -> bool {
        true
    }

    fn allocate(&mut self, seq_group: &mut SequenceGroup) {
        let seq = seq_group.only_seq();
        assert!(seq.num_kv_computed == 0);
        assert!(seq.gpu_blocks.is_empty());
        // seq_group.seqs[0].gpu_blocks = (0..seq.num_logical_blocks())
        //     .map(|_| self.alloc_gpu())
        //     .collect();
    }

    fn can_append_slot(&self, _seq_group: &SequenceGroup) -> bool {
        true
    }

    fn append_slots(&mut self, _seq: &mut Sequence, _outputs: &mut SchedulerOutputs) {}

    fn get_num_free_gpu_blocks(&self) -> usize {
        0
    }

    fn get_num_free_cpu_blocks(&self) -> usize {
        0
    }
}
