use rllm::{
    seq::{Sequence, SequenceGroup},
    SchedulerOutputs, TBlockSpaceManager,
};

use super::tmodel::TModel;

/// Manages the mapping between logical and physical token blocks.
pub struct CppBlockSpaceManager {}

impl TBlockSpaceManager<TModel> for CppBlockSpaceManager {
    fn can_allocate(&self, _seq_group: &SequenceGroup) -> bool {
        true
    }

    fn allocate(&mut self, seq_group: &mut SequenceGroup) {
        let seq = seq_group.only_seq();
        assert!(seq.num_kv_computed == 0);
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
