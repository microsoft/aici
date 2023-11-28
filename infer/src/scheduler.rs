use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::vec::Vec;

use log::warn;

use crate::blocks::BlockSpaceManager;
use crate::config::RllmConfig;
use crate::seq::{FinishReason, SchedulingPhase, SeqId, Sequence, SequenceGroup};

/// Preemption modes.
#[derive(Debug, Clone, Copy)]
pub enum PreemptionMode {
    /// Swap out the blocks of the preempted sequences to CPU memory
    /// and swap them back in when the sequences are resumed.
    Swap,

    /// Discard the blocks of the preempted sequences and
    /// recompute them when the sequences are resumed, treating the sequences as
    /// new prompts.
    Recompute,
}

/// Scheduler outputs.
pub struct SchedulerOutputs {
    pub prompt_run: bool,
    pub num_batched_tokens: usize,
    pub blocks_to_swap_in: HashMap<usize, usize>,
    pub blocks_to_swap_out: HashMap<usize, usize>,
    pub blocks_to_copy: HashMap<usize, Vec<usize>>,

    pub next_seq_groups: Vec<SequenceGroup>,
    pub dropped_seq_groups: Vec<SequenceGroup>,
}

impl SchedulerOutputs {
    fn validate(&self) {
        assert!(self.blocks_to_swap_in.is_empty() || self.blocks_to_swap_out.is_empty());
        // swapping not impl yet
        assert!(self.blocks_to_swap_in.is_empty());
        assert!(self.blocks_to_swap_out.is_empty());
        // CoW not impl yet
        assert!(self.blocks_to_copy.is_empty());
    }
    pub fn is_empty(&self) -> bool {
        // We do not consider the ignored sequence groups.
        self.next_seq_groups.is_empty()
            && self.blocks_to_swap_in.is_empty()
            && self.blocks_to_swap_out.is_empty()
            && self.blocks_to_copy.is_empty()
    }
}

/// Scheduler.
pub struct Scheduler {
    pub(crate) config: Arc<RllmConfig>,
    prompt_limit: usize,
    pub(crate) block_manager: BlockSpaceManager,
    freed_seq_ids: RefCell<Vec<SeqId>>,

    /// These have no KV cache stored anywhere. Each sequence group has only 1 sequence.
    waiting: Vec<SequenceGroup>,
    /// These currently sit on GPU but are not scheduled to run next. The ones to run next are in SchedulerOutputs.
    on_gpu: Vec<SequenceGroup>,
    /// These are swapped out to CPU memory.
    swapped: Vec<SequenceGroup>,
}

impl Scheduler {
    pub fn new(config: Arc<RllmConfig>) -> Self {
        let prompt_limit = std::cmp::min(
            config.scheduler.max_model_len,
            config.scheduler.max_num_batched_tokens,
        );
        let block_manager = BlockSpaceManager::new(
            config.cache.block_size,
            config.cache.num_gpu_blocks.unwrap(),
            config.cache.num_cpu_blocks.unwrap(),
            0.01,
            &config,
        );

        Self {
            config,
            prompt_limit,
            block_manager,
            freed_seq_ids: RefCell::new(Vec::new()),
            waiting: Vec::new(),
            on_gpu: Vec::new(),
            swapped: Vec::new(),
        }
    }

    pub fn add_seq_group(&mut self, seq_group: SequenceGroup) {
        log::debug!("add_seq_group: {}", seq_group.request_id);
        self.waiting.push(seq_group);
    }

    // pub fn abort_seq_groups(&mut self, request_ids: Vec<String>) {
    //     let request_ids: HashSet<String> = request_ids.into_iter().collect();
    //     for state_queue in [&mut self.waiting, &mut self.on_gpu, &mut self.swapped].iter_mut() {
    //         state_queue.retain(|seq_group| {
    //             if request_ids.contains(&seq_group.request_id) {
    //                 for seq in &mut seq_group.seqs {
    //                     if !seq.is_finished() {
    //                         seq.status = SequenceStatus::Finished(FinishReason::Aborted);
    //                         self.free_seq(seq);
    //                     }
    //                 }
    //                 false
    //             } else {
    //                 true
    //             }
    //         });
    //     }
    // }

    pub fn has_unfinished_seqs(&self) -> bool {
        self.get_num_unfinished_seq_groups() > 0
    }

    pub fn get_num_unfinished_seq_groups(&self) -> usize {
        self.waiting.len() + self.on_gpu.len() + self.swapped.len()
    }

    fn drop_finished(outputs: &mut SchedulerOutputs, q: &mut Vec<SequenceGroup>) {
        if q.iter().any(|sg| sg.is_finished()) {
            let mut not_finished = Vec::new();
            for e in q.drain(..) {
                if e.is_finished() {
                    outputs.dropped_seq_groups.push(e);
                } else {
                    not_finished.push(e);
                }
            }
            assert!(q.is_empty());
            log::debug!(
                "dropped {} seq groups; now {}",
                outputs.dropped_seq_groups.len(),
                q.len()
            );
            q.extend(not_finished);
        }
    }

    fn step_drop_finished(&mut self, outputs: &mut SchedulerOutputs) {
        let mut waiting = std::mem::replace(&mut self.waiting, Vec::new());
        for seq_group in waiting.iter_mut() {
            assert!(seq_group.seqs.len() == 1);
            let num_prompt_tokens = seq_group.get_seqs(None)[0].get_len();
            if num_prompt_tokens > self.prompt_limit {
                warn!(
                    "Sequence group {} has a prompt that is too long ({} > {})",
                    seq_group.request_id, num_prompt_tokens, self.prompt_limit
                );
                self.set_phase(seq_group, SchedulingPhase::Finished(FinishReason::Failed));
            }
        }
        self.waiting = waiting;

        Self::drop_finished(outputs, &mut self.waiting);
        Self::drop_finished(outputs, &mut self.on_gpu);
        Self::drop_finished(outputs, &mut self.swapped);
    }

    fn step_start_waiting(&mut self, outputs: &mut SchedulerOutputs) {
        log::trace!("step_start_waiting ({} seqs)", self.waiting.len());

        let mut num_curr_seqs = self
            .on_gpu
            .iter()
            .map(|sg| sg.get_max_num_running_seqs())
            .sum::<usize>();

        while let Some(mut seq_group) = self.waiting.pop() {
            assert!(seq_group.seqs.len() == 1);
            let num_prompt_tokens = seq_group.get_seqs(None)[0].get_len();
            let num_new_seqs = seq_group.get_max_num_running_seqs();

            log::trace!(
                "seq_group {} has {} prompt tokens and {} new seqs",
                seq_group.request_id,
                num_prompt_tokens,
                num_new_seqs
            );

            // Check allocation and batch token limits
            if !self.block_manager.can_allocate(&seq_group)
                || outputs.num_batched_tokens + num_prompt_tokens
                    > self.config.scheduler.max_num_batched_tokens
                || num_curr_seqs + num_new_seqs > self.config.scheduler.max_num_seqs
            {
                self.waiting.push(seq_group); // Put back the sequence group
                break;
            }

            self._allocate(&mut seq_group);
            outputs.next_seq_groups.push(seq_group);
            outputs.num_batched_tokens += num_prompt_tokens;
            num_curr_seqs += num_new_seqs;
        }
    }

    fn sort_by_priority(seq_groups: &mut Vec<SequenceGroup>) {
        // note that we take elements first from the end of the queue (Vec::pop())
        seq_groups.sort_by_key(|g| g.arrival_time);
        seq_groups.reverse();
    }

    fn step_preempt(&mut self, outputs: &mut SchedulerOutputs) -> bool {
        let mut did_preempt = false;
        Self::sort_by_priority(&mut self.on_gpu);

        while let Some(mut seq_group) = self.on_gpu.pop() {
            while !self.block_manager.can_append_slot(&seq_group) {
                did_preempt = true;
                if !self.on_gpu.is_empty() {
                    // take the first group in queue (lowest priority)
                    let victim_seq_group = self.on_gpu.remove(0);
                    self._preempt(victim_seq_group, outputs);
                } else {
                    // preempt the current sequence group and stop
                    self._preempt(seq_group, outputs);
                    return did_preempt;
                }
            }

            self._append_slot(&mut seq_group, outputs);
            outputs.next_seq_groups.push(seq_group);
        }
        return did_preempt;
    }

    fn _allocate(&mut self, seq_group: &mut SequenceGroup) {
        self.block_manager.allocate(seq_group);
        self.set_phase(seq_group, SchedulingPhase::Running);
    }

    fn _append_slot(&mut self, seq_group: &mut SequenceGroup, outputs: &mut SchedulerOutputs) {
        for seq in &mut seq_group.seqs {
            if seq.sched_phase == SchedulingPhase::Running {
                if let Some((src_block, dst_block)) = self.block_manager.append_slot(seq) {
                    outputs
                        .blocks_to_copy
                        .entry(src_block)
                        .or_insert_with(Vec::new)
                        .push(dst_block);
                }
            }
        }
    }

    fn _swap_in(&mut self, seq_group: &mut SequenceGroup, outputs: &mut SchedulerOutputs) {
        let src_to_dst = self.block_manager.swap_in(seq_group);
        outputs.blocks_to_swap_in.extend(src_to_dst);
    }

    fn _preempt(&mut self, mut seq_group: SequenceGroup, outputs: &mut SchedulerOutputs) {
        let mode = if seq_group.get_max_num_running_seqs() == 1 {
            PreemptionMode::Recompute
        } else {
            PreemptionMode::Swap
        };

        match mode {
            PreemptionMode::Swap => {
                if !self.block_manager.can_swap_out(&seq_group) {
                    panic!("Aborted due to the lack of CPU swap space. Please increase the swap space to avoid this error.");
                }
                let map = self.block_manager.swap_out(&mut seq_group);
                outputs.blocks_to_swap_out.extend(map);
                self.swapped.push(seq_group);
            }
            PreemptionMode::Recompute => {
                self.set_phase(&mut seq_group, SchedulingPhase::Waiting);
                self.waiting.push(seq_group);
            }
        }
    }

    fn step_swap_in(&mut self, outputs: &mut SchedulerOutputs) {
        Self::sort_by_priority(&mut self.swapped);

        let mut num_curr_seqs = self
            .on_gpu
            .iter()
            .map(|sg| sg.get_max_num_running_seqs())
            .sum::<usize>();
        while let Some(mut seq_group) = self.swapped.pop() {
            let num_new_seqs = seq_group.get_max_num_running_seqs();
            if !self.block_manager.can_swap_in(&seq_group)
                || num_curr_seqs + num_new_seqs > self.config.scheduler.max_num_seqs
            {
                self.swapped.push(seq_group);
                break;
            }
            self._swap_in(&mut seq_group, outputs);
            self._append_slot(&mut seq_group, outputs);
            num_curr_seqs += num_new_seqs;
            self.on_gpu.push(seq_group);
        }
    }

    pub fn step_finished(&mut self, mut outputs: SchedulerOutputs) {
        // everything that used to be "next_step" is now just on the GPU
        self.on_gpu.append(&mut outputs.next_seq_groups);
    }

    pub fn schedule(&mut self) -> SchedulerOutputs {
        let mut outputs = SchedulerOutputs {
            prompt_run: false,
            num_batched_tokens: 0,
            blocks_to_swap_in: HashMap::new(),
            blocks_to_swap_out: HashMap::new(),
            blocks_to_copy: HashMap::new(),
            dropped_seq_groups: Vec::new(),
            next_seq_groups: Vec::new(),
        };

        self.step_drop_finished(&mut outputs);

        if self.swapped.is_empty() {
            self.step_start_waiting(&mut outputs);
        }

        if outputs.next_seq_groups.is_empty() {
            // Preemption logic
            let did_preempt = self.step_preempt(&mut outputs);

            // Swap in logic for swapped sequences
            if !did_preempt {
                self.step_swap_in(&mut outputs);
            }

            // Update num_batched_tokens based on the sequences in the RUNNING state
            outputs.num_batched_tokens = self
                .on_gpu
                .iter()
                .flat_map(|sg| sg.get_seqs(Some(SchedulingPhase::Running)))
                .map(|seq| seq.get_len())
                .sum();
        }

        outputs.validate();
        outputs
    }

    pub fn finish_seq(&self, seq: &mut Sequence, reason: FinishReason) {
        if seq.is_finished() {
            return;
        }
        seq.sched_phase = SchedulingPhase::Finished(reason);
        self.freed_seq_ids.borrow_mut().push(seq.seq_id);
        seq.gpu_blocks.clear();
        seq.cpu_blocks.clear();
    }

    /// Sets the phase of all sequences in a group.
    fn set_phase(&self, seq_group: &mut SequenceGroup, status: SchedulingPhase) {
        let to_waiting = match status {
            SchedulingPhase::Waiting => true,
            SchedulingPhase::Running => false,
            SchedulingPhase::Swapped => false,
            SchedulingPhase::Finished(reason) => {
                log::debug!("seq_group {} finished: {:?}", seq_group.request_id, reason);
                seq_group
                    .seqs
                    .iter_mut()
                    .for_each(|seq| self.finish_seq(seq, reason));
                return;
            }
        };
        for seq in seq_group.seqs.iter_mut() {
            assert!(!seq.is_finished());
            seq.sched_phase = status;
            if to_waiting {
                seq.gpu_blocks.clear();
            }
        }
    }
}
