#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;
use std::vec::Vec;

use log::warn;

use crate::blocks::BlockSpaceManager;
use crate::config::RllmConfig;
use crate::seq::{FinishReason, SeqId, Sequence, SequenceGroup, SchedulingPhase};

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
    pub num_scheduled_seq_groups: usize,
    pub prompt_run: bool,
    pub num_batched_tokens: usize,
    pub blocks_to_swap_in: HashMap<usize, usize>,
    pub blocks_to_swap_out: HashMap<usize, usize>,
    pub blocks_to_copy: HashMap<usize, Vec<usize>>,
    pub ignored_seq_groups: Vec<SequenceGroup>,
}

impl SchedulerOutputs {
    fn validate(&self) {
        assert!(self.blocks_to_swap_in.is_empty() || self.blocks_to_swap_out.is_empty());
    }
    pub fn is_empty(&self) -> bool {
        // We do not consider the ignored sequence groups.
        self.num_scheduled_seq_groups == 0
            && self.blocks_to_swap_in.is_empty()
            && self.blocks_to_swap_out.is_empty()
            && self.blocks_to_copy.is_empty()
    }
}

/// Scheduler.
pub struct Scheduler {
    config: Arc<RllmConfig>,
    prompt_limit: usize,
    block_manager: BlockSpaceManager,
    freed_seq_ids: Vec<SeqId>,

    /// These have no KV cache stored anywhere. Each sequence group has only 1 sequence.
    waiting: Vec<SequenceGroup>,
    /// Scheduled to run on the next step.
    next_step: Vec<SequenceGroup>,
    /// These currently sit on GPU but are not scheduled to run next.
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
        );

        Self {
            config,
            prompt_limit,
            block_manager,
            freed_seq_ids: Vec::new(),
            waiting: Vec::new(),
            on_gpu: Vec::new(),
            next_step: Vec::new(),
            swapped: Vec::new(),
        }
    }

    pub fn add_seq_group(&mut self, seq_group: SequenceGroup) {
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

    pub fn free_seq(&mut self, seq: &mut Sequence) {
        self.freed_seq_ids.push(seq.seq_id);
        seq.phys_blocks.clear();
    }

    pub fn free_finished_seq_groups(&mut self) {
        self.on_gpu.retain(|seq_group| !seq_group.is_finished());
    }

    pub fn schedule(&mut self) -> SchedulerOutputs {
        // Schedule sequence groups
        let scheduler_outputs = self._schedule();

        // // Create seq_group_metadata_list based on scheduled sequence groups
        // let mut seq_group_metadata_list: Vec<SequenceGroupMetadata> = Vec::new();
        // for seq_group in &scheduler_outputs.scheduled_seq_groups {
        //     let mut seq_data: HashMap<i32, SequenceData> = HashMap::new();
        //     let mut block_tables: HashMap<i32, Vec<i32>> = HashMap::new();
        //     let seqs = seq_group.get_seqs(SequenceStatus::Running);
        //     let is_ff = seqs.iter().any(|seq| seq.data.num_pending_ff_tokens > 0);

        //     for seq in seqs {
        //         if seq.skip_round {
        //             continue;
        //         }
        //         seq_data.insert(seq.seq_id, seq.data.clone());
        //         block_tables.insert(seq.seq_id, self.block_manager.get_block_table(seq));
        //     }

        //     let seq_group_metadata = SequenceGroupMetadata {
        //         request_id: seq_group.request_id.clone(),
        //         is_prompt: scheduler_outputs.prompt_run,
        //         seq_data,
        //         sampling_params: seq_group.sampling_params.clone(),
        //         block_tables,
        //         is_ff,
        //     };
        //     seq_group_metadata_list.push(seq_group_metadata);
        // }

        scheduler_outputs
    }

    fn step_waiting(&mut self, outputs: &mut SchedulerOutputs) {
        let mut num_curr_seqs = self
            .on_gpu
            .iter()
            .map(|sg| sg.get_max_num_running_seqs())
            .sum::<usize>();

        while let Some(mut seq_group) = self.waiting.pop() {
            assert!(seq_group.seqs.len() == 1);
            let num_prompt_tokens = seq_group.get_seqs(None)[0].get_len();
            if num_prompt_tokens > self.prompt_limit {
                warn!(
                    "Sequence group {} has a prompt that is too long ({} > {})",
                    seq_group.request_id, num_prompt_tokens, self.prompt_limit
                );
                self.set_phase(
                    &mut seq_group,
                    SchedulingPhase::Finished(FinishReason::Ignored),
                );
                outputs.ignored_seq_groups.push(seq_group);
                continue;
            }

            let num_new_seqs = seq_group.get_max_num_running_seqs();

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
            self.next_step.push(seq_group);
            outputs.num_batched_tokens += num_prompt_tokens;
            num_curr_seqs += num_new_seqs;
        }
    }

    fn sort_by_priority(seq_groups: &mut Vec<SequenceGroup>) {
        // TODO check which direction?
        seq_groups.sort_by_key(|g| g.arrival_time);
    }

    fn step_preempt(&mut self, outputs: &mut SchedulerOutputs) -> bool {
        let mut did_preempt = false;
        Self::sort_by_priority(&mut self.on_gpu);

        while let Some(mut seq_group) = self.on_gpu.pop() {
            while !self.block_manager.can_append_slot(&seq_group) {
                did_preempt = true;
                if !self.on_gpu.is_empty() {
                    let victim_seq_group = self.on_gpu.remove(0);
                    self._preempt(victim_seq_group, outputs);
                } else {
                    // preempt the current sequence group and stop
                    self._preempt(seq_group, outputs);
                    return did_preempt;
                }
            }

            self._append_slot(&mut seq_group, outputs);
            self.next_step.push(seq_group);
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

    fn _schedule(&mut self) -> SchedulerOutputs {
        let mut outputs = SchedulerOutputs {
            num_scheduled_seq_groups: 0,
            prompt_run: false,
            num_batched_tokens: 0,
            blocks_to_swap_in: HashMap::new(),
            blocks_to_swap_out: HashMap::new(),
            blocks_to_copy: HashMap::new(),
            ignored_seq_groups: Vec::new(),
        };

        // first, everything that used to be "next_step" is now just on the GPU
        self.on_gpu.append(&mut self.next_step);

        // Join waiting sequences if possible
        if self.swapped.is_empty() {
            self.step_waiting(&mut outputs);
        }

        if self.next_step.is_empty() {
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

        outputs
    }

    /// Sets the status of all sequences.
    fn set_phase(&self, seq_group: &mut SequenceGroup, status: SchedulingPhase) {
        let clear = match status {
            SchedulingPhase::Waiting => true,
            SchedulingPhase::Running => false,
            SchedulingPhase::Swapped => false,
            SchedulingPhase::Finished(_) => true,
        };
        for seq in seq_group.seqs.iter_mut() {
            seq.sched_phase = status;
            if clear {
                seq.phys_blocks.clear();
            }
        }
    }
}
