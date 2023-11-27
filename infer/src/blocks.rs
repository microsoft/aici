#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::vec::Vec;

use crate::cache_engine::CacheEngine;
use crate::config::RllmConfig;
use crate::seq::{SchedulingPhase, Sequence, SequenceGroup};

#[derive(Debug, Clone, Copy)]
pub enum BlockLocation {
    GPU,
    CPU,
}

/// Represents the state of a block in the KV cache.
#[derive(Debug)]
pub struct PhysicalTokenBlock {
    device: BlockLocation,
    block_number: usize,
    block_size: usize,
    ref_count: usize,
}

impl PhysicalTokenBlock {
    pub fn new(device: BlockLocation, block_number: usize, block_size: usize) -> Self {
        Self {
            device,
            block_number,
            block_size,
            ref_count: 0,
        }
    }
}

/// Manages free physical token blocks for a device.
///
/// The allocator maintains a list of free blocks and allocates a block when
/// requested. When a block is freed, its reference count is decremented. If
/// the reference count becomes zero, the block is added back to the free list.
struct BlockAllocator {
    device: BlockLocation,
    block_size: usize,
    num_blocks: usize,
    free_list: Vec<usize>,
    all_blocks: Vec<PhysicalTokenBlock>,
}

pub struct BlockRef {
    allocator: Arc<Mutex<BlockAllocator>>,
    block_idx: usize,
}

impl Drop for BlockRef {
    fn drop(&mut self) {
        let mut alloc = self.allocator.lock().unwrap();
        let blk = &mut alloc.all_blocks[self.block_idx];
        assert!(blk.ref_count > 0);
        blk.ref_count -= 1;
        if blk.ref_count == 0 {
            alloc.free_list.push(self.block_idx);
        }
    }
}

impl BlockRef {
    pub fn fork(&self) -> Self {
        let mut alloc = self.allocator.lock().unwrap();
        let blk = &mut alloc.all_blocks[self.block_idx];
        assert!(blk.ref_count > 0);
        blk.ref_count += 1;
        Self {
            allocator: self.allocator.clone(),
            block_idx: self.block_idx,
        }
    }

    pub fn is_singlular(&self) -> bool {
        let mut alloc = self.allocator.lock().unwrap();
        let blk = &mut alloc.all_blocks[self.block_idx];
        assert!(blk.ref_count > 0);
        blk.ref_count == 1
    }

    pub fn get_index(&self) -> usize {
        self.block_idx
    }
}

impl BlockAllocator {
    pub fn new(device: BlockLocation, block_size: usize, num_blocks: usize) -> Self {
        let all_blocks = (0..num_blocks)
            .map(|i| PhysicalTokenBlock::new(device, i, block_size))
            .collect();
        Self {
            device,
            block_size,
            num_blocks,
            all_blocks,
            free_list: (0..num_blocks).collect(),
        }
    }

    pub fn get_num_free_blocks(&self) -> usize {
        self.free_list.len()
    }
}

fn allocate_block(m: &Arc<Mutex<BlockAllocator>>) -> BlockRef {
    let mut a = m.lock().unwrap();
    let block_idx = a
        .free_list
        .pop()
        .expect("Out of memory! No free blocks are available.");
    assert!(a.all_blocks[block_idx].ref_count == 0);
    a.all_blocks[block_idx].ref_count += 1;
    BlockRef {
        allocator: m.clone(),
        block_idx,
    }
}

/// Manages the mapping between logical and physical token blocks.
pub struct BlockSpaceManager {
    watermark_blocks: usize,
    gpu_allocator: Arc<Mutex<BlockAllocator>>,
    cpu_allocator: Arc<Mutex<BlockAllocator>>,
    block_size: usize,
}

impl BlockSpaceManager {
    fn new_allocator(
        location: BlockLocation,
        block_size: usize,
        num_blocks: usize,
        config: &RllmConfig,
    ) -> Arc<Mutex<BlockAllocator>> {
        log::info!(
            "{:?} {} blocks, {} MiB",
            location,
            num_blocks,
            (num_blocks * CacheEngine::get_cache_block_size(config)) >> 20
        );
        Arc::new(Mutex::new(BlockAllocator::new(
            location, block_size, num_blocks,
        )))
    }

    pub fn new(
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
        watermark: f32,
        config: &RllmConfig,
    ) -> Self {
        assert!(watermark >= 0.0);
        let watermark_blocks = (watermark * num_gpu_blocks as f32) as usize;

        log::info!("BlockSpaceManager: block_size: {} tokens", block_size);

        Self {
            watermark_blocks,
            block_size,
            gpu_allocator: Self::new_allocator(
                BlockLocation::GPU,
                block_size,
                num_gpu_blocks,
                config,
            ),
            cpu_allocator: Self::new_allocator(
                BlockLocation::CPU,
                block_size,
                num_cpu_blocks,
                config,
            ),
        }
    }

    fn num_logical_blocks(&self, seq: &Sequence) -> usize {
        (seq.tokens.len() + self.block_size - 1) / self.block_size
    }

    fn can_alloc_gpu(&self, num_required_blocks: usize) -> bool {
        self.get_num_free_gpu_blocks() >= num_required_blocks
    }

    pub fn can_allocate(&self, seq_group: &SequenceGroup) -> bool {
        let num_required_blocks = self.num_logical_blocks(seq_group.only_seq());
        self.can_alloc_gpu(num_required_blocks + self.watermark_blocks)
    }

    fn alloc_gpu(&mut self) -> BlockRef {
        allocate_block(&self.gpu_allocator)
    }

    fn alloc_cpu(&mut self) -> BlockRef {
        allocate_block(&self.cpu_allocator)
    }

    pub fn allocate(&mut self, seq_group: &mut SequenceGroup) {
        let seq = seq_group.only_seq();
        assert!(seq.gpu_blocks.is_empty());
        seq_group.seqs[0].gpu_blocks = (0..self.num_logical_blocks(seq))
            .map(|_| self.alloc_gpu())
            .collect();
    }

    pub fn can_append_slot(&self, seq_group: &SequenceGroup) -> bool {
        let num_seqs = seq_group.num_seqs(Some(SchedulingPhase::Running));
        // TODO this is not correct - more than one token can be appended
        self.can_alloc_gpu(num_seqs)
    }

    pub fn trim_physical_blocks(&mut self, seq: &mut Sequence) {
        let num_logical = self.num_logical_blocks(seq);
        if seq.gpu_blocks.len() > num_logical {
            seq.gpu_blocks.truncate(num_logical);
        }
        if seq.cpu_blocks.len() > num_logical {
            seq.cpu_blocks.truncate(num_logical);
        }
    }

    pub fn append_slot(&mut self, seq: &mut Sequence) -> Option<(usize, usize)> {
        let num_logical = self.num_logical_blocks(seq);
        let block_table = &mut seq.gpu_blocks;
        assert!(block_table.len() > 0); // TODO?

        if block_table.len() < num_logical {
            block_table.push(self.alloc_gpu());
            assert!(block_table.len() == num_logical);
            return None;
        }

        assert!(block_table.len() == num_logical);
        let last_block = block_table.last_mut().unwrap();
        if last_block.is_singlular() {
            None
        } else {
            let new_block = self.alloc_gpu();
            let old_block_number = last_block.block_idx;
            let new_block_number = new_block.block_idx;
            *last_block = new_block;
            Some((old_block_number, new_block_number))
        }
    }

    fn num_phys_blocks(&self, seq_group: &SequenceGroup) -> usize {
        seq_group
            .get_seqs(None)
            .iter()
            .map(|seq| {
                let n = seq.gpu_blocks.len();
                if n == 0 {
                    seq.cpu_blocks.len()
                } else {
                    assert!(seq.cpu_blocks.is_empty());
                    n
                }
            })
            .sum()
    }

    pub fn can_swap_in(&self, seq_group: &SequenceGroup) -> bool {
        let blocks = self.num_phys_blocks(seq_group);
        let num_swapped_seqs = seq_group.num_seqs(Some(SchedulingPhase::Swapped));
        let num_required_blocks = blocks + num_swapped_seqs;
        self.can_alloc_gpu(num_required_blocks + self.watermark_blocks)
    }

    fn to_idx_map(mapping: HashMap<usize, BlockRef>) -> HashMap<usize, usize> {
        mapping
            .iter()
            .map(|(the_old, the_new)| (*the_old, the_new.block_idx))
            .collect()
    }

    fn map_blocks(
        &mut self,
        mut blocks: Vec<BlockRef>,
        mapping: &mut HashMap<usize, BlockRef>,
        to_gpu: bool,
    ) -> Vec<BlockRef> {
        blocks
            .drain(..)
            .map(|b| match mapping.get(&b.block_idx) {
                Some(block) => block.fork(),
                None => {
                    let new_block = if to_gpu {
                        self.alloc_gpu()
                    } else {
                        self.alloc_cpu()
                    };
                    mapping.insert(b.block_idx, new_block.fork());
                    new_block
                }
            })
            .collect()
    }

    pub fn swap_in(&mut self, seq_group: &mut SequenceGroup) -> HashMap<usize, usize> {
        let mut mapping = HashMap::new();

        for seq in &mut seq_group.seqs {
            if seq.sched_phase == SchedulingPhase::Swapped {
                assert!(seq.cpu_blocks.is_empty());
                let bl = std::mem::take(&mut seq.gpu_blocks);
                seq.cpu_blocks = self.map_blocks(bl, &mut mapping, false);
                seq.sched_phase = SchedulingPhase::Running;
            }
        }

        Self::to_idx_map(mapping)
    }

    pub fn swap_out(&mut self, seq_group: &mut SequenceGroup) -> HashMap<usize, usize> {
        let mut mapping = HashMap::new();

        for seq in &mut seq_group.seqs {
            if seq.sched_phase == SchedulingPhase::Running {
                assert!(seq.cpu_blocks.is_empty());
                let bl = std::mem::take(&mut seq.cpu_blocks);
                seq.gpu_blocks = self.map_blocks(bl, &mut mapping, true);
                seq.sched_phase = SchedulingPhase::Swapped;
            }
        }

        Self::to_idx_map(mapping)
    }

    pub fn can_swap_out(&self, seq_group: &SequenceGroup) -> bool {
        let blocks = self.num_phys_blocks(seq_group);
        blocks <= self.get_num_free_cpu_blocks()
    }

    pub fn get_num_free_gpu_blocks(&self) -> usize {
        self.gpu_allocator.lock().unwrap().get_num_free_blocks()
    }

    pub fn get_num_free_cpu_blocks(&self) -> usize {
        self.cpu_allocator.lock().unwrap().get_num_free_blocks()
    }

    pub fn get_gpu_blocks(&self, seq: &Sequence) -> Vec<usize> {
        assert!(seq.gpu_blocks.len() > 0);
        assert!(seq.sched_phase == SchedulingPhase::Running);
        seq.gpu_blocks.iter().map(|x| x.block_idx).collect()
    }
}
