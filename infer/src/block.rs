#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::vec::Vec;

use crate::seq::Token;

#[derive(Debug, Clone, Copy)]
pub enum BlockLocation {
    GPU,
    CPU,
}

/// A block that stores a contiguous chunk of tokens from left to right.
///
/// Logical blocks are used to represent the states of the corresponding
/// physical blocks in the KV cache.
#[derive(Debug, Clone)]
pub struct LogicalTokenBlock {
    block_number: usize,
    block_size: usize,
    token_ids: Vec<Token>,
}

impl LogicalTokenBlock {
    pub fn new(block_number: usize, block_size: usize) -> Self {
        Self {
            block_number,
            block_size,
            token_ids: vec![],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    pub fn get_num_empty_slots(&self) -> usize {
        self.block_size - self.token_ids.len()
    }

    pub fn is_full(&self) -> bool {
        self.token_ids.len() == self.block_size
    }

    pub fn append_tokens(&mut self, token_ids: &[Token]) {
        assert!(token_ids.len() <= self.get_num_empty_slots());
        self.token_ids.extend_from_slice(token_ids);
    }

    pub fn get_token_ids(&self) -> &[Token] {
        &self.token_ids
    }

    pub fn get_last_token_id(&self) -> Token {
        *self.token_ids.last().unwrap()
    }
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

type BlockTable = Vec<BlockRef>;

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

    pub fn allocate(m: &Arc<Mutex<Self>>) -> BlockRef {
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

    pub fn get_num_free_blocks(&self) -> usize {
        self.free_list.len()
    }
}

/// Manages the mapping between logical and physical token blocks.
pub struct BlockSpaceManager {
    watermark_blocks: usize,
    gpu_allocator: BlockAllocator,
    cpu_allocator: BlockAllocator,
}

impl BlockSpaceManager {
    pub fn new(
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
        watermark: f32,
    ) -> Self {
        assert!(watermark >= 0.0);
        let watermark_blocks = (watermark * num_gpu_blocks as f32) as usize;

        Self {
            watermark_blocks,
            gpu_allocator: BlockAllocator::new(BlockLocation::GPU, block_size, num_gpu_blocks),
            cpu_allocator: BlockAllocator::new(BlockLocation::CPU, block_size, num_cpu_blocks),
        }
    }

    // pub fn can_allocate(&self, seq_group: &SequenceGroup) -> bool {
    //     let num_required_blocks = seq_group.get_seqs()[0].logical_token_blocks.len();
    //     let num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks();
    //     num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks
    // }

    // pub fn allocate(&mut self, seq_group: &SequenceGroup) {
    //     let seq = &seq_group.get_seqs()[0];

    //     let mut block_table = BlockTable::new();
    //     for _ in 0..seq.logical_token_blocks.len() {
    //         let mut block = self.gpu_allocator.allocate();
    //         block.ref_count = seq_group.num_seqs();
    //         block_table.push(block);
    //     }

    //     for seq in seq_group.get_seqs() {
    //         self.block_tables.insert(seq.seq_id, block_table.clone());
    //     }
    // }

    // pub fn can_append_slot(&self, seq_group: &SequenceGroup) -> bool {
    //     let num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks();
    //     let num_seqs = seq_group.num_seqs(SequenceStatus::Running);
    //     num_seqs <= num_free_gpu_blocks
    // }

    // pub fn trim_physical_blocks(&mut self, seq: &Sequence) {
    //     if let Some(block_table) = self.block_tables.get_mut(&seq.seq_id) {
    //         let logical_blocks = &seq.logical_token_blocks;
    //         while block_table.len() > logical_blocks.len() {
    //             if let Some(block) = block_table.pop() {
    //                 self.gpu_allocator.free(&mut block);
    //             }
    //         }
    //     }
    // }

    // pub fn append_slot(&mut self, seq: &Sequence) -> Option<(i32, i32)> {
    //     let logical_blocks = &seq.logical_token_blocks;
    //     if let Some(block_table) = self.block_tables.get_mut(&seq.seq_id) {
    //         if block_table.len() < logical_blocks.len() {
    //             let mut block = self.gpu_allocator.allocate();
    //             block_table.push(block);
    //             return None;
    //         }

    //         let last_block = block_table.last_mut().unwrap();
    //         if last_block.ref_count == 1 {
    //             None
    //         } else {
    //             let mut new_block = self.gpu_allocator.allocate();
    //             let old_block_number = last_block.block_number;
    //             *last_block = new_block;
    //             self.gpu_allocator.free(last_block);
    //             Some((old_block_number, new_block.block_number))
    //         }
    //     } else {
    //         None
    //     }
    // }

    // pub fn fork(&mut self, parent_seq: &Sequence, child_seq: &Sequence) {
    //     if let Some(src_block_table) = self.block_tables.get(&parent_seq.seq_id) {
    //         let mut new_block_table = src_block_table.clone();
    //         for block in &mut new_block_table {
    //             block.ref_count += 1;
    //         }
    //         self.block_tables.insert(child_seq.seq_id, new_block_table);
    //     }
    // }

    // pub fn can_swap_in(&self, seq_group: &SequenceGroup) -> bool {
    //     let blocks = self._get_physical_blocks(seq_group);
    //     let num_swapped_seqs = seq_group.num_seqs(SequenceStatus::Swapped);
    //     let num_free_blocks = self.gpu_allocator.get_num_free_blocks();
    //     let num_required_blocks = blocks.len() + num_swapped_seqs;
    //     num_free_blocks - num_required_blocks >= self.watermark_blocks
    // }

    // pub fn swap_in(&mut self, seq_group: &SequenceGroup) -> HashMap<i32, i32> {
    //     let mut mapping = HashMap::new();
    //     for seq in seq_group.get_seqs(SequenceStatus::Swapped) {
    //         let mut new_block_table = BlockTable::new();
    //         if let Some(block_table) = self.block_tables.get(&seq.seq_id) {
    //             for cpu_block in block_table {
    //                 let gpu_block = mapping.entry(cpu_block.block_number).or_insert_with(|| {
    //                     let mut new_gpu_block = self.gpu_allocator.allocate();
    //                     new_gpu_block.ref_count += 1;
    //                     new_gpu_block
    //                 });
    //                 new_block_table.push(gpu_block.clone());
    //                 self.cpu_allocator.free(cpu_block);
    //             }
    //         }
    //         self.block_tables.insert(seq.seq_id, new_block_table);
    //     }

    //     mapping
    //         .into_iter()
    //         .map(|(cpu, gpu)| (cpu, gpu.block_number))
    //         .collect()
    // }

    // pub fn can_swap_out(&self, seq_group: &SequenceGroup) -> bool {
    //     let blocks = self._get_physical_blocks(seq_group);
    //     blocks.len() <= self.cpu_allocator.get_num_free_blocks()
    // }

    // pub fn swap_out(&mut self, seq_group: &SequenceGroup) -> HashMap<i32, i32> {
    //     let mut mapping = HashMap::new();
    //     for seq in seq_group.get_seqs(SequenceStatus::Running) {
    //         let mut new_block_table = BlockTable::new();
    //         if let Some(block_table) = self.block_tables.get(&seq.seq_id) {
    //             for gpu_block in block_table {
    //                 let cpu_block = mapping.entry(gpu_block.block_number).or_insert_with(|| {
    //                     let mut new_cpu_block = self.cpu_allocator.allocate();
    //                     new_cpu_block.ref_count += 1;
    //                     new_cpu_block
    //                 });
    //                 new_block_table.push(cpu_block.clone());
    //                 self.gpu_allocator.free(gpu_block);
    //             }
    //         }
    //         self.block_tables.insert(seq.seq_id, new_block_table);
    //     }

    //     mapping
    //         .into_iter()
    //         .map(|(gpu, cpu)| (gpu, cpu.block_number))
    //         .collect()
    // }

    // fn _free_block_table(&mut self, block_table: &BlockTable) {
    //     for block in block_table {
    //         match block.device {
    //             BlockLocation::GPU => self.gpu_allocator.free(block),
    //             BlockLocation::CPU => self.cpu_allocator.free(block),
    //             _ => {}
    //         }
    //     }
    // }

    // pub fn free(&mut self, seq: &Sequence) {
    //     if let Some(block_table) = self.block_tables.remove(&seq.seq_id) {
    //         self._free_block_table(&block_table);
    //     }
    // }

    // pub fn reset(&mut self) {
    //     for (_, block_table) in self.block_tables.drain() {
    //         self._free_block_table(&block_table);
    //     }
    // }

    // pub fn get_block_table(&self, seq: &Sequence) -> Vec<i32> {
    //     self.block_tables
    //         .get(&seq.seq_id)
    //         .map_or(Vec::new(), |bt| bt.iter().map(|b| b.block_number).collect())
    // }

    pub fn get_num_free_gpu_blocks(&self) -> usize {
        self.gpu_allocator.get_num_free_blocks()
    }

    pub fn get_num_free_cpu_blocks(&self) -> usize {
        self.cpu_allocator.get_num_free_blocks()
    }

    // fn _get_physical_blocks(&self, seq_group: &SequenceGroup) -> HashSet<PhysicalTokenBlock> {
    //     let mut blocks = HashSet::new();
    //     for seq in seq_group.get_seqs() {
    //         if seq.is_finished() {
    //             continue;
    //         }
    //         if let Some(block_table) = self.block_tables.get(&seq.seq_id) {
    //             for block in block_table {
    //                 blocks.insert(block.clone());
    //             }
    //         }
    //     }
    //     blocks
    // }
}
