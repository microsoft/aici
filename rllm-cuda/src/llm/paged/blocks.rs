use super::super::tmodel::TModel;
use super::cache_engine::CacheEngine;
use crate::{
    config::RllmConfig,
    seq::{SchedulingPhase, Sequence, SequenceGroup},
    BlockLocation, CacheSize, HashMap, SchedulerOutputs, SeqId, SequenceManager,
    TBlockSpaceManager,
};
use std::{
    sync::{Arc, Mutex},
    vec::Vec,
};

/// Represents the state of a block in the KV cache.
#[derive(Debug)]
pub struct PhysicalTokenBlock {
    ref_count: usize,
}

impl PhysicalTokenBlock {
    pub fn new(_device: BlockLocation, _block_number: usize, _block_size: usize) -> Self {
        Self { ref_count: 0 }
    }
}

/// Manages free physical token blocks for a device.
///
/// The allocator maintains a list of free blocks and allocates a block when
/// requested. When a block is freed, its reference count is decremented. If
/// the reference count becomes zero, the block is added back to the free list.

struct Allocator {
    free_list: Vec<usize>,
    all_blocks: Vec<PhysicalTokenBlock>,
    block_size: usize,
}

struct BlockAllocatorInner {
    alloc: Allocator,
    seq_blocks: HashMap<SeqId, Vec<BlockRef>>,
}

#[derive(Clone)]
pub struct BlockAllocator {
    inner: Arc<Mutex<BlockAllocatorInner>>,
}

struct BlockRef {
    block_idx: usize,
}

impl Allocator {
    fn num_blocks(&self, length: usize) -> usize {
        (length + self.block_size - 1) / self.block_size
    }

    fn free(&mut self, block: BlockRef) {
        let blk = &mut self.all_blocks[block.block_idx];
        assert!(blk.ref_count > 0);
        blk.ref_count -= 1;
        if blk.ref_count == 0 {
            self.free_list.push(block.block_idx);
        }
    }

    fn fork(&mut self, block: &BlockRef) -> BlockRef {
        let blk = &mut self.all_blocks[block.block_idx];
        assert!(blk.ref_count > 0);
        blk.ref_count += 1;
        BlockRef {
            block_idx: block.block_idx,
        }
    }

    fn allocate(&mut self) -> BlockRef {
        let block_idx = self
            .free_list
            .pop()
            .expect("Out of memory! No free blocks are available.");
        assert!(self.all_blocks[block_idx].ref_count == 0);
        self.all_blocks[block_idx].ref_count += 1;
        BlockRef { block_idx }
    }

    fn is_singular(&self, block: &BlockRef) -> bool {
        let blk = &self.all_blocks[block.block_idx];
        assert!(blk.ref_count > 0);
        blk.ref_count == 1
    }
}

impl BlockAllocatorInner {
    fn copy(&mut self, src: SeqId, dst: SeqId, length: usize) {
        let alloc = &mut self.alloc;
        let seq_blocks = &mut self.seq_blocks;
        match seq_blocks.get(&src) {
            Some(v) => {
                let length = alloc.num_blocks(length);
                let mut new_v = Vec::with_capacity(std::cmp::min(length, v.len()));
                for e in v.iter().take(length) {
                    new_v.push(alloc.fork(e));
                }
                seq_blocks.insert(dst, new_v);
            }
            None => {}
        }
    }

    fn trim(&mut self, seq: SeqId, length: usize) {
        let alloc = &mut self.alloc;
        let length = alloc.num_blocks(length);
        self.seq_blocks.get_mut(&seq).map(|v| {
            for e in v.drain(length..) {
                alloc.free(e)
            }
        });
        if length == 0 {
            self.seq_blocks.remove(&seq);
        }
    }

    fn get_block_idx(&self, seq: SeqId, position: usize) -> usize {
        let blocks = self.seq_blocks.get(&seq).unwrap();
        let block_size = self.alloc.block_size;
        let block_offset = position % block_size;
        blocks[position / block_size].block_idx * block_size + block_offset
    }
}

impl BlockAllocator {
    fn new(device: BlockLocation, block_size: usize, num_blocks: usize) -> Self {
        let all_blocks = (0..num_blocks)
            .map(|i| PhysicalTokenBlock::new(device, i, block_size))
            .collect();
        let inner = BlockAllocatorInner {
            alloc: Allocator {
                all_blocks,
                free_list: (0..num_blocks).rev().collect(),
                block_size,
            },
            seq_blocks: HashMap::default(),
        };
        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    fn get_num_free_blocks(&self) -> usize {
        self.inner.lock().unwrap().alloc.free_list.len()
    }

    pub fn get_block_idxes(&self, seq: SeqId, len: usize) -> Vec<usize> {
        let l = self.inner.lock().unwrap();
        (0..len).map(|k| l.get_block_idx(seq, k)).collect()
    }

    fn num_needed_blocks(&self, seq: &Sequence) -> usize {
        let l = self.inner.lock().unwrap();
        l.alloc.num_blocks(seq.get_len())
    }

    fn num_allocated_blocks(&self, seq: &Sequence) -> usize {
        let l = self.inner.lock().unwrap();
        l.seq_blocks.get(&seq.seq_id).map(|v| v.len()).unwrap_or(0)
    }

    fn alloc_seq(&self, seq: &Sequence) {
        assert!(self.num_allocated_blocks(seq) == 0);
        let mut l = self.inner.lock().unwrap();
        let num_bl = l.alloc.num_blocks(seq.get_len());
        let mut v = Vec::with_capacity(num_bl);
        for _ in 0..num_bl {
            v.push(l.alloc.allocate())
        }
        l.seq_blocks.insert(seq.seq_id, v);
    }

    fn swap_out(&self, seq: &Sequence) -> Vec<usize> {
        let r = {
            let l = self.inner.lock().unwrap();
            l.seq_blocks
                .get(&seq.seq_id)
                .unwrap_or(&Vec::new())
                .iter()
                .map(|b| b.block_idx)
                .collect()
        };
        self.trim(seq.seq_id, 0);
        r
    }

    fn swap_in(&self, seq: &Sequence, block_idxs: Vec<usize>, mapping: &mut HashMap<usize, usize>) {
        assert!(self.num_allocated_blocks(seq) == 0);
        let mut l = self.inner.lock().unwrap();
        let mut v = Vec::with_capacity(block_idxs.len());
        for bidx in block_idxs {
            match mapping.get(&bidx) {
                Some(&new_bidx) => {
                    v.push(l.alloc.fork(&BlockRef {
                        block_idx: new_bidx,
                    }));
                }
                None => {
                    let b2 = l.alloc.allocate();
                    mapping.insert(bidx, b2.block_idx);
                    v.push(b2);
                }
            }
        }
        l.seq_blocks.insert(seq.seq_id, v);
    }

    fn append_slots(&self, seq: &Sequence, outputs: &mut SchedulerOutputs) {
        let mut l = self.inner.lock().unwrap();
        let block_size = l.alloc.block_size;
        let mut block_table = l.seq_blocks.remove(&seq.seq_id).unwrap();

        assert!(block_table.len() > 0);
        assert!(block_table.len() * block_size >= seq.num_kv_computed);

        let mut ptr = seq.num_kv_computed;
        while ptr < seq.get_len() {
            let block_idx = ptr / block_size;
            if block_idx < block_table.len() {
                let curr_block = &mut block_table[block_idx];
                if !l.alloc.is_singular(curr_block) {
                    let new_block = l.alloc.allocate();
                    let old_block_number = curr_block.block_idx;
                    let new_block_number = new_block.block_idx;
                    let old_block = std::mem::replace(curr_block, new_block);
                    l.alloc.free(old_block);
                    outputs.copy_block(old_block_number, new_block_number);
                }
            } else {
                assert!(block_table.len() == block_idx);
                block_table.push(l.alloc.allocate());
            }
            ptr = (block_idx + 1) * block_size;
        }

        assert!(block_table.len() == l.alloc.num_blocks(seq.get_len()));
        l.seq_blocks.insert(seq.seq_id, block_table);
    }

    fn copy(&self, src: SeqId, dst: SeqId, length: usize) {
        self.trim(dst, 0);
        self.inner.lock().unwrap().copy(src, dst, length)
    }

    fn trim(&self, seq: SeqId, length: usize) {
        self.inner.lock().unwrap().trim(seq, length);
    }

    fn delete(&self, seq: SeqId) {
        self.trim(seq, 0);
    }
}

/// Manages the mapping between logical and physical token blocks.
pub struct BlockSpaceManager {
    watermark_blocks: usize,
    gpu_allocator: BlockAllocator,
    cpu_allocator: BlockAllocator,
}

impl TBlockSpaceManager<TModel> for BlockSpaceManager {
    fn can_allocate(&self, seq_group: &SequenceGroup) -> bool {
        let num_required_blocks = self.gpu_allocator.num_needed_blocks(seq_group.only_seq());
        self.can_alloc_gpu(num_required_blocks + self.watermark_blocks)
    }

    fn allocate(&mut self, seq_group: &mut SequenceGroup) {
        let seq = seq_group.only_seq();
        assert!(seq.num_kv_computed == 0);
        self.gpu_allocator.alloc_seq(seq);
    }

    fn can_append_slot(&self, seq_group: &SequenceGroup) -> bool {
        let num_seqs = seq_group.num_seqs(Some(SchedulingPhase::Running));
        // TODO this is not correct - more than one token can be appended
        self.can_alloc_gpu(num_seqs)
    }

    fn append_slots(&mut self, seq: &mut Sequence, outputs: &mut SchedulerOutputs) {
        self.gpu_allocator.append_slots(seq, outputs)
    }

    fn can_swap_in(&self, seq_group: &SequenceGroup) -> bool {
        let blocks = self.num_phys_blocks(seq_group);
        let num_swapped_seqs = seq_group.num_seqs(Some(SchedulingPhase::Swapped));
        let num_required_blocks = blocks + num_swapped_seqs;
        self.can_alloc_gpu(num_required_blocks + self.watermark_blocks)
    }

    fn swap_in(&mut self, seq_group: &mut SequenceGroup) -> HashMap<usize, usize> {
        let mut mapping = HashMap::default();
        for seq in &mut seq_group.seqs {
            if seq.sched_phase == SchedulingPhase::Swapped {
                self.cpu_allocator
                    .swap_in(seq, self.gpu_allocator.swap_out(seq), &mut mapping);
                seq.sched_phase = SchedulingPhase::Running;
            }
        }
        mapping
    }

    fn swap_out(&mut self, seq_group: &mut SequenceGroup) -> HashMap<usize, usize> {
        let mut mapping = HashMap::default();
        for seq in &mut seq_group.seqs {
            if seq.sched_phase == SchedulingPhase::Running {
                self.gpu_allocator
                    .swap_in(seq, self.cpu_allocator.swap_out(seq), &mut mapping);
                seq.sched_phase = SchedulingPhase::Swapped;
            }
        }
        mapping
    }

    fn can_swap_out(&self, seq_group: &SequenceGroup) -> bool {
        let blocks = self.num_phys_blocks(seq_group);
        blocks <= self.get_num_free_cpu_blocks()
    }

    fn get_num_free_gpu_blocks(&self) -> usize {
        self.gpu_allocator.get_num_free_blocks()
    }

    fn get_num_free_cpu_blocks(&self) -> usize {
        self.cpu_allocator.get_num_free_blocks()
    }
}

impl BlockSpaceManager {
    pub(crate) fn new(
        block_size: usize,
        cache_size: &CacheSize,
        watermark: f32,
        config: &RllmConfig<TModel>,
    ) -> Self {
        assert!(watermark >= 0.0);
        let watermark_blocks = (watermark * cache_size.gpu as f32) as usize;

        log::info!("BlockSpaceManager: block_size: {} tokens", block_size);

        Self {
            watermark_blocks,
            gpu_allocator: Self::new_allocator(
                BlockLocation::GPU,
                block_size,
                cache_size.gpu,
                config,
            ),
            cpu_allocator: Self::new_allocator(
                BlockLocation::CPU,
                block_size,
                cache_size.cpu,
                config,
            ),
        }
    }

    pub(crate) fn build_seq_mgr(&self) -> TchSeqMgr {
        TchSeqMgr::new(self.cpu_allocator.clone(), self.gpu_allocator.clone())
    }

    fn new_allocator(
        location: BlockLocation,
        block_size: usize,
        num_blocks: usize,
        config: &RllmConfig<TModel>,
    ) -> BlockAllocator {
        log::info!(
            "{:?} {} blocks, {} MiB",
            location,
            num_blocks,
            (num_blocks * CacheEngine::get_cache_block_size(config)) >> 20
        );
        BlockAllocator::new(location, block_size, num_blocks)
    }

    fn can_alloc_gpu(&self, num_required_blocks: usize) -> bool {
        self.get_num_free_gpu_blocks() >= num_required_blocks
    }

    fn num_phys_blocks(&self, seq_group: &SequenceGroup) -> usize {
        seq_group
            .get_seqs(None)
            .iter()
            .map(|seq| {
                let n = self.gpu_allocator.num_allocated_blocks(seq);
                if n == 0 {
                    self.cpu_allocator.num_allocated_blocks(seq)
                } else {
                    assert!(self.cpu_allocator.num_allocated_blocks(seq) == 0);
                    n
                }
            })
            .sum()
    }
}

pub struct TchSeqMgr {
    next: Mutex<usize>,
    cpu_allocator: BlockAllocator,
    gpu_allocator: BlockAllocator,
}

impl TchSeqMgr {
    pub(super) fn new(cpu_allocator: BlockAllocator, gpu_allocator: BlockAllocator) -> Self {
        Self {
            next: Mutex::new(1),
            cpu_allocator,
            gpu_allocator,
        }
    }
    pub(crate) fn get_gpu_allocator(&self) -> &BlockAllocator {
        &self.gpu_allocator
    }
}

impl SequenceManager for TchSeqMgr {
    fn new_sequence(&self) -> SeqId {
        let mut l = self.next.lock().unwrap();
        let r = SeqId(*l);
        *l = *l + 1;
        r
    }

    fn copy(&self, src: SeqId, dst: SeqId, length: usize) {
        self.cpu_allocator.copy(src, dst, length);
        self.gpu_allocator.copy(src, dst, length);
    }

    fn trim(&self, seq: SeqId, length: usize) {
        self.cpu_allocator.trim(seq, length);
        self.gpu_allocator.trim(seq, length);
    }

    fn delete(&self, seq: SeqId) {
        self.cpu_allocator.delete(seq);
        self.gpu_allocator.delete(seq);
    }
}
