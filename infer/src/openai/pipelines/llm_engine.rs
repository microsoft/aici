use std::{collections::HashMap, sync::Arc};

use candle_core::{Device, Tensor};

use crate::{
    openai::responses::APIError,
    paged_attention::{
        cache_engine::{CacheConfig, CacheEngine, ModelConfig, ParallelConfig},
        sequence::{SequenceGroupMetadata, SequenceGroupOutput},
    },
};

use super::ModulePipeline;

pub struct LlmEngine<'a> {
    pipeline: Box<dyn ModulePipeline<'a>>,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    parallel_config: ParallelConfig,
    cache_engine: Option<CacheEngine>,
}

pub struct ProfiledBlocks {
    num_gpu_blocks: usize,
    num_cpu_blocks: usize,
}

impl<'a> LlmEngine<'a> {
    pub fn new(
        pipeline: Box<dyn ModulePipeline<'a>>,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> Self {
        let mut this = Self {
            pipeline,
            model_config,
            cache_config,
            parallel_config,
            cache_engine: None,
        };
        this._init_cache();
        this
    }

    // The following 2 are functions with todo!() internals
    // TODO_URGENT(EricLBuehler)
    fn get_peak_memory() -> usize {
        // Pending on https://github.com/huggingface/candle/pull/1412
        todo!()
    }
    fn total_gpu_memory() -> usize {
        todo!()
    }

    fn profile_num_available_blocks(
        &mut self,
        block_size: usize,
        gpu_memory_utilization: f64,
        cpu_swap_space: usize,
    ) -> ProfiledBlocks {
        // Pending on https://github.com/huggingface/candle/pull/1412
        // reset_peak_memory_stats

        self.pipeline.profile_run();

        let peak_memory: usize = Self::get_peak_memory();

        let total_gpu_memory = Self::total_gpu_memory();
        let cache_block_size = CacheEngine::get_cache_block_size(
            block_size,
            &self.model_config,
            &self.parallel_config,
        );
        let num_gpu_blocks = (total_gpu_memory as f64 * gpu_memory_utilization - peak_memory as f64)
            as usize
            / cache_block_size;
        let num_cpu_blocks = cpu_swap_space / cache_block_size;
        let num_gpu_blocks = num_gpu_blocks.max(0);
        let num_cpu_blocks = num_cpu_blocks.max(0);

        ProfiledBlocks {
            num_gpu_blocks,
            num_cpu_blocks,
        }
    }

    pub fn _init_cache(&mut self) -> Result<(), APIError> {
        let ProfiledBlocks {
            num_gpu_blocks,
            num_cpu_blocks,
        } = self.profile_num_available_blocks(
            self.cache_config.block_size,
            self.cache_config.gpu_mem_utilization,
            self.cache_config.swap_space_bytes,
        );
        eprintln!("{num_gpu_blocks} GPU blocks.");
        eprintln!("{num_cpu_blocks} CPU blocks.");

        if num_gpu_blocks <= 0 {
            return Err(APIError::new_str("No available memory for the cache blocks. Try increasing `gpu_mem_utilization` when initializing the engine."));
        }

        self.cache_config.num_cpu_blocks = Some(num_cpu_blocks);
        self.cache_config.num_gpu_blocks = Some(num_gpu_blocks);

        todo!("init_cache_engine");

        Ok(())
    }

    // Calls the module pipeline model executer, called by .step
    fn execute_model(
        &mut self,
        seq_group_metadatas: Vec<SequenceGroupMetadata>,
        blocks_to_swap_in: HashMap<usize, usize>,
        blocks_to_swap_out: HashMap<usize, usize>,
        blocks_to_copy: HashMap<usize, Vec<usize>>,
    ) -> Vec<SequenceGroupOutput> {
        if !blocks_to_swap_in.is_empty() {
            self.cache_engine.unwrap().swap_in(blocks_to_swap_in);
        }
        if !blocks_to_swap_out.is_empty() {
            self.cache_engine.unwrap().swap_out(blocks_to_swap_out);
        }
        if !blocks_to_copy.is_empty() {
            self.cache_engine.unwrap().copy(blocks_to_copy);
        }

        //https://github.com/vllm-project/vllm/blob/05ff90b692a6cdac4d8c06e7a4a4606d1b8fe1d6/vllm/worker/worker.py#L119

        todo!()
    }
}
