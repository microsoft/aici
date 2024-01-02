// based on https://github.com/vllm-project/vllm/blob/b9fe4616f98b77b4b9458bce203aa6544cb31ef2/vllm/worker/cache_engine.py

use tch_cuda::{CudaEvent, CudaStream};

use crate::{config::RllmConfig, llm::kernels, Device, Tensor};
use std::{collections::HashMap, sync::Arc};

use super::CacheAwaiter;

type KVCache = (Tensor, Tensor);

pub struct CacheEngine {
    gpu_cache: Vec<KVCache>,
    cpu_cache: Vec<KVCache>,

    cache_stream: CudaStream,
    events: Arc<Vec<CudaEvent>>,
}

pub struct CacheSize {
    pub gpu: usize,
    pub cpu: usize,
}

struct MyCacheAwaiter {
    events: Arc<Vec<CudaEvent>>,
    stream: CudaStream,
}

impl CacheAwaiter for MyCacheAwaiter {
    fn wait(&self, layer: usize) {
        self.events[layer].wait(&self.stream)
    }
}

impl CacheEngine {
    pub fn new(config: Arc<RllmConfig>, num_blocks: &CacheSize) -> Self {
        let num_layers = config.get_num_layers_parallel();
        let (gpu_cache, cpu_cache) = Self::allocate_caches(&config, num_blocks);
        Self {
            gpu_cache,
            cpu_cache,
            cache_stream: CudaStream::new(config.device),
            events: Arc::new((0..num_layers).map(|_| CudaEvent::new()).collect()),
        }
    }

    pub fn get_cache_awaiter(&self) -> Box<dyn CacheAwaiter> {
        let d = self.gpu_cache[0].0.device();
        Box::new(MyCacheAwaiter {
            events: self.events.clone(),
            stream: CudaStream::current(d),
        })
    }

    pub fn get_gpu_cache(&self) -> Vec<KVCache> {
        self.gpu_cache
            .iter()
            .map(|(k, v)| (k.shallow_clone(), v.shallow_clone()))
            .collect()
    }

    pub fn swap_in(&self, src_to_dst: &HashMap<usize, usize>) {
        self.swap(&self.cpu_cache, &self.gpu_cache, src_to_dst);
    }

    pub fn swap_out(&self, src_to_dst: &HashMap<usize, usize>) {
        self.swap(&self.gpu_cache, &self.cpu_cache, src_to_dst);
    }

    fn alloc_key_block(config: &RllmConfig, num_bl: i64, device: Device) -> Tensor {
        let head_size = config.get_head_size() as i64;
        let num_heads = config.get_num_heads_parallel() as i64;
        let block_size = config.cache.block_size as i64;
        let x = 16 / (config.dtype.elt_size_in_bytes() as i64);
        Tensor::empty(
            &[num_bl, num_heads, head_size / x, block_size, x],
            (config.dtype, device),
        )
    }

    fn alloc_value_block(config: &RllmConfig, num_bl: i64, device: Device) -> Tensor {
        let head_size = config.get_head_size() as i64;
        let num_heads = config.get_num_heads_parallel() as i64;
        let block_size = config.cache.block_size as i64;
        Tensor::empty(
            &[num_bl, num_heads, head_size, block_size],
            (config.dtype, device),
        )
    }

    pub fn alloc_gpu_cache_layer(config: &RllmConfig, num_bl: i64) -> (Tensor, Tensor) {
        let device = config.device;
        (
            Self::alloc_key_block(config, num_bl, device),
            Self::alloc_value_block(config, num_bl, device),
        )
    }

    fn allocate_caches(
        config: &RllmConfig,
        num_blocks: &CacheSize,
    ) -> (Vec<KVCache>, Vec<KVCache>) {
        let num_layers = config.get_num_layers_parallel() as i64;

        let gpu_cache = (0..num_layers)
            .map(|_| Self::alloc_gpu_cache_layer(config, num_blocks.gpu as i64))
            .collect();

        let cpu_cache = (0..num_layers)
            .map(|_| {
                // TODO: vllm sets pin_memory=True here
                let device = Device::Cpu;
                (
                    Self::alloc_key_block(config, num_blocks.cpu as i64, device),
                    Self::alloc_value_block(config, num_blocks.cpu as i64, device),
                )
            })
            .collect();

        (gpu_cache, cpu_cache)
    }

    fn swap(&self, src: &[KVCache], dst: &[KVCache], src_to_dst: &HashMap<usize, usize>) {
        let stream = &self.cache_stream;
        for (i, (src_k_cache, src_v_cache)) in src.iter().enumerate() {
            let (dst_k_cache, dst_v_cache) = &dst[i];
            kernels::swap_blocks(src_k_cache, dst_k_cache, src_to_dst, &self.cache_stream);
            kernels::swap_blocks(src_v_cache, dst_v_cache, src_to_dst, &self.cache_stream);
            self.events[i].record(stream);
        }
    }

    pub fn copy(&self, src_to_dsts: &HashMap<usize, Vec<usize>>) {
        let mut key_caches: Vec<_> = self
            .gpu_cache
            .iter()
            .map(|(key, _)| key.shallow_clone())
            .collect();
        let mut value_caches: Vec<_> = self
            .gpu_cache
            .iter()
            .map(|(_, value)| value.shallow_clone())
            .collect();
        kernels::copy_blocks(&mut key_caches, &mut value_caches, &src_to_dsts);
    }

    pub fn get_cache_block_size(config: &RllmConfig) -> usize {
        let block_size = config.cache.block_size;
        let head_size = config.get_head_size();
        let num_heads = config.get_num_heads_parallel();
        let num_layers = config.get_num_layers_parallel();

        let key_cache_block = block_size * num_heads * head_size;
        let value_cache_block = key_cache_block;
        let total = num_layers * (key_cache_block + value_cache_block);
        config.dtype.elt_size_in_bytes() * total
    }
}
