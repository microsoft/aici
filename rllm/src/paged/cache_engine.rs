// based on https://github.com/vllm-project/vllm/blob/b9fe4616f98b77b4b9458bce203aa6544cb31ef2/vllm/worker/cache_engine.py

use std::{collections::HashMap, sync::Arc};

use crate::{Device, Tensor};

use crate::{config::RllmConfig, llm::kernels};

type KVCache = (Tensor, Tensor);

// TODO
pub struct CudaEvent;
pub struct CudaStream;

impl CudaEvent {
    pub fn new() -> Self {
        Self
    }

    pub fn record(&self, _stream: &CudaStream) {}
}

pub struct CacheEngine {
    gpu_cache: Vec<KVCache>,
    cpu_cache: Vec<KVCache>,

    cache_stream: CudaStream,
    events: Vec<CudaEvent>,
    // cuda_device: candle_core::CudaDevice,
}

pub struct CacheSize {
    pub gpu: usize,
    pub cpu: usize,
}

impl CacheEngine {
    pub fn new(config: Arc<RllmConfig>, num_blocks: &CacheSize) -> Self {
        let num_layers = config.get_num_layers_parallel();

        let (gpu_cache, cpu_cache) = Self::allocate_caches(&config, num_blocks);

        // let cuda_device = match &config.device {
        //     Device::Cuda(c) => c.clone(),
        //     _ => panic!(),
        // };

        // let cache_stream = cuda_device.fork_default_stream().unwrap();
        let events = (0..num_layers).map(|_| CudaEvent::new()).collect();

        Self {
            gpu_cache,
            cpu_cache,
            cache_stream: CudaStream,
            events,
            // cuda_device,
        }
    }

    pub fn wait_for_copy(&self) {
        // self.cuda_device.wait_for(&self.cache_stream).unwrap();
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
        for (i, (src_key_cache, src_value_cache)) in src.iter().enumerate() {
            let (dst_key_cache, dst_value_cache) = &dst[i];
            kernels::swap_blocks(src_key_cache, dst_key_cache, src_to_dst);
            kernels::swap_blocks(src_value_cache, dst_value_cache, src_to_dst);
            let event = &self.events[i];
            event.record(stream);
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
