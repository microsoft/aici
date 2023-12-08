// based on https://github.com/vllm-project/vllm/blob/b9fe4616f98b77b4b9458bce203aa6544cb31ef2/vllm/worker/cache_engine.py

use std::collections::HashMap;
use std::sync::Arc;

use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::{Device, Tensor};

use crate::config::RllmConfig;
use crate::kernels;

type KVCache = (Tensor, Tensor);

// TODO
pub struct CudaEvent;

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
    cuda_device: candle::CudaDevice,
}

impl CacheEngine {
    pub fn new(config: Arc<RllmConfig>) -> Self {
        let num_layers = config.get_num_layers_parallel();

        let (gpu_cache, cpu_cache) = Self::allocate_caches(&config);

        let cuda_device = match &config.device {
            Device::Cuda(c) => c.clone(),
            _ => panic!(),
        };

        let cache_stream = cuda_device.fork_default_stream().unwrap();
        let events = (0..num_layers).map(|_| CudaEvent::new()).collect();

        Self {
            gpu_cache,
            cpu_cache,
            cache_stream,
            events,
            cuda_device,
        }
    }

    pub fn wait_for_copy(&self) {
        self.cuda_device.wait_for(&self.cache_stream).unwrap();
    }

    pub fn get_gpu_cache(&self) -> Vec<KVCache> {
        self.gpu_cache.clone()
    }

    pub fn swap_in(&self, src_to_dst: &HashMap<usize, usize>) {
        self.swap(&self.cpu_cache, &self.gpu_cache, src_to_dst);
    }

    pub fn swap_out(&self, src_to_dst: &HashMap<usize, usize>) {
        self.swap(&self.gpu_cache, &self.cpu_cache, src_to_dst);
    }

    fn allocate_caches(config: &RllmConfig) -> (Vec<KVCache>, Vec<KVCache>) {
        let head_size = config.get_head_size();
        let num_layers = config.get_num_layers_parallel();
        let num_heads = config.get_num_heads_parallel();
        let dtype = config.dtype;
        let block_size = config.cache.block_size;
        let x = 16 / dtype.size_in_bytes();

        let key_block = |num_bl, device| unsafe {
            kernels::unset_tensor(
                (num_bl, num_heads, head_size / x, block_size, x),
                dtype,
                device,
            )
        };

        let value_block = |num_bl, device| unsafe {
            kernels::unset_tensor((num_bl, num_heads, head_size, block_size), dtype, device)
        };

        let gpu_cache = {
            let num_gpu_blocks = config.cache.num_gpu_blocks.unwrap();
            (0..num_layers)
                .map(|_| {
                    let device = &config.device;
                    (
                        key_block(num_gpu_blocks, device),
                        value_block(num_gpu_blocks, device),
                    )
                })
                .collect()
        };

        let cpu_cache = {
            let num_cpu_blocks = config.cache.num_cpu_blocks.unwrap();
            (0..num_layers)
                .map(|_| {
                    // TODO: vllm sets pin_memory=True here
                    let device = &Device::Cpu;
                    (
                        key_block(num_cpu_blocks, device),
                        value_block(num_cpu_blocks, device),
                    )
                })
                .collect()
        };

        (gpu_cache, cpu_cache)
    }

    fn swap(&self, src: &[KVCache], dst: &[KVCache], src_to_dst: &HashMap<usize, usize>) {
        let stream = &self.cache_stream;
        for (i, (src_key_cache, src_value_cache)) in src.iter().enumerate() {
            let (dst_key_cache, dst_value_cache) = &dst[i];
            kernels::swap_blocks(src_key_cache, dst_key_cache, src_to_dst, stream);
            kernels::swap_blocks(src_value_cache, dst_value_cache, src_to_dst, stream);
            let event = &self.events[i];
            event.record(stream);
        }
    }

    pub fn copy(&self, src_to_dsts: &HashMap<usize, Vec<usize>>) {
        let mut key_caches: Vec<_> = self.gpu_cache.iter().map(|(key, _)| key).collect();
        let mut value_caches: Vec<_> = self.gpu_cache.iter().map(|(_, value)| value).collect();
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
        config.dtype.size_in_bytes() * total
    }
}
