// based on https://github.com/vllm-project/vllm/blob/b9fe4616f98b77b4b9458bce203aa6544cb31ef2/vllm/worker/cache_engine.py

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::{DType, Device, Tensor};

use crate::config::{CacheConfig, ModelConfig, ParallelConfig};
use crate::kernels;
// use crate::utils::{in_wsl, get_dtype_element_size};

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
    cache_config: Arc<CacheConfig>,
    model_config: Arc<ModelConfig>,
    parallel_config: Arc<ParallelConfig>,

    head_size: usize,
    num_layers: usize,
    num_heads: usize,
    dtype: DType,

    block_size: usize,
    num_gpu_blocks: usize,
    num_cpu_blocks: usize,

    gpu_cache: Vec<KVCache>,
    cpu_cache: Vec<KVCache>,

    cache_stream: CudaStream,
    events: Vec<CudaEvent>,
    cuda_device: candle::CudaDevice,
}

impl CacheEngine {
    pub fn new(
        cache_config: Arc<CacheConfig>,
        model_config: Arc<ModelConfig>,
        parallel_config: Arc<ParallelConfig>,
    ) -> Self {
        let head_size = model_config.get_head_size();
        let num_layers = model_config.get_num_layers(&parallel_config);
        let num_heads = model_config.get_num_heads(&parallel_config);
        let dtype = model_config.dtype;

        let block_size = cache_config.block_size;
        let num_gpu_blocks = cache_config.num_gpu_blocks.unwrap();
        let num_cpu_blocks = cache_config.num_cpu_blocks.unwrap();

        let (gpu_cache, cpu_cache) =
            Self::allocate_caches(&model_config, &cache_config, &parallel_config);

        let cuda_device = match &model_config.device {
            Device::Cuda(c) => c.clone(),
            _ => panic!(),
        };

        let cache_stream = cuda_device.fork_default_stream().unwrap();
        let events = (0..num_layers).map(|_| CudaEvent::new()).collect();

        Self {
            cache_config,
            model_config,
            parallel_config,
            head_size,
            num_layers,
            num_heads,
            dtype,
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            gpu_cache,
            cpu_cache,
            cache_stream,
            events,
            cuda_device,
        }
    }

    fn get_key_block_shape(&self) -> (usize, usize, usize, usize) {
        let x = 16 / self.dtype.size_in_bytes();
        (self.num_heads, self.head_size / x, self.block_size, x)
    }

    fn get_value_block_shape(&self) -> (usize, usize, usize) {
        (self.num_heads, self.head_size, self.block_size)
    }

    pub fn swap_in(&self, src_to_dst: HashMap<usize, usize>) {
        self.swap(&self.cpu_cache, &self.gpu_cache, &src_to_dst);
    }

    pub fn swap_out(&self, src_to_dst: HashMap<usize, usize>) {
        self.swap(&self.gpu_cache, &self.cpu_cache, &src_to_dst);
    }

    fn allocate_caches(
        model_config: &ModelConfig,
        cache_config: &CacheConfig,
        parallel_config: &ParallelConfig,
    ) -> (Vec<KVCache>, Vec<KVCache>) {
        let head_size = model_config.get_head_size();
        let num_layers = model_config.get_num_layers(parallel_config);
        let num_heads = model_config.get_num_heads(parallel_config);
        let dtype = model_config.dtype;

        let gpu_cache = {
            let num_gpu_blocks = cache_config.num_gpu_blocks.unwrap();
            let key_block_shape = (
                num_heads,
                head_size / (16 / dtype.size_in_bytes()),
                num_gpu_blocks,
                16 / dtype.size_in_bytes(),
            );
            let value_block_shape = (num_heads, head_size, num_gpu_blocks);
            (0..num_layers)
                .map(|_| {
                    let key_blocks =
                        Tensor::zeros(key_block_shape, dtype, &model_config.device).unwrap();
                    let value_blocks =
                        Tensor::zeros(value_block_shape, dtype, &model_config.device).unwrap();
                    (key_blocks, value_blocks)
                })
                .collect()
        };

        let cpu_cache = {
            let num_cpu_blocks = cache_config.num_cpu_blocks.unwrap();
            let key_block_shape = (
                num_heads,
                head_size / (16 / dtype.size_in_bytes()),
                num_cpu_blocks,
                16 / dtype.size_in_bytes(),
            );
            let value_block_shape = (num_heads, head_size, num_cpu_blocks);
            (0..num_layers)
                .map(|_| {
                    let key_blocks = Tensor::zeros(key_block_shape, dtype, &Device::Cpu).unwrap();
                    let value_blocks =
                        Tensor::zeros(value_block_shape, dtype, &Device::Cpu).unwrap();
                    (key_blocks, value_blocks)
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

    pub fn copy(&self, src_to_dsts: HashMap<usize, Vec<usize>>) {
        let mut key_caches: Vec<_> = self.gpu_cache.iter().map(|(key, _)| key).collect();
        let mut value_caches: Vec<_> = self.gpu_cache.iter().map(|(_, value)| value).collect();
        kernels::copy_blocks(&mut key_caches, &mut value_caches, &src_to_dsts);
    }

    pub fn get_cache_block_size(
        block_size: usize,
        model_config: &ModelConfig,
        parallel_config: &ParallelConfig,
    ) -> usize {
        let head_size = model_config.get_head_size();
        let num_heads = model_config.get_num_heads(parallel_config);
        let num_layers = model_config.get_num_layers(parallel_config);

        let key_cache_block = block_size * num_heads * head_size;
        let value_cache_block = key_cache_block;
        let total = num_layers * (key_cache_block + value_cache_block);
        model_config.dtype.size_in_bytes() * total
    }
}

fn in_wsl() -> bool {
    false
}
