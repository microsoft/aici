// based on https://github.com/vllm-project/vllm/blob/b9fe4616f98b77b4b9458bce203aa6544cb31ef2/vllm/config.py

use anyhow::{bail, Result};
use candle::{DType, Device};
use log::warn;
use serde::{Deserialize, Serialize};

static GB: usize = 1 << 30;

pub struct ModelConfig {
    pub num_attention_heads: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub max_sequence_length: usize,
    pub dtype: DType,
    pub device: Device,
}

impl ModelConfig {
    pub fn get_hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn get_head_size(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    pub fn get_num_heads(&self, _parallel_config: &ParallelConfig) -> usize {
        self.num_key_value_heads
    }
    pub fn get_num_layers(&self, _parallel_config: &ParallelConfig) -> usize {
        self.num_hidden_layers
    }
    pub fn get_max_model_len(&self) -> usize {
        self.max_sequence_length
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ParallelConfig {}

#[derive(Debug, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Size of a cache block in number of tokens.
    pub block_size: usize,
    /// Fraction of GPU memory to use for the vLLM execution.
    pub gpu_memory_utilization: f64,
    ///  Size of the CPU swap space per GPU (in GiB).
    pub swap_space: usize,

    #[serde(skip)]
    pub swap_space_bytes: usize,
    #[serde(skip)]
    pub num_gpu_blocks: Option<usize>,
    #[serde(skip)]
    pub num_cpu_blocks: Option<usize>,
}

impl CacheConfig {
    pub fn new(block_size: usize, gpu_memory_utilization: f64, swap_space: usize) -> Result<Self> {
        if gpu_memory_utilization > 1.0 {
            bail!(
                "GPU memory utilization must be less than 1.0. Got {}.",
                gpu_memory_utilization
            );
        }
        let total_cpu_memory = get_cpu_memory();
        let swap_space_bytes = swap_space * GB;
        let msg = format!(
            "{:.2} GiB out of the {:.2} GiB total CPU memory is allocated for the swap space.",
            swap_space_bytes as f64 / GB as f64,
            total_cpu_memory as f64 / GB as f64
        );
        if swap_space_bytes > (total_cpu_memory * 7 / 10) {
            bail!("Too large swap space. {}", msg);
        } else if swap_space_bytes > (total_cpu_memory * 4 / 10) {
            warn!("Possibly too large swap space. {}", msg);
        }
        Ok(Self {
            block_size,
            gpu_memory_utilization,
            swap_space,
            swap_space_bytes,
            num_gpu_blocks: None,
            num_cpu_blocks: None,
        })
    }
}

fn get_cpu_memory() -> usize {
    // TODO
    64 * GB
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Maximum number of tokens to be processed in a single iteration.
    pub max_num_batched_tokens: usize,
    /// Maximum number of sequences to be processed in a single iteration.
    pub max_num_seqs: usize,
    /// Maximum length of a sequence (including prompt and generated text).
    pub max_model_len: usize,
}

impl SchedulerConfig {
    pub fn new(max_num_batched_tokens: usize, max_num_seqs: usize, max_model_len: usize) -> Self {
        Self {
            max_num_batched_tokens,
            max_num_seqs,
            max_model_len,
        }
    }
}
