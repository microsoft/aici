use crate::config::{ModelMeta, RllmConfig};
use aicirt::bail_user;
use anyhow::Result;
use tch::Device;

use super::{tmodel::TModel, DType};

const GB: usize = 1 << 30;

pub trait TchRllmConfig {
    fn get_hidden_size(&self) -> usize;
    fn get_head_size(&self) -> usize;
    fn get_num_heads_parallel(&self) -> usize;
    fn get_num_layers_parallel(&self) -> usize;
    fn get_max_model_len(&self) -> usize;
    fn verify_args(&self) -> Result<()>;
}

impl TchRllmConfig for RllmConfig<TModel> {
    fn verify_args(&self) -> Result<()> {
        let model = &self.model;
        let parallel = &self.parallel;
        if model.num_hidden_layers % parallel.pipeline_parallel_size != 0 {
            bail_user!(
                "Number of hidden layers ({}) must be divisible by the pipeline parallel size ({}).",
                model.num_hidden_layers,
                parallel.pipeline_parallel_size
            );
        }
        if model.num_key_value_heads % parallel.tensor_parallel_size != 0 {
            bail_user!(
                "Number of key/value heads ({}) must be divisible by the tensor parallel size ({}).",
                model.num_key_value_heads,
                parallel.tensor_parallel_size
            );
        }
        if self.aici.max_fuel < 100 {
            bail_user!("max_fuel not configured");
        }
        Ok(())
    }

    fn get_hidden_size(&self) -> usize {
        self.model.hidden_size
    }
    fn get_head_size(&self) -> usize {
        self.model.hidden_size / self.model.num_attention_heads
    }
    fn get_num_heads_parallel(&self) -> usize {
        self.model.num_key_value_heads / self.parallel.tensor_parallel_size
    }
    fn get_num_layers_parallel(&self) -> usize {
        self.model.num_hidden_layers / self.parallel.pipeline_parallel_size
    }
    fn get_max_model_len(&self) -> usize {
        self.meta.max_sequence_length
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Llama,
    Phi,
    LlamaCpp,
}

pub struct CommonModelConfig {
    pub meta: ModelMeta,
    pub device: Device,
    pub dtype: Option<DType>,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: ModelType,
    pub meta: ModelMeta,

    pub num_attention_heads: usize,
    pub hidden_size: usize, // head_dim * num_attention_heads
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,

    pub intermediate_size: usize,

    pub layer_norm_eps: f64, // default 1e-5
    pub rope_theta: f32,     // default 10000

    pub device: Device,
    pub dtype: DType,

    pub profile_step_no: usize,
    pub cache: CacheConfig,
}

impl ModelConfig {
    pub fn dtype_from_str(explicit: Option<DType>, torch_dtype: &str) -> DType {
        if let Some(dtype) = explicit {
            return dtype;
        }
        match torch_dtype {
            "float" => DType::Float,
            "half" | "float16" => DType::Half,
            "bfloat16" => DType::BFloat16,
            _ => panic!("Unknown dtype {}", torch_dtype),
        }
    }
}
pub trait RllmModelConfig {
    fn into_config(self, common: CommonModelConfig) -> ModelConfig;
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Size of a cache block in number of tokens.
    pub block_size: usize,
    /// Fraction of GPU memory to use for the vLLM execution.
    pub gpu_memory_utilization: f64,
    ///  Size of the CPU swap space per GPU (in GiB).
    pub swap_space: usize,

    /// 0 - don't use paged_attention_v1/2(), otherwise version
    pub paged_attn_kernel_v: usize,

    // #[serde(skip)]
    pub swap_space_bytes: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self::new(16, 0.9, 4).unwrap()
    }
}

impl CacheConfig {
    pub fn new(block_size: usize, gpu_memory_utilization: f64, swap_space: usize) -> Result<Self> {
        if gpu_memory_utilization > 1.0 {
            bail_user!(
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
            bail_user!("Too large swap space. {}", msg);
        } else if swap_space_bytes > (total_cpu_memory * 4 / 10) {
            log::warn!("Possibly too large swap space. {}", msg);
        }
        #[cfg(feature = "cuda")]
        let paged_attn_kernel_v = 1;
        #[cfg(not(feature = "cuda"))]
        let paged_attn_kernel_v = 0;
        Ok(Self {
            block_size,
            gpu_memory_utilization,
            swap_space,
            swap_space_bytes,
            paged_attn_kernel_v,
        })
    }
}

fn get_cpu_memory() -> usize {
    // TODO
    64 * GB
}
