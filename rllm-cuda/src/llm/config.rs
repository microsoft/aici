use crate::config::{ModelMeta, RllmConfig};
use aicirt::bail_user;
use anyhow::Result;
use tch::Device;

use super::{tmodel::TModel, DType};

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
