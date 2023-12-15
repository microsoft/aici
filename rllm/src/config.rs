// based on https://github.com/vllm-project/vllm/blob/b9fe4616f98b77b4b9458bce203aa6544cb31ef2/vllm/config.py

use crate::{DType, Device};
use anyhow::{bail, Result};
use log::warn;
use serde::{Deserialize, Serialize};

static GB: usize = 1 << 30;

#[derive(Debug)]
pub struct RllmConfig {
    pub model: ModelConfig,
    pub parallel: ParallelConfig,
    pub cache: CacheConfig,
    pub scheduler: SchedulerConfig,

    pub dtype: DType,
    pub device: Device,
}

impl RllmConfig {
    pub fn new(
        model: ModelConfig,
        parallel: ParallelConfig,
        cache: CacheConfig,
        scheduler: SchedulerConfig,
    ) -> Result<Self> {
        if model.num_hidden_layers % parallel.pipeline_parallel_size != 0 {
            bail!(
                "Number of hidden layers ({}) must be divisible by the pipeline parallel size ({}).",
                model.num_hidden_layers,
                parallel.pipeline_parallel_size
            );
        }
        if model.num_key_value_heads % parallel.tensor_parallel_size != 0 {
            bail!(
                "Number of key/value heads ({}) must be divisible by the tensor parallel size ({}).",
                model.num_key_value_heads,
                parallel.tensor_parallel_size
            );
        }
        let dtype = model.dtype;
        let device = model.device;
        Ok(Self {
            model,
            parallel,
            cache,
            scheduler,
            device,
            dtype,
        })
    }

    pub fn get_hidden_size(&self) -> usize {
        self.model.hidden_size
    }
    pub fn get_head_size(&self) -> usize {
        self.model.hidden_size / self.model.num_attention_heads
    }
    pub fn get_num_heads_parallel(&self) -> usize {
        self.model.num_key_value_heads / self.parallel.tensor_parallel_size
    }
    pub fn get_num_layers_parallel(&self) -> usize {
        self.model.num_hidden_layers / self.parallel.pipeline_parallel_size
    }
    pub fn get_max_model_len(&self) -> usize {
        self.model.max_sequence_length
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Llama,
    Phi,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: ModelType,

    pub num_attention_heads: usize,
    pub hidden_size: usize, // head_dim * num_attention_heads
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub max_sequence_length: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,

    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub tok_vocab_size: usize,

    pub layer_norm_eps: f64, // defl. 1e-5
    pub rope_theta: f32,     // defl. 10000

    pub device: Device,
    pub dtype: DType,
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

#[derive(Debug, Serialize, Deserialize)]
pub struct ParallelConfig {
    pub pipeline_parallel_size: usize,
    pub tensor_parallel_size: usize,
}

impl ParallelConfig {
    pub fn single() -> Self {
        Self {
            pipeline_parallel_size: 1,
            tensor_parallel_size: 1,
        }
    }
}

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

impl Default for CacheConfig {
    fn default() -> Self {
        Self::new(16, 0.9, 4).unwrap()
    }
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

pub const SAMPLING_EPS: f32 = 1e-5;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum EarlyStopping {
    True,
    False,
    Never,
}

/// Sampling parameters for text generation.
///
/// Overall, we follow the sampling parameters from the OpenAI text completion
/// API (https://platform.openai.com/docs/api-reference/completions/create).
/// In addition, we support beam search, which is not supported by OpenAI.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SamplingParams {
    /// Which AICI module to run, if any.
    pub aici_module: Option<String>,

    /// What argument to pass to the module.
    pub aici_arg: String,

    /// Number of output sequences to return for the given prompt.
    pub n: usize,

    /// Number of output sequences that are generated from the prompt.
    pub best_of: usize,

    /// Float that penalizes new tokens based on whether they appear in the generated text so far.
    pub presence_penalty: f32,

    /// Float that penalizes new tokens based on their frequency in the generated text so far.
    pub frequency_penalty: f32,

    /// Float that controls the randomness of the sampling. Default is 1.0.
    pub temperature: f32,

    /// Float that controls the cumulative probability of the top tokens to consider. Default is 1.0.
    pub top_p: f32,

    /// Integer that controls the number of top tokens to consider. Default is -1.
    pub top_k: isize,

    /// Whether to use beam search instead of sampling.
    pub use_beam_search: bool,

    /// Float that penalizes sequences based on their length. Used in beam search.
    pub length_penalty: f32,

    /// Controls the stopping condition for beam search.
    pub early_stopping: EarlyStopping,

    /// List of strings that stop the generation when they are generated.
    pub stop: Vec<String>,

    /// Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.
    pub ignore_eos: bool,

    /// Maximum number of tokens to generate per output sequence.
    pub max_tokens: usize,

    /// Number of log probabilities to return per output token.
    pub logprobs: Option<i32>,
}

impl SamplingParams {
    pub fn default() -> Self {
        let r = Self {
            aici_module: None,
            aici_arg: String::new(),
            n: 1,
            best_of: 1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            temperature: 0.0,
            top_p: 1.0,
            top_k: -1,
            use_beam_search: false,
            length_penalty: 1.0,
            early_stopping: EarlyStopping::False,
            stop: Vec::new(),
            ignore_eos: false,
            max_tokens: 16,
            logprobs: None,
        };
        r.verify_args().unwrap();
        r
    }

    /// Verifies the arguments of the sampling parameters.
    pub fn verify_args(&self) -> Result<()> {
        self._verify_args()?;
        if self.use_beam_search {
            self._verify_beam_search()?;
        } else {
            self._verify_non_beam_search()?;
            if self.temperature < SAMPLING_EPS {
                self._verify_greedy_sampling()?;
            }
        }
        Ok(())
    }

    fn _verify_args(&self) -> Result<()> {
        fn is_hex_string(s: &str) -> bool {
            s.chars().all(|c| c.is_digit(16))
        }

        if let Some(mod_id) = self.aici_module.as_ref() {
            if !is_hex_string(mod_id) || mod_id.len() != 64 {
                bail!("aici_module must be a 64-char hex string, got {}.", mod_id);
            }
        }

        if self.n < 1 {
            bail!("n must be at least 1, got {}.", self.n);
        }
        if self.best_of < self.n {
            bail!(
                "best_of must be greater than or equal to n, got n={} and best_of={}.",
                self.n,
                self.best_of
            );
        }
        if !(self.presence_penalty >= -2.0 && self.presence_penalty <= 2.0) {
            bail!(
                "presence_penalty must be in [-2, 2], got {}.",
                self.presence_penalty
            );
        }
        if !(self.frequency_penalty >= -2.0 && self.frequency_penalty <= 2.0) {
            bail!(
                "frequency_penalty must be in [-2, 2], got {}.",
                self.frequency_penalty
            );
        }
        if self.temperature < 0.0 {
            bail!(
                "temperature must be non-negative, got {}.",
                self.temperature
            );
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            bail!("top_p must be in (0, 1], got {}.", self.top_p);
        }
        if self.top_k < -1 || self.top_k == 0 {
            bail!(
                "top_k must be -1 (disable), or at least 1, got {}.",
                self.top_k
            );
        }
        if self.max_tokens < 1 {
            bail!("max_tokens must be at least 1, got {}.", self.max_tokens);
        }
        if let Some(logprobs) = self.logprobs {
            if logprobs < 0 {
                bail!("logprobs must be non-negative, got {}.", logprobs);
            }
        }
        Ok(())
    }

    fn _verify_beam_search(&self) -> Result<()> {
        if self.use_beam_search {
            if self.best_of == 1 {
                bail!(
                    "best_of must be greater than 1 when using beam search. Got {}.",
                    self.best_of
                );
            }
            if self.temperature > SAMPLING_EPS {
                bail!("temperature must be 0 when using beam search.");
            }
            if self.top_p < 1.0 - SAMPLING_EPS {
                bail!("top_p must be 1 when using beam search.");
            }
            if self.top_k != -1 {
                bail!("top_k must be -1 when using beam search.");
            }
            Ok(())
        } else {
            Ok(())
        }
    }

    fn _verify_non_beam_search(&self) -> Result<()> {
        if !self.use_beam_search {
            if let EarlyStopping::True = self.early_stopping {
                bail!(
                    "early_stopping is not effective and must be False when not using beam search."
                );
            }
            if !(self.length_penalty >= 1.0 - SAMPLING_EPS
                && self.length_penalty <= 1.0 + SAMPLING_EPS)
            {
                bail!("length_penalty is not effective and must be the default value of 1.0 when not using beam search.");
            }
        }
        Ok(())
    }

    fn _verify_greedy_sampling(&self) -> Result<()> {
        if self.temperature < SAMPLING_EPS {
            if self.best_of > 1 {
                bail!(
                    "best_of must be 1 when using greedy sampling. Got {}.",
                    self.best_of
                );
            }
            if self.top_p < 1.0 - SAMPLING_EPS {
                bail!("top_p must be 1 when using greedy sampling.");
            }
            if self.top_k != -1 {
                bail!("top_k must be -1 when using greedy sampling.");
            }
        }
        Ok(())
    }
}
