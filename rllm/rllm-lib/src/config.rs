// based on https://github.com/vllm-project/vllm/blob/b9fe4616f98b77b4b9458bce203aa6544cb31ef2/vllm/config.py

use crate::ModelExec;
use aicirt::{bail_user, valid_module_or_tag};
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct RllmConfig<ME: ModelExec> {
    pub model: ME::ModelConfig,
    pub meta: ModelMeta,
    pub parallel: ParallelConfig,
    pub scheduler: SchedulerConfig,
    pub aici: AiciConfig,
}

#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub id: String,
    pub max_sequence_length: usize,
    pub vocab_size: usize,
    pub tok_vocab_size: usize,
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
pub struct SchedulerConfig {
    /// Maximum number of tokens to be processed in a single iteration (passed through FFN).
    pub max_num_batched_tokens: usize,
    /// Maximum number of KV entries to be processed in a single iteration.
    pub max_num_kv_tokens: usize,
    /// Maximum number of sequences to be processed in a single iteration.
    pub max_num_seqs: usize,
    /// Maximum length of a sequence (including prompt and generated text).
    pub max_model_len: usize,
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
    pub controller: Option<String>,

    /// What argument to pass to the module.
    pub controller_arg: String,

    /// Maximum number of tokens to use as fuel for the AICI module.
    pub aici_fuel: Option<usize>,

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
            controller: None,
            controller_arg: String::new(),
            aici_fuel: None,
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
        if let Some(mod_id) = self.controller.as_ref() {
            if !valid_module_or_tag(mod_id) {
                bail_user!(
                    "'controller' must be a 64-char hex string or tag name, got {}.",
                    mod_id
                );
            }
        }

        if self.n < 1 {
            bail_user!("n must be at least 1, got {}.", self.n);
        }
        if self.best_of < self.n {
            bail_user!(
                "best_of must be greater than or equal to n, got n={} and best_of={}.",
                self.n,
                self.best_of
            );
        }
        if !(self.presence_penalty >= -2.0 && self.presence_penalty <= 2.0) {
            bail_user!(
                "presence_penalty must be in [-2, 2], got {}.",
                self.presence_penalty
            );
        }
        if !(self.frequency_penalty >= -2.0 && self.frequency_penalty <= 2.0) {
            bail_user!(
                "frequency_penalty must be in [-2, 2], got {}.",
                self.frequency_penalty
            );
        }
        if self.temperature < 0.0 {
            bail_user!(
                "temperature must be non-negative, got {}.",
                self.temperature
            );
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            bail_user!("top_p must be in (0, 1], got {}.", self.top_p);
        }
        if self.top_k < -1 || self.top_k == 0 {
            bail_user!(
                "top_k must be -1 (disable), or at least 1, got {}.",
                self.top_k
            );
        }
        if self.max_tokens < 1 {
            bail_user!("max_tokens must be at least 1, got {}.", self.max_tokens);
        }
        if let Some(logprobs) = self.logprobs {
            if logprobs < 0 {
                bail_user!("logprobs must be non-negative, got {}.", logprobs);
            }
        }
        Ok(())
    }

    fn _verify_beam_search(&self) -> Result<()> {
        if self.use_beam_search {
            if self.best_of == 1 {
                bail_user!(
                    "best_of must be greater than 1 when using beam search. Got {}.",
                    self.best_of
                );
            }
            if self.temperature > SAMPLING_EPS {
                bail_user!("temperature must be 0 when using beam search.");
            }
            if self.top_p < 1.0 - SAMPLING_EPS {
                bail_user!("top_p must be 1 when using beam search.");
            }
            if self.top_k != -1 {
                bail_user!("top_k must be -1 when using beam search.");
            }
            Ok(())
        } else {
            Ok(())
        }
    }

    fn _verify_non_beam_search(&self) -> Result<()> {
        if !self.use_beam_search {
            if let EarlyStopping::True = self.early_stopping {
                bail_user!(
                    "early_stopping is not effective and must be False when not using beam search."
                );
            }
            if !(self.length_penalty >= 1.0 - SAMPLING_EPS
                && self.length_penalty <= 1.0 + SAMPLING_EPS)
            {
                bail_user!("length_penalty is not effective and must be the default value of 1.0 when not using beam search.");
            }
        }
        Ok(())
    }

    fn _verify_greedy_sampling(&self) -> Result<()> {
        if self.temperature < SAMPLING_EPS {
            if self.best_of > 1 {
                bail_user!(
                    "best_of must be 1 when using greedy sampling. Got {}.",
                    self.best_of
                );
            }
            if self.top_p < 1.0 - SAMPLING_EPS {
                bail_user!("top_p must be 1 when using greedy sampling.");
            }
            if self.top_k != -1 {
                bail_user!("top_k must be -1 when using greedy sampling.");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AiciConfig {
    pub max_fuel: usize,
}

impl Default for AiciConfig {
    fn default() -> Self {
        Self { max_fuel: 0 }
    }
}
