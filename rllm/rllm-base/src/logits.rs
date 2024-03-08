// based on https://github.com/huggingface/candle/blob/main/candle-transformers/src/generation/mod.rs

use crate::config::{SamplingParams, SAMPLING_EPS};
use rand::SeedableRng;

pub struct LogitsProcessor {
    pub rng: rand::rngs::StdRng,
    pub temperature: Option<f32>,
    pub top_p: f32,
}

impl LogitsProcessor {
    pub fn new(sampling_params: &SamplingParams) -> Self {
        let temperature = if sampling_params.temperature < SAMPLING_EPS {
            None
        } else {
            Some(sampling_params.temperature)
        };

        Self {
            rng: rand::rngs::StdRng::from_entropy(),
            // seed_from_u64(42),
            temperature,
            top_p: sampling_params.top_p,
        }
    }
}
