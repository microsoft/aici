// based on https://github.com/huggingface/candle/blob/main/candle-transformers/src/generation/mod.rs

use crate::{
    config::{SamplingParams, SAMPLING_EPS},
    util::to_vec1,
    DType, Tensor,
};
use aici_abi::toktree::TokTrie;
use anyhow::Result;
use rand::{distributions::Distribution, SeedableRng};
use std::sync::Arc;

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    temperature: Option<f32>,
    top_p: f32,
    #[allow(dead_code)]
    tokenizer: Arc<TokTrie>,
}

impl LogitsProcessor {
    pub fn new(sampling_params: &SamplingParams, tokenizer: Arc<TokTrie>) -> Self {
        let temperature = if sampling_params.temperature < SAMPLING_EPS {
            None
        } else {
            Some(sampling_params.temperature)
        };

        Self {
            rng: rand::rngs::StdRng::seed_from_u64(42),
            temperature,
            top_p: sampling_params.top_p,
            tokenizer,
        }
    }

    fn sample_argmax(&mut self, logits: &Tensor) -> Result<u32> {
        let tokid = logits.argmax(0, false).int64_value(&[]);
        // let tokid = to_vec1::<f32>(logits)
        //     .into_iter()
        //     .enumerate()
        //     .max_by(|(_, u), (_, v)| u.total_cmp(v))
        //     .unwrap()
        //     .0;
        Ok(tokid as u32)
    }

    fn sample_multinomial(&mut self, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(prs)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    fn sample_topp(&mut self, prs: &mut Vec<f32>, top_p: f32) -> Result<u32> {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability top_p. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].partial_cmp(&prs[i]).unwrap());

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        // Sample with clamped probabilities.
        self.sample_multinomial(prs)
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let next_token = match self.temperature {
            None => self.sample_argmax(&logits)?,
            Some(temperature) => {
                let logits = logits.to_kind(DType::Float);
                let logits = logits / (temperature as f64);
                let prs = logits.softmax(-1, DType::Float);

                let top_p = self.top_p;
                if top_p <= 0.0 || top_p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    let sample = prs.multinomial(1, false).int64_value(&[0]);
                    sample as u32
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    let mut prs: Vec<f32> = to_vec1(&prs);
                    self.sample_topp(&mut prs, top_p as f32)?
                }
            }
        };
        Ok(next_token)
    }
}
