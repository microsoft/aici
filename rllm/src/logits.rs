// based on https://github.com/huggingface/candle/blob/main/candle-transformers/src/generation/mod.rs

use std::sync::Arc;

use candle_core::{DType, Error, Result, Tensor};
use rand::{distributions::Distribution, SeedableRng};
use tokenizers::Tokenizer;

use crate::config::{SamplingParams, SAMPLING_EPS};

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    temperature: Option<f32>,
    top_p: f32,
    tokenizer: Arc<Tokenizer>,
    pub num_ambiguous: usize,
}

impl LogitsProcessor {
    pub fn new(sampling_params: &SamplingParams, tokenizer: Arc<Tokenizer>) -> Self {
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
            num_ambiguous: 0,
        }
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<u32> {
        let mut logits_v: Vec<_> = logits.to_vec1::<f32>()?.into_iter().enumerate().collect();
        logits_v.sort_by(|u, v| v.1.total_cmp(&u.1));
        let d = (logits_v[0].1 - logits_v[1].1) / logits_v[0].1;
        if d < 0.05 {
            self.num_ambiguous += 1;
            log::debug!(
                "argmax: {:?} {:?} {:?} {:?}",
                logits_v[0],
                self.tokenizer
                    .decode(&vec![logits_v[0].0 as u32], false)
                    .unwrap(),
                logits_v[1],
                self.tokenizer
                    .decode(&vec![logits_v[1].0 as u32], false)
                    .unwrap(),
            );
        } else {
            log::trace!(
                "argmax: {:?} {:?} {:?}",
                logits_v[0],
                logits_v[1],
                logits_v[2]
            );
        }
        Ok(logits_v[0].0 as u32)
        // let next_token = logits_v
        //     .iter()
        //     .enumerate()
        //     .max_by(|(_, u), (_, v)| u.total_cmp(v))
        //     .map(|(i, _)| i as u32)
        //     .unwrap();
        // Ok(next_token)
    }

    fn sample_multinomial(&mut self, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(prs).map_err(Error::wrap)?;
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
        let logits = logits.to_dtype(DType::F32)?;
        let next_token = match self.temperature {
            None => self.sample_argmax(logits)?,
            Some(temperature) => {
                let logits = &(&logits / (temperature as f64))?;
                let prs = candle_nn::ops::softmax_last_dim(logits)?;
                let mut prs: Vec<f32> = prs.to_vec1()?;
                let top_p = self.top_p;
                if top_p <= 0.0 || top_p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    self.sample_multinomial(&prs)?
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&mut prs, top_p as f32)?
                }
            }
        };
        Ok(next_token)
    }
}
