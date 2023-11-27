// based on https://github.com/huggingface/candle/blob/main/candle-transformers/src/generation/mod.rs

use candle::{DType, Error, Result, Tensor};
use rand::{distributions::Distribution, SeedableRng};

use crate::config::{SamplingParams, SAMPLING_EPS};

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    temperature: Option<f32>,
    top_p: f32,
}

impl LogitsProcessor {
    pub fn new(sampling_params: &SamplingParams) -> Self {
        let temperature = if sampling_params.temperature < SAMPLING_EPS {
            None
        } else {
            Some(sampling_params.temperature)
        };

        Self {
            rng: rand::rngs::StdRng::seed_from_u64(42),
            temperature,
            top_p: sampling_params.top_p,
        }
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<u32> {
        let logits_v: Vec<f32> = logits.to_vec1()?;
        let next_token = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, u), (_, v)| u.total_cmp(v))
            .map(|(i, _)| i as u32)
            .unwrap();
        Ok(next_token)
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
