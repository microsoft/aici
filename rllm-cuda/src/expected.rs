use aici_abi::bytes::vec_from_bytes;
use aicirt::api::Token;
use anyhow::Result;
use safetensors::Dtype;
use std::path::PathBuf;

use crate::{ExpectedGeneration, ExpectedToken};

fn read_flat_i32_vec(view: &impl safetensors::View) -> Vec<i32> {
    match view.dtype() {
        Dtype::I32 => vec_from_bytes(&view.data()),
        Dtype::I16 => vec_from_bytes::<i16>(&view.data())
            .iter()
            .map(|x| *x as i32)
            .collect(),
        Dtype::I8 => vec_from_bytes::<i8>(&view.data())
            .iter()
            .map(|x| *x as i32)
            .collect(),
        Dtype::U8 => vec_from_bytes::<u8>(&view.data())
            .iter()
            .map(|x| *x as i32)
            .collect(),
        Dtype::I64 => vec_from_bytes::<i64>(&view.data())
            .iter()
            .map(|x| (*x).try_into().expect("i64->i32 failed"))
            .collect(),
        Dtype::BOOL => vec_from_bytes::<u8>(&view.data())
            .iter()
            .map(|x| if *x != 0 { 1 } else { 0 })
            .collect(),
        _ => panic!("expected int type"),
    }
}

fn read_flat_f32_vec(view: &impl safetensors::View) -> Vec<f32> {
    match view.dtype() {
        Dtype::F32 => vec_from_bytes(&view.data()),
        Dtype::F16 => vec_from_bytes::<u16>(&view.data())
            .iter()
            .map(|x| half::f16::from_bits(*x).to_f32())
            .collect(),
        Dtype::BF16 => vec_from_bytes::<u16>(&view.data())
            .iter()
            .map(|x| half::bf16::from_bits(*x).to_f32())
            .collect(),
        Dtype::F64 => vec_from_bytes::<f64>(&view.data())
            .iter()
            .map(|x| *x as f32)
            .collect(),
        _ => read_flat_i32_vec(view).iter().map(|x| *x as f32).collect(),
    }
}

fn to_2d<T: Clone>(v: Vec<T>, view: &impl safetensors::View) -> Result<Vec<Vec<T>>> {
    let size = view.shape();
    if size.len() != 2 {
        anyhow::bail!("expected 2d tensor");
    }
    Ok((0..size[0])
        .map(|i| v[i * size[1]..(i + 1) * size[1]].to_vec())
        .collect())
}

impl ExpectedGeneration {
    pub fn load(f: &PathBuf) -> Result<Self> {
        let fp = std::fs::File::open(f)?;
        let content = unsafe { memmap2::MmapOptions::new().map(&fp)? };
        let s = safetensors::SafeTensors::deserialize(&content)?;

        let prompt = read_flat_i32_vec(&s.tensor("prompt")?);
        let output = read_flat_i32_vec(&s.tensor("output")?);
        let prob_mass = read_flat_f32_vec(&s.tensor("prob_mass")?);

        let view = s.tensor("tokens")?;
        let tokens = to_2d(read_flat_i32_vec(&view), &view)?;
        let view = s.tensor("logits")?;
        let logits = to_2d(read_flat_f32_vec(&view), &view)?;

        let num_tokens = output.len();
        assert!(tokens.len() == num_tokens);
        assert!(logits.len() == num_tokens);
        assert!(prob_mass.len() == num_tokens);

        Ok(ExpectedGeneration {
            prompt: prompt.into_iter().map(|x| x as Token).collect(),
            output: (0..num_tokens)
                .map(|i| ExpectedToken {
                    sampled: output[i] as Token,
                    ff_section_len: 1,
                    prob_mass: prob_mass[i],
                    logits: tokens[i]
                        .iter()
                        .zip(logits[i].iter())
                        .map(|(t, p)| (*t as Token, *p))
                        .collect(),
                })
                .collect(),
        })
    }
}
