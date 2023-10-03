use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize)]
pub struct TokenInfo {
    pub eos_token: u32,
    pub special: BTreeMap<String, u32>,
    pub binary: BTreeMap<String, u32>,
    pub text: BTreeMap<String, u32>,
}

pub struct Tokenizer {
    pub name: String,
    pub description: String,
    pub info: Option<TokenInfo>,
    info_bytes: &'static [u8],
}

macro_rules! tok {
    ($name:literal, $desc:literal) => {
        Tokenizer {
            name: $name.into(),
            description: $desc.into(),
            info_bytes: include_bytes!(concat!("tokenizers/", $name, ".json")),
            info: None,
        }
    };
}

pub fn tokenizers() -> Vec<Tokenizer> {
    vec![
        tok!("gpt4", "cl100k_base, used by GPT-4 and GPT-3.5"),
        tok!("llama", "used by Llama, CodeLlama, etc."),
        tok!("falcon", "used by Falcon 7b, 40b, etc."),
        tok!("mpt", "MPT"),
        tok!("phi", "Phi 1.5"),
        tok!("gpt2", "GPT-2"),
    ]
}

fn from_hex(hex_str: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for i in (0..hex_str.len()).step_by(2) {
        bytes.push(u8::from_str_radix(&hex_str[i..(i + 2)], 16)?);
    }
    Ok(bytes)
}

impl Tokenizer {
    pub fn load(&mut self) {
        if self.info.is_none() {
            self.info = Some(serde_json::from_slice::<TokenInfo>(self.info_bytes).unwrap());
        }
    }
    pub fn token_bytes(&self) -> Vec<Vec<u8>> {
        let mut r = Vec::new();

        let info = self.info.as_ref().unwrap();

        for (k, v) in &info.text {
            let idx = *v as usize;
            if r.len() <= idx {
                r.resize(idx + 1, Vec::new())
            }
            r[idx] = k.as_bytes().to_vec();
        }

        for (k, v) in &info.binary {
            let idx = *v as usize;
            if r.len() <= idx {
                r.resize(idx + 1, Vec::new())
            }

            r[idx] = from_hex(k).unwrap();
        }

        r
    }
}
