use aici_abi::bytes::TokRxInfo;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize)]
pub struct TokenInfo {
    pub hf_model: String,
    pub eos_token: u32,
    pub vocab_size: Option<u32>,
    pub special: BTreeMap<String, u32>,
    pub binary: BTreeMap<String, u32>,
    pub text: BTreeMap<String, u32>,
}

pub struct Tokenizer {
    pub name: String,
    pub description: String,
    info: Option<TokenInfo>,
    base_tokenizer: Option<&'static str>,
    info_bytes: Option<&'static [u8]>,
    hf_bytes: Option<&'static [u8]>,
    add_tokens: Value,
}

macro_rules! tok {
    ($name:literal, $desc:literal) => {
        Tokenizer {
            name: $name.into(),
            description: $desc.into(),
            base_tokenizer: None,
            info_bytes: Some(include_bytes!(concat!("tokenizers/", $name, ".json"))),
            hf_bytes: Some(include_bytes!(concat!("hf-tokenizers/", $name, ".json"))),
            info: None,
            add_tokens: json!({}),
        }
    };
    ($username:literal, $name:literal, $desc:literal, $add:expr) => {
        Tokenizer {
            name: $username.into(),
            description: $desc.into(),
            base_tokenizer: Some($name),
            info_bytes: None,
            hf_bytes: None,
            info: None,
            add_tokens: $add,
        }
    };
}

pub fn tokenizers() -> Vec<Tokenizer> {
    vec![
        tok!("gpt4", "cl100k_base, used by GPT-4 and GPT-3.5"),
        tok!("llama", "used by Llama, CodeLlama, etc."),
        tok!(
            "llama16",
            "llama",
            "same as llama, with 16 added tokens (used by 13B codellama)",
            json!({
                "▁<SU": 32000,
                "▁<SUF": 32001,
                "▁<PRE": 32002,
                "▁<M": 32003,
                "▁<MID": 32004,
                "▁<E": 32005,
                "▁<EOT": 32006,
                "▁<PRE>": 32007,
                "▁<SUF>": 32008,
                "▁<MID>": 32009,
                "▁<EOT>": 32010,
                "▁<EOT><EOT>": 32011,
                "▁<EOT><EOT><EOT>": 32012,
                "▁<EOT><EOT><EOT><EOT>": 32013,
                "▁<EOT><EOT><EOT><EOT><EOT>": 32014,
                "▁<EOT><EOT><EOT><EOT><EOT><EOT>": 32015
            })
        ),
        tok!("falcon", "used by Falcon 7b, 40b, etc."),
        tok!("mpt", "MPT"),
        tok!("phi", "Phi 1.5"),
        tok!("gpt2", "GPT-2"),
    ]
}

pub fn find_tokenizer(name: &str) -> Result<Tokenizer> {
    for mut t in tokenizers() {
        if t.name == name {
            t.load();
            return Ok(t);
        }
    }

    println!("unknown tokenizer: {}", name);
    println!("available tokenizers:");
    for t in tokenizers() {
        println!("  {:20} {}", t.name, t.description);
    }
    return Err(anyhow!("unknown tokenizer: {}", name));
}

fn from_hex(hex_str: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for i in (0..hex_str.len()).step_by(2) {
        bytes.push(u8::from_str_radix(&hex_str[i..(i + 2)], 16)?);
    }
    Ok(bytes)
}

impl Tokenizer {
    fn load(&mut self) {
        if self.info.is_none() {
            let mut info = serde_json::from_slice::<TokenInfo>(self.get_info_bytes()).unwrap();
            self.add_tokens
                .as_object_mut()
                .unwrap()
                .iter()
                .for_each(|(k, v)| {
                    info.special
                        .insert(k.to_string(), v.as_u64().unwrap() as u32);
                });
            let max = vec![
                info.binary.values().max(),
                info.special.values().max(),
                info.text.values().max(),
            ]
            .iter()
            .filter_map(|x| *x)
            .max()
            .unwrap();
            assert!(*max < 1_000_000);
            info.vocab_size = Some(max + 1);
            self.info = Some(info);
        }
    }
    fn get_info_bytes(&self) -> &'static [u8] {
        match self.info_bytes {
            Some(x) => x,
            None => {
                let base = find_tokenizer(self.base_tokenizer.unwrap()).unwrap();
                base.get_info_bytes()
            }
        }
    }
    fn get_hf_bytes_raw(&self) -> &'static [u8] {
        match self.hf_bytes {
            Some(x) => x,
            None => {
                let base = find_tokenizer(self.base_tokenizer.unwrap()).unwrap();
                base.get_hf_bytes_raw()
            }
        }
    }
    pub fn get_hf_bytes(&self) -> Vec<u8> {
        let mut obj: Value = serde_json::from_slice(self.get_hf_bytes_raw()).unwrap();
        self.add_tokens
            .as_object()
            .unwrap()
            .iter()
            .for_each(|(k, v)| {
                obj["added_tokens"].as_array_mut().unwrap().push(json!({
                    "id": v.as_u64().unwrap(),
                    "content": k.to_string(),
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }));
                obj["model"]["vocab"][k] = v.clone();
            });
        serde_json::to_vec_pretty(&obj).unwrap()
    }
    pub fn tokrx_info(&self) -> TokRxInfo {
        let info = self.info.as_ref().unwrap();
        TokRxInfo {
            vocab_size: info.vocab_size.unwrap(),
            tok_eos: info.eos_token,
        }
    }
    pub fn token_bytes(&self) -> Vec<Vec<u8>> {
        let tinfo = self.tokrx_info();
        let mut r = Vec::with_capacity(tinfo.vocab_size as usize);
        r.resize_with(tinfo.vocab_size as usize, Vec::new);

        let info = self.info.as_ref().unwrap();

        for (k, v) in &info.text {
            let idx = *v as usize;
            r[idx] = k.as_bytes().to_vec();
        }

        for (k, v) in &info.binary {
            let idx = *v as usize;
            r[idx] = from_hex(k).unwrap();
        }

        r
    }
}
