use crate::HashMap;
use aici_abi::bytes::TokRxInfo;
use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fmt::Debug};
use tokenizers::Tokenizer;

#[derive(Serialize, Deserialize)]
pub struct TokenInfo {
    pub hf_model: String,
    pub eos_token: u32,
    pub vocab_size: Option<u32>,
    pub special: BTreeMap<String, u32>,
    pub binary: BTreeMap<String, u32>,
    pub text: BTreeMap<String, u32>,
}

pub struct BinTokenizer {
    pub name: String,
    pub description: String,
    info: Option<TokenInfo>,
    base_tokenizer: Option<&'static str>,
    info_bytes: Option<&'static [u8]>,
    hf_bytes: Option<&'static [u8]>,
    add_tokens: Value,
    model_ids: String,
}

macro_rules! tok {
    ($name:literal, $desc:literal, $models:literal) => {
        BinTokenizer {
            name: $name.into(),
            description: $desc.into(),
            base_tokenizer: None,
            info_bytes: Some(include_bytes!(concat!("tokenizers/", $name, ".json"))),
            hf_bytes: Some(include_bytes!(concat!("hf-tokenizers/", $name, ".json"))),
            info: None,
            add_tokens: json!({}),
            model_ids: $models.into(),
        }
    };
    ($username:literal, $name:literal, $desc:literal, $models:literal, $add:expr) => {
        BinTokenizer {
            name: $username.into(),
            description: $desc.into(),
            base_tokenizer: Some($name),
            info_bytes: None,
            hf_bytes: None,
            info: None,
            add_tokens: $add,
            model_ids: $models.into(),
        }
    };
}

pub fn tokenizers() -> Vec<BinTokenizer> {
    vec![
        tok!("gpt4", "cl100k_base, used by GPT-4 and GPT-3.5", "gpt-4"),
        tok!("llama", "used by Llama, CodeLlama, etc.", ""),
        tok!(
            "llama16",
            "llama",
            "same as llama, with 16 added tokens (used by 13B codellama)",
            "codellama-13b",
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
        tok!(
            "orca",
            "llama",
            "for microsoft/Orca models; similar to llama, with 3 tokens added for chat",
            "",
            json!({
                "<|im_end|>": 32002,
                "<|im_start|>": 32001,
                "[PAD]": 32000
            })
        ),
        tok!("falcon", "used by Falcon 7b, 40b, etc.", ""),
        tok!("mistral", "used by Mistral and Mixtral", "mixtral"),
        tok!("mpt", "MPT", ""),
        tok!("phi", "Phi 1.5 and Phi 2", ""),
        tok!("gpt2", "GPT-2", "gpt-2"),
    ]
}

fn is_self_mapped(c: char) -> bool {
    match c {
        '!'..='~' | '\u{00A1}'..='\u{00AC}' | '\u{00AE}'..='\u{00FF}' => true,
        _ => false,
    }
}

fn build_char_map() -> HashMap<char, u8> {
    let mut res = HashMap::default();
    let mut k = 0x100u32;
    for byte in 0..=255u8 {
        let c = byte as char;
        if is_self_mapped(c) {
            res.insert(c, byte);
        } else {
            res.insert(char::from_u32(k).unwrap(), byte);
            k += 1;
        }
    }
    res
}

fn build_bintok(hft: Tokenizer) -> Result<TokenInfo> {
    let mut is_byte_level = false;
    let mut is_byte_fallback = false;
    let mut space_ch = ' ';

    if let Some(d) = hft.get_decoder() {
        let v = serde_json::to_value(d).unwrap();
        if v["type"].as_str() == Some("ByteLevel") {
            is_byte_level = true;
        } else if v["type"].as_str() == Some("Sequence") {
            if let Some(decoders) = v["decoders"].as_array() {
                for decoder in decoders {
                    if decoder["type"].as_str() == Some("ByteFallback") {
                        is_byte_fallback = true;
                    } else if decoder["type"].as_str() == Some("Replace")
                        && decoder["content"].as_str() == Some(" ")
                    {
                        if let Some(s) = decoder["pattern"]["String"].as_str() {
                            let s: Vec<char> = s.chars().collect();
                            if s.len() == 1 {
                                space_ch = s[0];
                            }
                        }
                    }
                }
            }
        }
    }

    if !is_byte_fallback && !is_byte_level {
        bail!("can't determine decoder type: {:?}", hft.get_decoder());
    }

    let vocab_size = hft.get_vocab_size(true) as u32;
    let mut res = TokenInfo {
        hf_model: "foobar".to_string(),
        eos_token: 0,
        vocab_size: Some(vocab_size),
        special: BTreeMap::new(),
        binary: BTreeMap::new(),
        text: BTreeMap::new(),
    };

    for (id, info) in hft.get_added_tokens_decoder().iter() {
        if info.special {
            match info.content.as_str() {
                "</s>" | "<|endoftext|>" => res.eos_token = *id,
                _ => {}
            }
            res.special.insert(info.content.clone(), *id);
        } else {
            res.text.insert(info.content.clone(), *id);
        }
    }

    let added = hft.get_added_tokens_decoder();
    let char_map = build_char_map();

    for tok_id in 0..vocab_size {
        if added.contains_key(&tok_id) {
            continue;
        }
        if let Some(tok_name) = hft.id_to_token(tok_id) {
            if is_byte_fallback {
                if tok_name.len() == 6 && tok_name.starts_with("<0x") && tok_name.ends_with(">") {
                    // parse hex number from tok_name
                    let hex_str = &tok_name[3..5];
                    let byte = u8::from_str_radix(hex_str, 16).unwrap();
                    if byte >= 0x80 {
                        let s = format!("{:02x}", byte);
                        res.binary.insert(s, tok_id);
                    } else {
                        let s = format!("{}", byte as char);
                        res.text.insert(s, tok_id);
                    }
                } else {
                    assert!(!tok_name.starts_with("<0x"));
                    let tok_name = tok_name.replace(space_ch, " ");
                    res.text.insert(tok_name, tok_id);
                }
            } else if is_byte_level {
                let bytes: Result<Vec<u8>> = tok_name
                    .chars()
                    .map(|c| {
                        char_map
                            .get(&c)
                            .map(|c| *c)
                            .ok_or_else(|| anyhow!("missing char: {}", c))
                    })
                    .collect();
                let bytes = match bytes {
                    Ok(b) => b,
                    Err(e) => {
                        println!("error: {} for {:?}", e, tok_name);
                        continue;
                    }
                };

                if let Ok(s) = String::from_utf8(bytes.clone()) {
                    res.text.insert(s, tok_id);
                } else {
                    let hexstr = String::from_iter(bytes.iter().map(|b| format!("{:02x}", b)));
                    res.binary.insert(hexstr, tok_id);
                }
            } else {
                panic!();
            }
        } else {
            println!("missing token: {}", tok_id);
        }
    }

    Ok(res)
}

fn cmp_maps<T: Ord + Eq + Debug, U: Eq + Debug>(a: &BTreeMap<T, U>, b: &BTreeMap<T, U>) {
    for (k, v) in a {
        if b.get(k) != Some(v) {
            println!("{:?}: {:?} != {:?}", k, v, b.get(k));
        }
    }
    for (k, v) in b {
        if a.get(k) == None {
            println!("{:?}: None != {:?}", k, v);
        }
    }
}

pub fn convert_tokenizers() {
    for bintok in tokenizers() {
        if bintok.base_tokenizer.is_some() {
            continue;
        }

        if bintok.name != "llama" && bintok.name != "phi" {
            //continue;
        }

        //

        let mut info = serde_json::from_slice::<TokenInfo>(bintok.get_info_bytes()).unwrap();
        let max = vec![
            info.binary.values().max(),
            info.special.values().max(),
            info.text.values().max(),
        ]
        .iter()
        .filter_map(|x| *x)
        .max()
        .unwrap();
        info.vocab_size = Some(max + 1);

        println!("{}: {}", bintok.name, max + 1);
        let hft = Tokenizer::from_bytes(bintok.hf_bytes.unwrap()).unwrap();
        let info2 = build_bintok(hft).unwrap();

        assert!(info.eos_token == info2.eos_token);
        assert!(info.vocab_size == info2.vocab_size);
        cmp_maps(&info.special, &info2.special);
        cmp_maps(&info.binary, &info2.binary);
        cmp_maps(&info.text, &info2.text);
    }
}

pub fn list_tokenizers() -> String {
    format!(
        "Available tokenizers for -t or --tokenizer:\n{}",
        tokenizers()
            .iter()
            .map(|t| format!("  -t {:16} {}", t.name, t.description))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

pub fn guess_tokenizer(model_name: &str) -> Option<String> {
    let m = model_name.to_lowercase();
    tokenizers()
        .iter()
        .find(|t| {
            m.contains(&t.name)
                || t.model_ids
                    .split(',')
                    .map(|x| x.trim())
                    .filter(|x| x.len() > 0)
                    .any(|x| m.contains(x))
        })
        .map(|t| t.name.clone())
}

pub fn find_tokenizer(name: &str) -> Result<BinTokenizer> {
    for mut t in tokenizers() {
        if t.name == name {
            t.load();
            return Ok(t);
        }
    }

    println!("unknown tokenizer: {}", name);
    println!("{}", list_tokenizers());
    return Err(anyhow!("unknown tokenizer: {}", name));
}

fn from_hex(hex_str: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for i in (0..hex_str.len()).step_by(2) {
        bytes.push(u8::from_str_radix(&hex_str[i..(i + 2)], 16)?);
    }
    Ok(bytes)
}

impl BinTokenizer {
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
