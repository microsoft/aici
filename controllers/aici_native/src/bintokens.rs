use aici_abi::bytes::TokRxInfo;
use anyhow::{anyhow, bail, Result};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tokenizers::{normalizers::Sequence, FromPretrainedParameters, NormalizerWrapper, Tokenizer};

#[derive(Serialize, Deserialize)]
pub struct ByteTokenizer {
    pub hf_model: String,
    pub hf_tokenizer: Tokenizer,
    pub eos_token: u32,
    pub vocab_size: u32,
    token_bytes: Vec<Vec<u8>>,
    pub special: BTreeMap<String, u32>,
}

pub struct TokenizerInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub hf_model: &'static str,
    pub model_ids: &'static str,
}

pub fn tokenizers() -> Vec<TokenizerInfo> {
    vec![
        TokenizerInfo {
            name: "gpt4",
            description: "cl100k_base, used by GPT-4 and GPT-3.5",
            hf_model: "Xenova/gpt-4",
            model_ids: "gpt-4",
        },
        TokenizerInfo {
            name: "llama16",
            description: "same as llama, with 16 added tokens (used by 13B codellama)",
            hf_model: "codellama/CodeLlama-13b-Instruct-hf",
            model_ids: "codellama-13b",
        },
        TokenizerInfo {
            name: "llama70",
            description: "used by codellama-70b; with <step> token",
            hf_model: "codellama/CodeLlama-70b-Instruct-hf",
            model_ids: "codellama-70b",
        },
        TokenizerInfo {
            name: "llama",
            description: "used by Llama, CodeLlama, etc.",
            hf_model: "codellama/CodeLlama-34b-Instruct-hf",
            model_ids: "",
        },
        TokenizerInfo {
            name: "orca",
            description: "llama",
            hf_model: "microsoft/Orca-2-13b@refs/pr/23",
            model_ids: "for microsoft/Orca models; similar to llama, with 3 tokens added for chat",
        },
        TokenizerInfo {
            name: "falcon",
            description: "used by Falcon 7b, 40b, etc.",
            hf_model: "tiiuae/falcon-7b",
            model_ids: "",
        },
        TokenizerInfo {
            name: "mistral",
            description: "used by Mistral and Mixtral",
            hf_model: "mistralai/Mistral-7B-Instruct-v0.2",
            model_ids: "mixtral",
        },
        TokenizerInfo {
            name: "mpt",
            description: "MPT",
            hf_model: "mosaicml/mpt-7b",
            model_ids: "",
        },
        TokenizerInfo {
            name: "phi",
            description: "Phi 1.5 and Phi 2",
            hf_model: "microsoft/phi-1_5",
            model_ids: "",
        },
        TokenizerInfo {
            name: "gpt2",
            description: "GPT-2",
            hf_model: "gpt2",
            model_ids: "gpt-2",
        },
    ]
}

// useful when debugging this: https://www.cogsci.ed.ac.uk/~richard/utf-8.cgi

fn is_self_mapped(c: char) -> bool {
    match c {
        '!'..='~' | '\u{00A1}'..='\u{00AC}' | '\u{00AE}'..='\u{00FF}' => true,
        _ => false,
    }
}

fn build_char_map() -> FxHashMap<char, u8> {
    let mut res = FxHashMap::default();
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

pub fn list_tokenizers() -> String {
    format!(
        "Available tokenizers for -t or --tokenizer:\n{}\n{}",
        tokenizers()
            .iter()
            .map(|t| format!("  -t {:16} {}", t.name, t.description))
            .collect::<Vec<_>>()
            .join("\n"),
        "You can also use a HuggingFace model name, in format 'user/modelname'."
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
        .map(|t| t.name.to_string())
}

fn strip_suffix(sep: &str, s: &mut String) -> Option<String> {
    let mut parts = s.splitn(2, sep);
    let core = parts.next().unwrap().to_string();
    let suff = parts.next().map(|s| s.to_string());
    *s = core;
    suff
}

pub fn test_tokenizers() {
    for t in tokenizers() {
        let t = find_tokenizer(t.name).unwrap();
        println!("tokenizer: {} {}", t.hf_model, t.vocab_size);
    }
}

pub fn find_tokenizer(mut name: &str) -> Result<ByteTokenizer> {
    if !name.contains("/") {
        for t in tokenizers() {
            if t.name == name {
                name = t.hf_model;
                break;
            }
        }
    }

    log::info!("loading tokenizer: {}", name);

    let mut name2 = name.to_string();
    let mut args = FromPretrainedParameters::default();

    match strip_suffix("@", &mut name2) {
        Some(s) => args.revision = s,
        None => {}
    }

    match Tokenizer::from_pretrained(name2, Some(args)) {
        Err(e) => {
            let msg = format!("can't load tokenizer {}: {}", name, e);
            println!("{}\n{}", msg, list_tokenizers());
            return Err(anyhow!("{}", msg));
        }
        Ok(t) => {
            let bt = ByteTokenizer::from_tokenizer(t)?;
            Ok(bt)
        }
    }
}

impl ByteTokenizer {
    pub fn from_tokenizer(mut hft: Tokenizer) -> Result<ByteTokenizer> {
        let mut is_byte_level = false;
        let mut is_byte_fallback = false;
        let mut space_ch = ' ';

        // remove the "Prepend space"
        if let Some(n) = hft.get_normalizer() {
            let n = match n {
                NormalizerWrapper::Sequence(x) => NormalizerWrapper::Sequence(Sequence::new(
                    x.get_normalizers()
                        .iter()
                        .filter_map(|n| match n {
                            NormalizerWrapper::Prepend(_) => None,
                            _ => Some(n.clone()),
                        })
                        .collect(),
                )),
                _ => n.clone(),
            };
            hft.with_normalizer(n);
        }

        if let Some(d) = hft.get_decoder() {
            // DecoderWrapper::Sequence() doesn't let one access the decoders
            // so we resort to json munching
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
        let added = hft.get_added_tokens_decoder();

        let mut res = ByteTokenizer {
            hf_model: "foobar".to_string(),
            eos_token: 0,
            vocab_size,
            special: BTreeMap::new(),
            token_bytes: (0..vocab_size).map(|_| Vec::new()).collect(),
            hf_tokenizer: hft,
        };

        for (id, info) in added.iter() {
            if info.special {
                match info.content.as_str() {
                    "</s>" | "<|endoftext|>" => res.eos_token = *id,
                    _ => {}
                }
                res.special.insert(info.content.clone(), *id);
            } else {
                res.token_bytes[*id as usize] = info.content.clone().into_bytes();
            }
        }

        let char_map = build_char_map();

        for tok_id in 0..vocab_size {
            if added.contains_key(&tok_id) {
                continue;
            }
            if let Some(tok_name) = res.hf_tokenizer.id_to_token(tok_id) {
                if is_byte_fallback {
                    if tok_name.len() == 6 && tok_name.starts_with("<0x") && tok_name.ends_with(">")
                    {
                        // parse hex number from tok_name
                        let hex_str = &tok_name[3..5];
                        let byte = u8::from_str_radix(hex_str, 16).unwrap();
                        res.token_bytes[tok_id as usize] = vec![byte];
                    } else {
                        assert!(!tok_name.starts_with("<0x"));
                        let tok_name = tok_name.replace(space_ch, " ");
                        res.token_bytes[tok_id as usize] = tok_name.as_bytes().to_vec();
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

                    res.token_bytes[tok_id as usize] = bytes;
                } else {
                    panic!();
                }
            } else {
                log::warn!("missing token: {}", tok_id);
            }
        }

        Ok(res)
    }
}

impl ByteTokenizer {
    pub fn tokrx_info(&self) -> TokRxInfo {
        TokRxInfo {
            vocab_size: self.vocab_size,
            tok_eos: self.eos_token,
        }
    }
    pub fn token_bytes(&self) -> Vec<Vec<u8>> {
        self.token_bytes.clone()
    }
}
