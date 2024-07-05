use anyhow::{anyhow, Result};
use tokenizers::{FromPretrainedParameters, Tokenizer};

pub use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

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

pub fn list_tokenizers() -> String {
    format!(
        "Available tokenizers for -t or --tokenizer:\n{}\n{}\n{}",
        tokenizers()
            .iter()
            .map(|t| format!("  -t {:16} {}", t.name, t.description))
            .collect::<Vec<_>>()
            .join("\n"),
        "You can also use a HuggingFace model name, in format 'user/modelname',",
        "or a local file in format './path/to/tokenizer.json'."
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

    let loaded = if name.starts_with(".") || name.starts_with("/") {
        Tokenizer::from_file(name)
    } else {
        let mut name2 = name.to_string();
        let mut args = FromPretrainedParameters::default();

        match strip_suffix("@", &mut name2) {
            Some(s) => args.revision = s,
            None => {}
        }
        Tokenizer::from_pretrained(name2, Some(args))
    };

    match loaded {
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
