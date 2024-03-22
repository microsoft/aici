use aici_abi::{
    arg_bytes, bytes::to_hex_string, tokenize_bytes, toktree::TokTrie, AiciCtrl, MidProcessArg,
    MidProcessResult, PostProcessArg, PostProcessResult, PreProcessArg, PreProcessResult, TokenId,
};
use base64::{self, Engine as _};
use earley::{earley_grm_from_guidance, Parser};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::earley::ParseResult;

mod earley;
mod serialization;

const INFO: bool = true;

macro_rules! infoln {
    ($($arg:tt)*) => {
        if INFO {
            println!($($arg)*);
        }
    };
}

pub struct Runner {
    toktrie: TokTrie,
    parser: Parser,
    llm_tokens: Vec<TokenId>,
    is_ff: bool,
    reported_captures: usize,
}

#[derive(Serialize, Deserialize)]
struct RunnerArg {
    guidance_b64: String,
}

impl Runner {
    pub fn new() -> Self {
        let arg: RunnerArg = serde_json::from_slice(&arg_bytes()).expect("invalid JSON arg");
        let guidance = base64::engine::general_purpose::STANDARD
            .decode(arg.guidance_b64)
            .expect("invalid base64");
        let grm = earley_grm_from_guidance(&guidance).expect("invalid guidance protobuf");
        infoln!("original: {:?}", grm);
        let grm = grm.optimize();
        infoln!("optimized: {:?}", grm);
        let cgrm = grm.compile();
        let parser = Parser::new(cgrm);
        Runner {
            toktrie: TokTrie::from_host(),
            parser,
            llm_tokens: Vec::new(),
            is_ff: false,
            reported_captures: 0,
        }
    }

    fn report_captures(&mut self) {
        let captures = &self.parser.captures()[self.reported_captures..];
        for (name, val) in captures {
            self.reported_captures += 1;
            let mut obj = json!({
                "type": "capture",
                "name": name,
            });
            match String::from_utf8(val.clone()) {
                Ok(s) => {
                    obj["string"] = json!(s);
                }
                Err(_) => {
                    obj["hex"] = json!(to_hex_string(val));
                }
            };
            println!("JSON-OUT: {}", obj);
        }
    }
}

impl AiciCtrl for Runner {
    fn pre_process(&mut self, _arg: PreProcessArg) -> PreProcessResult {
        PreProcessResult::continue_()
    }

    fn mid_process(&mut self, _arg: MidProcessArg) -> MidProcessResult {
        let start_time = std::time::Instant::now();
        let _ = self.parser.force_bytes();
        let fixed_bytes = self.parser.get_bytes();
        let mut fixed_tokens = tokenize_bytes(&fixed_bytes);
        let mut suff = Vec::new();
        let mut chop_tokens = 0;
        let mut chop_bytes = 0;
        for (idx, t) in fixed_tokens.iter().rev().enumerate() {
            suff.splice(0..0, self.toktrie.token(*t).iter().cloned());
            if suff.len() > self.toktrie.max_token_len() {
                break;
            }
            if self.toktrie.has_extensions(&suff) {
                chop_tokens = idx + 1;
                chop_bytes = suff.len();
            }
        }
        fixed_tokens.truncate(fixed_tokens.len() - chop_tokens);

        for idx in 0..fixed_tokens.len() {
            if self.llm_tokens.get(idx) != fixed_tokens.get(idx) {
                let backtrack: u32 = (self.llm_tokens.len() - idx).try_into().unwrap();
                let ff_tokens = fixed_tokens[idx..].to_vec();
                infoln!(
                    "backtrack: {}, ff_tokens: {}",
                    backtrack,
                    self.toktrie.tokens_dbg(&ff_tokens)
                );
                self.llm_tokens = fixed_tokens;
                self.is_ff = true;
                self.report_captures();
                return MidProcessResult::Splice {
                    backtrack,
                    ff_tokens,
                };
            }
        }

        let llm_bytes = self.toktrie.decode(&self.llm_tokens[fixed_tokens.len()..]);
        let byte_suffix = fixed_bytes[fixed_bytes.len() - chop_bytes..].to_vec();

        let byte_suffix = if byte_suffix.len() <= llm_bytes.len() {
            if !llm_bytes.starts_with(&byte_suffix) {
                panic!("llm_bytes: {:?}, byte_suffix: {:?}", llm_bytes, byte_suffix);
            }

            for b in &llm_bytes[byte_suffix.len()..] {
                let r = self.parser.scan(*b);
                if r == ParseResult::Reject {
                    panic!("rejected byte: {}", b);
                }
            }
            vec![]
        } else {
            if !byte_suffix.starts_with(&llm_bytes) {
                panic!("llm_bytes: {:?}, byte_suffix: {:?}", llm_bytes, byte_suffix);
            }
            byte_suffix[llm_bytes.len()..].to_vec()
        };

        self.is_ff = false;

        let mut set = self.toktrie.alloc_token_set();
        self.toktrie
            .compute_bias_ext(&mut self.parser, &mut set, &byte_suffix);
        infoln!(
            "bias: (pref: {:?}) {:?} {}",
            String::from_utf8_lossy(&byte_suffix),
            start_time.elapsed(),
            self.toktrie.token_set_dbg(&set)
        );

        self.report_captures();

        MidProcessResult::SampleWithBias {
            allowed_tokens: set,
        }
    }

    fn post_process(&mut self, arg: PostProcessArg) -> PostProcessResult {
        infoln!(
            "post tokens:{} {}",
            if self.is_ff { " ff" } else { "" },
            self.toktrie.tokens_dbg(&arg.tokens)
        );
        if !self.is_ff {
            self.llm_tokens.extend(&arg.tokens);
        }
        // TODO EOS!
        PostProcessResult::from_arg(&arg)
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        earley::bench::earley_test(TokTrie::from_host());
    }
}

aici_abi::aici_expose_all!(Runner, Runner::new());
