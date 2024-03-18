use aici_abi::{
    arg_bytes, tokenize_bytes, toktree::TokTrie, AiciCtrl, MidProcessArg, MidProcessResult,
    PostProcessArg, PostProcessResult, PreProcessArg, PreProcessResult, TokenId,
};
use base64::{self, Engine as _};
use earley::{earley_grm_from_guidance, Parser};
use serde::{Deserialize, Serialize};

mod earley;
mod serialization;

pub struct Runner {
    toktrie: TokTrie,
    parser: Parser,
    all_tokens: Vec<TokenId>,
    is_ff: bool,
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
        println!("original: {:?}", grm);
        let grm = grm.optimize();
        println!("optimized: {:?}", grm);
        let cgrm = grm.compile();
        let parser = Parser::new(cgrm);
        Runner {
            toktrie: TokTrie::from_host(),
            parser,
            all_tokens: Vec::new(),
            is_ff: false,
        }
    }
}

impl AiciCtrl for Runner {
    fn pre_process(&mut self, _arg: PreProcessArg) -> PreProcessResult {
        PreProcessResult::continue_()
    }

    fn mid_process(&mut self, _arg: MidProcessArg) -> MidProcessResult {
        let bytes = self.parser.force_bytes();
        if bytes.len() > 0 {
            let mut tokens = tokenize_bytes(&bytes);
            let mut suff = Vec::new();
            let mut chop_tokens = 0;
            let mut chop_bytes = 0;
            for (idx, t) in tokens.iter().rev().enumerate() {
                suff.splice(0..0, self.toktrie.token(*t).iter().cloned());
                if suff.len() > self.toktrie.max_token_len() {
                    break;
                }
                if self.toktrie.has_extensions(&suff) {
                    chop_tokens = idx + 1;
                    chop_bytes = suff.len();
                }
            }
            tokens.truncate(tokens.len() - chop_tokens);
            self.parser.pop_rows(chop_bytes);
            if tokens.len() > 0 {
                let fixed_tokens = {
                    let all_tokens = self
                        .all_tokens
                        .iter()
                        .chain(tokens.iter())
                        .cloned()
                        .collect::<Vec<_>>();
                    let all_bytes = self.toktrie.decode(&all_tokens);
                    tokenize_bytes(&all_bytes)
                };
                for idx in 0..=self.all_tokens.len() {
                    if idx == self.all_tokens.len() || self.all_tokens[idx] != fixed_tokens[idx] {
                        let backtrack = (self.all_tokens.len() - idx) as u32;
                        let ff_tokens = fixed_tokens[idx..].to_vec();
                        println!(
                            "backtrack: {}, ff_tokens: {}",
                            backtrack,
                            self.toktrie.tokens_dbg(&ff_tokens)
                        );
                        self.all_tokens = fixed_tokens;
                        self.is_ff = true;
                        return MidProcessResult::Splice {
                            backtrack,
                            ff_tokens,
                        };
                    }
                }
                panic!("unreachable");
            } else {
                assert!(chop_bytes == bytes.len());
            }
        }

        self.is_ff = false;

        let mut set = self.toktrie.alloc_token_set();
        self.toktrie.compute_bias(&mut self.parser, &mut set);
        println!("bias: {}", self.toktrie.token_set_dbg(&set));

        MidProcessResult::SampleWithBias {
            allowed_tokens: set,
        }
    }

    fn post_process(&mut self, arg: PostProcessArg) -> PostProcessResult {
        println!(
            "post tokens:{} {}",
            if self.is_ff { " ff" } else { "" },
            self.toktrie.tokens_dbg(&arg.tokens)
        );
        if !self.is_ff {
            self.all_tokens.extend(&arg.tokens);
            self.toktrie.append_tokens(&mut self.parser, &arg.tokens);
        }
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
