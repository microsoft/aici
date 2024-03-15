use aici_abi::{
    arg_bytes, tokenize, tokenize_bytes,
    toktree::{Recognizer, TokTrie},
    AiciCtrl, MidProcessArg, MidProcessResult, PostProcessArg, PostProcessResult, PreProcessArg,
    PreProcessResult,
};
use base64::{self, Engine as _};
use earley::{earley_grm_from_guidance, Parser};
use serde::{Deserialize, Serialize};

mod earley;
mod serialization;

pub struct Runner {
    toktrie: TokTrie,
    parser: Parser,
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
        let grm = grm.optimize();
        println!("optimized: {:?}", grm);
        let cgrm = grm.compile();
        let parser = Parser::new(cgrm);
        Runner {
            toktrie: TokTrie::from_host(),
            parser,
        }
    }
}

impl AiciCtrl for Runner {
    fn pre_process(&mut self, _arg: PreProcessArg) -> PreProcessResult {
        let bytes = self.parser.force_bytes();
        if bytes.len() > 0 {
            let mut tokens = tokenize_bytes(&bytes);
            let mut suff = Vec::new();
            let mut chop_tokens = 0;
            for (idx, t) in tokens.iter().rev().enumerate() {
                suff.splice(0..0, self.toktrie.token(*t).iter().cloned());
                if suff.len() > self.toktrie.max_token_len() {
                    break;
                }
                if self.toktrie.has_extensions(&suff) {
                    chop_tokens = idx + 1;
                }
            }
            tokens.truncate(tokens.len() - chop_tokens);
            self.parser.pop_rows(bytes.len());
            if tokens.len() > 0 {
                println!("ff_tokens: {:?}", self.toktrie.decode_str(&tokens));
                return PreProcessResult::ff_tokens(tokens);
            }
        }

        PreProcessResult::continue_()
    }

    fn mid_process(&mut self, _arg: MidProcessArg) -> MidProcessResult {
        let mut set = self.toktrie.alloc_token_set();
        self.toktrie.compute_bias(&mut self.parser, &mut set);
        println!("bias: {}", self.toktrie.token_set_dbg(&set));

        MidProcessResult::SampleWithBias {
            allowed_tokens: set,
        }
    }

    fn post_process(&mut self, arg: PostProcessArg) -> PostProcessResult {
        println!("post tokens: {:?}", self.toktrie.decode_str(&arg.tokens));
        self.toktrie.append_tokens(&mut self.parser, &arg.tokens);
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
