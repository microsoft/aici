use aici_abi::{
    arg_bytes, toktree::TokTrie, AiciCtrl, MidProcessArg, MidProcessResult, PostProcessArg,
    PostProcessResult, PreProcessArg, PreProcessResult,
};
use base64::{self, Engine as _};
use earley::{earley_grm_from_guidance, Parser};
use serde::{Deserialize, Serialize};

mod earley;
mod serialization;

pub struct Runner {
    toktrie: TokTrie,
    tokens: Vec<u32>,
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
        let cgrm = grm.compile();
        let parser = Parser::new(cgrm);
        Runner {
            toktrie: TokTrie::from_host(),
            tokens: Vec::new(),
            parser,
        }
    }
}

impl AiciCtrl for Runner {
    fn pre_process(&mut self, _arg: PreProcessArg) -> PreProcessResult {
        PreProcessResult::continue_()
    }

    fn mid_process(&mut self, _arg: MidProcessArg) -> MidProcessResult {
        let mut set = self.toktrie.alloc_token_set();
        self.toktrie.compute_bias(&mut self.parser, &mut set);
        MidProcessResult::SampleWithBias {
            allowed_tokens: set,
        }
    }

    fn post_process(&mut self, arg: PostProcessArg) -> PostProcessResult {
        self.tokens.extend_from_slice(&arg.tokens);
        self.toktrie.append_tokens(&mut self.parser, &arg.tokens);
        // ::from_arg() will translate generation of EOS token into Stop instruction
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
