use std::sync::Arc;

use aici_abi::{
    export, tokenizer,
    toktrie::{InferenceCapabilities, StepArg},
    AiciCtrl, ExportedProgram, Guest, InitPromptArg, InitPromptResult, MidProcessArg,
    MidProcessResult, Program, TokenizerEnv,
};
use serde::{Deserialize, Serialize};

use llguidance_parser::{api::TopLevelGrammar, output::Reporter, Logger, TokenParser};

const INFO: bool = true;

macro_rules! infoln {
    ($($arg:tt)*) => {
        if INFO {
            println!($($arg)*);
        }
    };
}

pub struct Runner {
    tok_parser: TokenParser,
    reporter: Reporter,
}

#[derive(Serialize, Deserialize)]
struct RunnerArg {
    grammar: TopLevelGrammar,
}

struct AmbientTokenEnv {
    toktrie: aici_abi::toktrie::TokTrie,
}

impl AmbientTokenEnv {
    fn new() -> Self {
        AmbientTokenEnv {
            toktrie: aici_abi::toktrie::TokTrie::from_bytes(
                &aici_abi::tokenizer::token_trie_bytes(),
            ),
        }
    }
}

impl TokenizerEnv for AmbientTokenEnv {
    fn stop(&self) -> ! {
        aici_abi::aici_stop()
    }

    fn tok_trie(&self) -> &aici_abi::toktrie::TokTrie {
        &self.toktrie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<aici_abi::toktrie::TokenId> {
        tokenizer::tokenize_bytes(s)
    }

    fn tokenize(&self, s: &str) -> Vec<aici_abi::toktrie::TokenId> {
        tokenizer::tokenize(s)
    }

    fn eos_token(&self) -> aici_abi::toktrie::TokenId {
        tokenizer::eos_token()
    }
}

impl Runner {
    pub fn new(arg: String) -> Self {
        infoln!("building runner...");
        let arg: RunnerArg = serde_json::from_str(&arg).expect("invalid JSON arg");
        let log_level = 2;
        let inf = InferenceCapabilities {
            backtrack: aici_abi::runtime::get_config("backtrack") != 0,
            ff_tokens: aici_abi::runtime::get_config("ff_tokens") != 0,
            conditional_ff_tokens: aici_abi::runtime::get_config("ff_tokens") != 0,
            fork: false,
        };
        let tok_parser = TokenParser::from_llguidance_json(
            Arc::new(AmbientTokenEnv::new()),
            arg.grammar,
            Logger::new(0, log_level),
            inf,
        )
        .expect("invalid guidance protobuf");

        let reporter = Reporter::new(&tok_parser);
        Runner {
            tok_parser,
            reporter,
        }
    }
}

fn json_out<T: Serialize>(obj: &T) {
    println!("JSON-OUT: {}", serde_json::to_string(obj).unwrap());
}

impl AiciCtrl for Runner {
    fn init_prompt(&mut self, arg: InitPromptArg) -> InitPromptResult {
        InitPromptResult {
            prompt: self.tok_parser.process_prompt(arg.prompt),
        }
    }
    fn mid_process(&mut self, arg: MidProcessArg) -> MidProcessResult {
        let r = self.tok_parser.mid_process(StepArg {
            backtrack: arg.backtrack,
            tokens: arg.tokens,
            sampled: arg.sampled,
        });
        for v in self.reporter.get_progress(&mut self.tok_parser, &r) {
            json_out(&v);
        }
        MidProcessResult::from_branch(r)
    }
}

fn main() {}

impl Program for Runner {
    fn new(arg: String) -> Self {
        Runner::new(arg)
    }
}

impl Guest for Runner {
    type Runner = ExportedProgram<Runner>;
}

export!(Runner);
