use aici_abi::{
    arg_bytes, AiciCtrl, InitPromptArg, InitPromptResult, MidProcessArg, MidProcessResult,
};
use serde::{Deserialize, Serialize};

use aici_llguidance_ctrl::{api::TopLevelGrammar, output::Reporter, TokenParser};

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

impl Runner {
    pub fn new() -> Self {
        infoln!("building runner...");
        let arg: RunnerArg = serde_json::from_slice(&arg_bytes()).expect("invalid JSON arg");
        let tok_parser = TokenParser::from_llguidance_json(
            Box::new(aici_abi::WasmTokenizerEnv::default()),
            arg.grammar,
            2,
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
        let r = self.tok_parser.mid_process(arg);
        for v in self.reporter.get_progress(&mut self.tok_parser, &r) {
            json_out(&v);
        }
        r
    }
}

fn main() {}

aici_abi::aici_expose_all!(Runner, Runner::new());
