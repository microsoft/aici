use aici_abi::{
    arg_bytes, bytes::to_hex_string, AiciCtrl, InitPromptArg, InitPromptResult, MidProcessArg,
    MidProcessResult,
};
use base64::{self, Engine as _};
use serde::{Deserialize, Serialize};

use aici_guidance_ctrl::TokenParser;

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
    reported_captures: usize,
}

#[derive(Serialize, Deserialize)]
struct RunnerArg {
    guidance_b64: String,
}

impl Runner {
    pub fn new() -> Self {
        infoln!("building runner...");
        let arg: RunnerArg = serde_json::from_slice(&arg_bytes()).expect("invalid JSON arg");
        let guidance = base64::engine::general_purpose::STANDARD
            .decode(arg.guidance_b64)
            .expect("invalid base64");
        Runner {
            tok_parser: TokenParser::from_guidance_protobuf(
                Box::new(aici_abi::WasmTokenizerEnv::default()),
                &guidance,
            )
            .expect("invalid guidance protobuf"),
            reported_captures: 0,
        }
    }

    fn report_captures(&mut self) {
        let captures = &self.tok_parser.parser.captures()[self.reported_captures..];
        for (name, val) in captures {
            self.reported_captures += 1;
            let cap = Capture {
                object: "capture",
                name: name.clone(),
                str: String::from_utf8_lossy(val).to_string(),
                hex: to_hex_string(val),
            };
            println!("JSON-OUT: {}", serde_json::to_string(&cap).unwrap());
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Capture {
    object: &'static str, // "capture"
    name: String,
    str: String,
    hex: String,
}

impl AiciCtrl for Runner {
    fn init_prompt(&mut self, arg: InitPromptArg) -> InitPromptResult {
        InitPromptResult {
            prompt: self.tok_parser.process_prompt(arg.prompt),
        }
    }
    fn mid_process(&mut self, arg: MidProcessArg) -> MidProcessResult {
        let r = self.tok_parser.mid_process(arg);
        self.report_captures();
        r
    }
}

fn main() {}

aici_abi::aici_expose_all!(Runner, Runner::new());
