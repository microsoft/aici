use aici_abi::{
    arg_bytes, bytes::to_hex_string, AiciCtrl, InitPromptArg, InitPromptResult, MidProcessArg,
    MidProcessResult,
};
use serde::{Deserialize, Serialize};

use ag2_ctrl::{grammar::TopLevelGrammar, TokenParser};

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
    text_ptr: usize,
    token_ptr: usize,
}

#[derive(Serialize, Deserialize)]
struct RunnerArg {
    grammar: TopLevelGrammar,
}

impl Runner {
    pub fn new() -> Self {
        infoln!("building runner...");
        let arg: RunnerArg = serde_json::from_slice(&arg_bytes()).expect("invalid JSON arg");
        let tok_parser = TokenParser::from_guidance_protobuf(
            Box::new(aici_abi::WasmTokenizerEnv::default()),
            arg.grammar,
        )
        .expect("invalid guidance protobuf");
        let token_ptr = tok_parser.num_tokens();
        Runner {
            tok_parser,
            reported_captures: 0,
            text_ptr: 0,
            token_ptr,
        }
    }

    fn report_captures(&mut self) {
        // first report newly generated text
        let new_text = self.tok_parser.bytes_since(self.text_ptr);
        if new_text.len() > 0 {
            // TODO log_prob
            let text =
                Text::from_bytes(new_text, 0.0, self.tok_parser.num_tokens() - self.token_ptr);
            json_out(&text);
            self.text_ptr += new_text.len();
            self.token_ptr = self.tok_parser.num_tokens();
        }

        // then the captures
        let captures = &self.tok_parser.parser.captures()[self.reported_captures..];
        self.reported_captures += captures.len();

        // remove duplicate names
        let mut seen = std::collections::HashSet::new();
        let captures = captures
            .iter()
            .rev()
            .filter(|(name, _)| seen.insert(name))
            .collect::<Vec<_>>();
        for (name, val) in captures.iter().rev() {
            let cap = Capture {
                object: "capture",
                name: name.clone(),
                str: String::from_utf8_lossy(val).to_string(),
                hex: to_hex_string(val),
                log_prob: 0.0, // TODO
            };
            json_out(&cap);
        }
    }
}

fn json_out<T: Serialize>(obj: &T) {
    println!("JSON-OUT: {}", serde_json::to_string(obj).unwrap());
}

#[derive(Serialize, Deserialize)]
struct Capture {
    object: &'static str, // "capture"
    name: String,
    str: String,
    hex: String,
    log_prob: f64,
}

#[derive(Serialize, Deserialize)]
struct FinalText {
    object: &'static str, // "final_text"
    str: String,
    hex: String,
}

#[derive(Serialize, Deserialize)]
struct Text {
    object: &'static str, // "text"
    str: String,
    hex: String,
    log_prob: f64,
    num_tokens: usize,
}

impl Text {
    pub fn from_bytes(bytes: &[u8], log_prob: f64, num_tokens: usize) -> Self {
        Text {
            object: "text",
            str: String::from_utf8_lossy(bytes).to_string(),
            hex: to_hex_string(bytes),
            log_prob,
            num_tokens,
        }
    }
}

impl FinalText {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        FinalText {
            object: "final_text",
            str: String::from_utf8_lossy(bytes).to_string(),
            hex: to_hex_string(bytes),
        }
    }
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
        if r.is_stop() {
            let final_text = FinalText::from_bytes(self.tok_parser.final_bytes());
            json_out(&final_text);
        }
        r
    }
}

fn main() {}

aici_abi::aici_expose_all!(Runner, Runner::new());
