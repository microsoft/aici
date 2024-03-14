use aici_abi::{
    arg_string,
    recognizer::{FunctionalRecognizer, StackRecognizer},
    tokenize,
    toktree::{SpecialToken, TokTrie},
    AiciCtrl, MidProcessArg, MidProcessResult, PostProcessArg, PostProcessResult, PreProcessArg,
    PreProcessResult,
};

pub mod earley;
mod serialization;

// This constraints enforces an upper case letter every 4th byte
// The state is the position in the output stream
struct QuadUpper {}
impl FunctionalRecognizer<usize> for QuadUpper {
    fn initial(&self) -> usize {
        0
    }

    fn append(&self, state: usize, _byte: u8) -> usize {
        state + 1
    }

    fn byte_allowed(&self, state: usize, byte: u8) -> bool {
        if state % 4 == 0 {
            byte.is_ascii_uppercase()
        } else {
            true
        }
    }

    fn special_allowed(&self, _state: usize, tok: SpecialToken) -> bool {
        match tok {
            SpecialToken::EndOfSentence => false,
            _ => false,
        }
    }
}

pub struct Runner {
    toktrie: TokTrie,
    tokens: Vec<u32>,
    prompt: String,
    recognizer: StackRecognizer<usize, QuadUpper>,
}

impl Runner {
    pub fn new() -> Self {
        Runner {
            toktrie: TokTrie::from_host(),
            prompt: arg_string() + "\n",
            tokens: Vec::new(),
            recognizer: StackRecognizer::from(QuadUpper {}),
        }
    }
}

impl AiciCtrl for Runner {
    fn pre_process(&mut self, _arg: PreProcessArg) -> PreProcessResult {
        if self.tokens.is_empty() {
            // if no tokens yet, send our prompt
            let prompt = if self.prompt.len() <= 1 {
                "Here's a tweet:\n"
            } else {
                &self.prompt
            };
            let toks = tokenize(prompt);
            PreProcessResult::ff_tokens(toks)
        } else {
            // otherwise just continue - the other option is to suspend
            PreProcessResult::continue_()
        }
    }

    fn mid_process(&mut self, _arg: MidProcessArg) -> MidProcessResult {
        if self.tokens.len() > 50 {
            // stop after 50 tokens
            return MidProcessResult::Stop;
        }

        // otherwise, compute bias according to our recognizer
        let mut set = self.toktrie.alloc_token_set();
        self.toktrie.compute_bias(&mut self.recognizer, &mut set);
        MidProcessResult::SampleWithBias {
            allowed_tokens: set,
        }
    }

    fn post_process(&mut self, arg: PostProcessArg) -> PostProcessResult {
        // save our tokens
        self.tokens.extend_from_slice(&arg.tokens);
        // and update the state of our recognizer
        self.toktrie
            .append_tokens(&mut self.recognizer, &arg.tokens);
        // ::from_arg() will translate generation of EOS token into Stop instruction
        PostProcessResult::from_arg(&arg)
    }
}

fn main() {
    // test code here?
}

aici_abi::aici_expose_all!(Runner, Runner::new());
