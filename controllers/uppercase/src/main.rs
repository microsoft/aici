use aici_abi::{
    host_trie,
    recognizer::{FunctionalRecognizer, StackRecognizer},
    tokenize,
    toktree::{SpecialToken, TokTrie},
    AiciCtrl, InitPromptArg, InitPromptResult, MidProcessArg, MidProcessResult,
};

// This constraints enforces an upper case letter every 4th byte
// The state is the position in the output stream
struct QuadUpper {}
impl FunctionalRecognizer<usize> for QuadUpper {
    fn initial(&self) -> usize {
        0
    }

    fn try_append(&self, state: usize, byte: u8) -> Option<usize> {
        if state % 4 == 0 && !byte.is_ascii_uppercase() {
            None
        } else {
            Some(state + 1)
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
    recognizer: StackRecognizer<usize, QuadUpper>,
}

impl Runner {
    pub fn new() -> Self {
        Runner {
            toktrie: host_trie(),
            tokens: Vec::new(),
            recognizer: StackRecognizer::from(QuadUpper {}),
        }
    }
}

impl AiciCtrl for Runner {
    fn init_prompt(&mut self, arg: InitPromptArg) -> InitPromptResult {
        if arg.prompt.len() <= 1 {
            // in case no prompt was provided, invent some
            InitPromptResult {
                prompt: tokenize("Here's a tweet:\n"),
            }
        } else {
            InitPromptResult::from_arg(arg)
        }
    }

    fn mid_process(&mut self, arg: MidProcessArg) -> MidProcessResult {
        // store our tokens
        arg.save_tokens(&mut self.tokens);
        // and update the state of our recognizer
        self.toktrie
            .append_tokens(&mut self.recognizer, &arg.tokens)
            .unwrap();

        // stop after 50 tokens
        if self.tokens.len() > 50 || arg.has_eos() {
            return MidProcessResult::stop();
        }

        // otherwise, compute bias according to our recognizer
        let mut set = self.toktrie.alloc_token_set();
        self.toktrie.compute_bias(&mut self.recognizer, &mut set);
        MidProcessResult::sample(set)
    }
}

fn main() {
    // test code here?
}

aici_abi::aici_expose_all!(Runner, Runner::new());
