use aici_abi::{
    arg_string, tokenize, toktree::TokTrie, AiciCtrl, MidProcessArg, MidProcessResult,
    PostProcessArg, PostProcessResult, PreProcessArg, PreProcessResult, TokenId,
};

pub struct Runner {
    toktrie: TokTrie,
    tokens: Vec<TokenId>,
    question: String,
    yes: TokenId,
    no: TokenId,
}

impl Runner {
    pub fn new() -> Self {
        let yes = tokenize("Yes")[0];
        let no = tokenize("No")[0];
        // ignore user-passed arg
        Runner {
            toktrie: TokTrie::from_host(),
            tokens: Vec::new(),
            question: arg_string() + "\n",
            yes,
            no,
        }
    }
}

impl AiciCtrl for Runner {
    fn pre_process(&mut self, _arg: PreProcessArg) -> PreProcessResult {
        if self.tokens.is_empty() {
            let toks = tokenize(&self.question);
            PreProcessResult::ff_tokens(toks)
        } else {
            PreProcessResult::continue_()
        }
    }

    fn mid_process(&mut self, _arg: MidProcessArg) -> MidProcessResult {
        let mut set = self.toktrie.alloc_token_set();
        set.allow_token(self.yes);
        set.allow_token(self.no);
        MidProcessResult::SampleWithBias {
            allowed_tokens: set,
        }
    }

    fn post_process(&mut self, arg: PostProcessArg) -> PostProcessResult {
        // save our tokens
        self.tokens.extend_from_slice(&arg.tokens);
        if self.tokens.len() >= 2 {
            PostProcessResult::stop()
        } else {
            PostProcessResult::from_arg(&arg)
        }
    }
}

fn main() {
    // test code here?
}

aici_abi::aici_expose_all!(Runner, Runner::new());
