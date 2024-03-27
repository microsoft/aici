use aici_abi::{tokenize, toktree::TokTrie, AiciCtrl, MidProcessArg, MidProcessResult, TokenId};

pub struct Runner {
    toktrie: TokTrie,
    tokens: Vec<TokenId>,
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
            yes,
            no,
        }
    }
}

impl AiciCtrl for Runner {
    fn mid_process(&mut self, arg: MidProcessArg) -> MidProcessResult {
        arg.save_tokens(&mut self.tokens);
        if self.tokens.len() >= 1 {
            // we only want the first token
            MidProcessResult::stop()
        } else {
            let mut set = self.toktrie.alloc_token_set();
            set.allow_token(self.yes);
            set.allow_token(self.no);
            MidProcessResult::sample(set)
        }
    }
}

fn main() {
    // test code here?
}

aici_abi::aici_expose_all!(Runner, Runner::new());
