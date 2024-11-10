use aici_abi::{
    export, tokenizer, toktrie::TokTrie, AiciCtrl, ExportedProgram, Guest, MidProcessArg,
    MidProcessResult, TokenId,
};

pub struct Runner {
    toktrie: TokTrie,
    tokens: Vec<TokenId>,
    yes: TokenId,
    no: TokenId,
}

impl aici_abi::Program for Runner {
    fn new(_arg: String) -> Self {
        let yes = tokenizer::tokenize("Yes")[0];
        let no = tokenizer::tokenize("No")[0];
        // ignore user-passed arg
        Runner {
            toktrie: TokTrie::from_bytes(&tokenizer::token_trie_bytes()),
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

impl Guest for Runner {
    type Runner = ExportedProgram<Runner>;
}

export!(Runner);
