use aici_abi::{
    export, exports, tokenizer, toktree::TokTrie, AiciCtrl, ExportedProgram, Guest, MidProcessArg,
    MidProcessResult, PostProcessArg, PostProcessResult, PreProcessResult,
    SampleWithBias, TokenId,
};

pub struct Runner {
    toktrie: TokTrie,
    tokens: Vec<TokenId>,
    question: Vec<TokenId>,
    yes: TokenId,
    no: TokenId,
}

impl aici_abi::Program for Runner {
    fn new(arg: String) -> Self {
        let yes = tokenizer::tokenize("Yes")[0];
        let no = tokenizer::tokenize("No")[0];
        // ignore user-passed arg
        Runner {
            toktrie: TokTrie::from_host(),
            tokens: Vec::new(),
            question: {
                let s: &str = &(arg + "\n");
                tokenizer::tokenize(s)
            },
            yes,
            no,
        }
    }
}

impl AiciCtrl for Runner {
    fn pre_process(&mut self) -> PreProcessResult {
        if self.tokens.is_empty() {
            PreProcessResult::ff_tokens(self.question.clone())
        } else {
            PreProcessResult::continue_()
        }
    }

    fn mid_process(&mut self, _arg: MidProcessArg) -> MidProcessResult {
        let mut set = self.toktrie.alloc_token_set();
        set.allow_token(self.yes);
        set.allow_token(self.no);
        MidProcessResult::SampleWithBias(SampleWithBias {
            allowed_tokens: set,
        })
    }

    fn post_process(&mut self, arg: PostProcessArg) -> PostProcessResult {
        // save our tokens
        self.tokens.extend_from_slice(&arg.tokens);
        if self.tokens.len() >= self.question.len() + 1 {
            PostProcessResult::stop()
        } else {
            PostProcessResult::from_arg(&arg)
        }
    }
}

impl Guest for Runner {
    type Runner = ExportedProgram<Runner>;
}

export!(Runner);
