use crate::{
    api::TopLevelGrammar,
    earley::{grammars_from_json, lexer::LexemeSpec, ModelVariable, Parser},
};
use aici_abi::{MidProcessArg, MidProcessResult, TokenId, TokenizerEnv};
use anyhow::Result;

const INFO: bool = true;

macro_rules! infoln {
    ($($arg:tt)*) => {
        if INFO {
            println!($($arg)*);
        }
    };
}

pub struct TokenParser {
    pub token_env: Box<dyn TokenizerEnv>,
    pub parser: Parser,
    first_token_of_eos_marker: TokenId,
    // tokens currently in KV cache
    llm_tokens: Vec<TokenId>,
    llm_bytes: Vec<u8>,
    grm_prefix: Vec<u8>,
    max_tokens: usize,
}

impl TokenParser {
    pub fn from_guidance_protobuf(
        token_env: Box<dyn TokenizerEnv>,
        buf: TopLevelGrammar,
    ) -> Result<Self> {
        let max_tokens = buf.max_tokens.unwrap_or(usize::MAX);
        let grm = grammars_from_json(buf)?;
        infoln!("original: {:?}", grm);
        let grm = grm[0].optimize();
        infoln!("optimized: {:?}", grm);
        let cgrm = grm.compile();
        let parser = Parser::new(cgrm)?;

        let first_token_of_eos_marker = token_env
            .tok_trie()
            .greedy_tokenize(&LexemeSpec::EOS_MARKER.as_bytes())[0];

        Ok(TokenParser {
            token_env,
            parser,
            first_token_of_eos_marker,
            llm_tokens: Vec::new(),
            llm_bytes: Vec::new(),
            grm_prefix: Vec::new(),
            max_tokens,
        })
    }

    pub fn num_tokens(&self) -> usize {
        self.llm_tokens.len()
    }

    pub fn final_bytes(&self) -> &[u8] {
        &self.llm_bytes[self.grm_prefix.len()..]
    }

    pub fn bytes_since(&self, mut idx: usize) -> &[u8] {
        idx += self.grm_prefix.len();
        let endp = std::cmp::min(self.llm_bytes.len(), self.parser.hidden_start());
        if idx >= self.llm_bytes.len() || idx >= endp {
            return &[];
        }
        &self.llm_bytes[idx..endp]
    }

    pub fn process_prompt(&mut self, prompt: Vec<TokenId>) -> Vec<TokenId> {
        assert!(self.llm_tokens.is_empty());

        let trie = self.token_env.tok_trie();
        infoln!("prompt: {}", trie.tokens_dbg(&prompt));
        let mut prompt_bytes = trie.decode(&prompt);
        self.parser.force_bytes();
        let grm_bytes = self.parser.get_bytes();
        prompt_bytes.extend_from_slice(&grm_bytes);
        let tokens = self.token_env.tokenize_bytes(&prompt_bytes);
        infoln!("prompt+grm: {}", trie.tokens_dbg(&tokens));
        let (chop_tokens, chop_bytes) = trie.chop_tokens(&mut self.parser, &tokens);
        let res_prompt = tokens[..tokens.len() - chop_tokens].to_vec();

        // if we moved a bunch of grammar to the prompt, update llm_tokens to reflect that
        if chop_bytes <= grm_bytes.len() {
            self.llm_bytes = grm_bytes[0..grm_bytes.len() - chop_bytes].to_vec();
            self.llm_tokens = self.token_env.tokenize_bytes(&self.llm_bytes);
            infoln!("ini_tokens: {}", trie.tokens_dbg(&self.llm_tokens));
        } else {
            // pretend the final bit of prompt was the prefix of the grammar
            self.grm_prefix = prompt_bytes
                [prompt_bytes.len() - chop_bytes..prompt_bytes.len() - grm_bytes.len()]
                .to_vec();
            infoln!(
                "force_prefix: {:?}",
                String::from_utf8_lossy(&self.grm_prefix)
            );
        }

        infoln!("res_prompt: {}", trie.tokens_dbg(&res_prompt));
        res_prompt
    }

    fn grm_bytes(&self) -> Vec<u8> {
        let mut grm_bytes = self.grm_prefix.clone();
        grm_bytes.extend_from_slice(&self.parser.get_bytes());
        grm_bytes
    }

    pub fn mid_process(&mut self, mut arg: MidProcessArg) -> MidProcessResult {
        if self.max_tokens == 0 {
            infoln!("max_tokens=0, stopping");
            return MidProcessResult::stop();
        }
        self.max_tokens -= 1;

        let start_time = std::time::Instant::now();

        infoln!("\n");
        let trie = self.token_env.tok_trie();

        infoln!(
            "post tokens: bt={} {}",
            arg.backtrack,
            trie.tokens_dbg(&arg.tokens)
        );

        if arg.tokens.contains(&trie.eos_token()) {
            assert!(arg.tokens.len() == 1);
            if self.parser.scan_model_variable(ModelVariable::eos_token()) {
                // it got scanned correctly, so we remove it
                arg.tokens.clear();
            }
        } else {
            arg.save_tokens(&mut self.llm_tokens);
        }

        let new_bytes = trie.decode(&arg.tokens);
        self.llm_bytes.extend_from_slice(&new_bytes);

        // TODO maybe remove in future
        if self.llm_bytes != trie.decode(&self.llm_tokens) {
            panic!(
                "llm_bytes mismatch: {:?} {:?}",
                String::from_utf8_lossy(&self.llm_bytes),
                String::from_utf8_lossy(&trie.decode(&self.llm_tokens))
            );
        }

        match self
            .parser
            .apply_tokens(trie, &self.llm_tokens, self.grm_prefix.len())
        {
            Ok("") => {}
            Ok(msg) => infoln!("parser: {}", msg),
            Err(e) => {
                infoln!("Parser Error: {}", e);
                self.token_env.stop();
            }
        };

        self.parser.filter_max_tokens();

        // force after scanning tokens from LLM (this may walk the parser some more)
        self.parser.force_bytes();
        let grm_bytes = self.grm_bytes();

        let mut backtrack = 0;

        // println!(
        //     "\nllm_bytes: {:?}\ngrm_bytes: {:?}\n",
        //     String::from_utf8_lossy(&self.llm_bytes),
        //     String::from_utf8_lossy(&grm_bytes),
        // );

        // now, see if we need to backtrack
        if self.llm_bytes.len() > grm_bytes.len()
            || self.llm_bytes != grm_bytes[0..self.llm_bytes.len()]
        {
            let mut ptr = 0;
            for (idx, t) in self.llm_tokens.iter().enumerate() {
                let b = trie.token(*t);
                let pend = ptr + b.len();
                if pend > grm_bytes.len() || b != &grm_bytes[ptr..pend] {
                    backtrack = self.llm_tokens.len() - idx;
                    infoln!(
                        "backtrack: {} (deletes: {:?})",
                        backtrack,
                        String::from_utf8_lossy(&self.llm_bytes[ptr..])
                    );
                    assert!(backtrack > 0);
                    self.llm_bytes.drain(ptr..);
                    break;
                }
                ptr = pend;
            }
            if backtrack == 0 {
                panic!(
                    "backtrack failed {:?} {:?}",
                    String::from_utf8_lossy(&self.llm_bytes),
                    String::from_utf8_lossy(&grm_bytes)
                );
            }
        }

        if arg.tokens.contains(&trie.eos_token()) {
            return MidProcessResult::stop();
        }

        let new_forced = grm_bytes[self.llm_bytes.len()..].to_vec();
        let mut token_prefix = Vec::new();

        if new_forced.len() > 0 || backtrack > 0 {
            let mut grm_tokens = self.token_env.tokenize_bytes(&new_forced);
            infoln!("forced: {} {:?} {:?}", trie.tokens_dbg(&grm_tokens), new_forced, grm_tokens);
            let (chop_tokens, chop_bytes) = trie.chop_tokens(&mut self.parser, &grm_tokens);
            token_prefix = new_forced[new_forced.len() - chop_bytes..].to_vec();
            // here we remove a suffix from grm_tokens that could be possibly tokenized differently
            grm_tokens.truncate(grm_tokens.len() - chop_tokens);

            if grm_tokens.len() > 0 || backtrack > 0 {
                infoln!(
                    "fixed_tokens: {} bt={}",
                    trie.tokens_dbg(&grm_tokens),
                    backtrack
                );
                return MidProcessResult::splice(backtrack as u32, grm_tokens);
            } else {
                infoln!("no fixed tokens");
            }
        }

        let mut set = trie.alloc_token_set();
        // self.parser.print_row(self.parser.num_rows() - 1);
        trie.compute_bias_ext(&mut self.parser, &mut set, &token_prefix);

        // clean damage from EOS_MARKER
        if self.parser.lexer_allows_eos() {
            set.disallow_token(self.first_token_of_eos_marker);
        }

        if set.num_set() == 1 && set.is_allowed(trie.eos_token()) {
            infoln!("only eos token allowed, stopping");
            return MidProcessResult::stop();
        }

        infoln!(
            "bias: (pref: {:?}) {:?} {}",
            String::from_utf8_lossy(&token_prefix),
            start_time.elapsed(),
            trie.token_set_dbg(&set)
        );

        if set.num_set() == 0 {
            return MidProcessResult::stop();
        }

        return MidProcessResult::sample_with_temp(set, Some(self.parser.temperature()));
    }
}
