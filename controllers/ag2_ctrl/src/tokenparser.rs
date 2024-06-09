use std::rc::Rc;

use crate::{
    api::TopLevelGrammar,
    earley::{grammars_from_json, lexer::LexemeSpec, CGrammar, CSymIdx, ModelVariable, Parser},
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

macro_rules! warn {
    ($($arg:tt)*) => {
        print!("Warning: ");
        println!($($arg)*);
    };
}

pub struct TokenParser {
    pub token_env: Box<dyn TokenizerEnv>,
    pub parser: Parser,
    parser_stack: Vec<ParserStackEntry>,
    parser_llm_tokens_offset: usize,
    // this is empty for top-level parser,
    // and the previous grm_bytes for sub-parsers
    previous_grm_bytes: Vec<u8>,

    first_token_of_eos_marker: TokenId,
    max_tokens: usize,
    compiled_grammars: Vec<Rc<CGrammar>>,

    // tokens currently in KV cache
    llm_tokens: Vec<TokenId>,
    llm_bytes: Vec<u8>,
    grm_prefix: Vec<u8>,
}

struct ParserStackEntry {
    parser: Parser,
    parser_llm_tokens_offset: usize,
    previous_grm_bytes_len: usize,
    symidx: CSymIdx,
}

impl TokenParser {
    pub fn from_guidance_protobuf(
        token_env: Box<dyn TokenizerEnv>,
        buf: TopLevelGrammar,
    ) -> Result<Self> {
        let max_tokens = buf.max_tokens.unwrap_or(usize::MAX);
        let grm = grammars_from_json(buf)?;

        let compiled_grammars = grm
            .iter()
            .enumerate()
            .map(|(idx, g)| {
                infoln!("grammar #{}:\n{:?}", idx, g);
                let g = g.optimize();
                infoln!("optimized #{}:\n{:?}", idx, g);
                Rc::new(g.compile())
            })
            .collect::<Vec<_>>();

        let parser = Parser::new(Rc::clone(&compiled_grammars[0]))?;

        let first_token_of_eos_marker = token_env
            .tok_trie()
            .greedy_tokenize(&LexemeSpec::EOS_MARKER.as_bytes())[0];

        Ok(TokenParser {
            token_env,
            parser,
            parser_llm_tokens_offset: 0,
            parser_stack: Vec::new(),
            previous_grm_bytes: Vec::new(),
            compiled_grammars,
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

    pub fn bytes_since(&mut self, mut idx: usize) -> &[u8] {
        idx += self.grm_prefix.len();
        let endp = std::cmp::min(
            self.llm_bytes.len(),
            self.previous_grm_bytes.len() + self.parser.hidden_start(),
        );
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

    fn grm_bytes(&mut self) -> Vec<u8> {
        let mut grm_bytes = self.grm_prefix.clone();
        grm_bytes.extend_from_slice(&self.previous_grm_bytes);
        grm_bytes.extend_from_slice(&self.parser.get_bytes());
        grm_bytes
    }

    fn is_top_level_parser(&self) -> bool {
        self.parser_stack.is_empty()
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

        match self.parser.apply_tokens(
            trie,
            &self.llm_tokens[self.parser_llm_tokens_offset..],
            if self.is_top_level_parser() {
                self.grm_prefix.len()
            } else {
                0
            },
        ) {
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
        let trie = self.token_env.tok_trie(); // make borrow-checker happy

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
            infoln!(
                "forced: {} {:?} {:?}",
                trie.tokens_dbg(&grm_tokens),
                new_forced,
                grm_tokens
            );
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

        if token_prefix.is_empty() {
            if let Err(e) = self.maybe_gen_grammar() {
                warn!("Error creating parser: {}", e);
                return MidProcessResult::stop();
            }
        }

        let trie = self.token_env.tok_trie();
        let mut set = trie.alloc_token_set();
        // self.parser.print_row(self.parser.num_rows() - 1);
        trie.compute_bias_ext(&mut self.parser, &mut set, &token_prefix);

        // clean damage from EOS_MARKER
        if self.parser.lexer_allows_eos() {
            set.disallow_token(self.first_token_of_eos_marker);
        }

        if set.num_set() == 1 && set.is_allowed(trie.eos_token()) {
            if self.parser_stack.is_empty() {
                infoln!("only eos token allowed, stopping");
                return MidProcessResult::stop();
            } else {
                infoln!("pop_parser");
                self.pop_parser();
                // re-start the whole process
                return self.mid_process(MidProcessArg {
                    backtrack: 0,
                    tokens: Vec::new(),
                    fork_group: Vec::new(),
                });
            }
        }

        infoln!(
            "bias: (pref: {:?}) {:?} {}",
            String::from_utf8_lossy(&token_prefix),
            start_time.elapsed(),
            self.token_env.tok_trie().token_set_dbg(&set)
        );

        if set.num_set() == 0 {
            return MidProcessResult::stop();
        }

        return MidProcessResult::sample_with_temp(set, Some(self.parser.temperature()));
    }

    fn maybe_gen_grammar(&mut self) -> Result<()> {
        if let Some((msg, symidx, gen_grammar)) = self.parser.maybe_gen_grammar() {
            if msg.len() > 0 {
                warn!("{}", msg);
            }
            let parser = Parser::new(Rc::clone(&self.compiled_grammars[gen_grammar.grammar.0]))?;
            let old_parser = std::mem::replace(&mut self.parser, parser);
            let mut entry = ParserStackEntry {
                parser: old_parser,
                parser_llm_tokens_offset: self.parser_llm_tokens_offset,
                previous_grm_bytes_len: self.previous_grm_bytes.len(),
                symidx,
            };
            self.parser_llm_tokens_offset = self.llm_tokens.len();
            self.previous_grm_bytes
                .extend_from_slice(&entry.parser.get_bytes());
            self.parser_stack.push(entry);
        }
        Ok(())
    }

    fn pop_parser(&mut self) {
        let entry = self.parser_stack.pop().unwrap();
        self.parser = entry.parser;
        self.parser_llm_tokens_offset = entry.parser_llm_tokens_offset;
        self.previous_grm_bytes
            .truncate(entry.previous_grm_bytes_len);
        self.parser.scan_gen_grammar(entry.symidx);
    }
}
