use std::sync::Arc;

use crate::{
    api::{GenGrammarOptions, TopLevelGrammar},
    earley::{grammars_from_json, CGrammar, CSymIdx, ModelVariable, Parser, ParserStats},
};
use aici_abi::{svob::SimpleVob, MidProcessArg, MidProcessResult, TokenId, TokenizerEnv};
use anyhow::Result;
use serde_json::json;

macro_rules! infoln {
    ($s:expr, $($arg:tt)*) => {
        if $s.log_level >= 2 {
            eprintln!($($arg)*);
        }
    };
}

macro_rules! warn {
    ($s:expr, $($arg:tt)*) => {
        if $s.log_level >= 1 {
            eprint!("Warning: ");
            eprintln!($($arg)*);
        }
    };
}

#[derive(Clone)]
pub struct TokenParser {
    pub token_env: Arc<dyn TokenizerEnv + Sync>,
    pub parser: Parser,
    pub log_level: isize,
    pub mid_process_start_time: std::time::Instant,
    // sampling any of these will pop the parser stack:
    pop_tokens: Option<SimpleVob>,
    test_trace: bool,
    parser_stack: Vec<ParserStackEntry>,
    parser_llm_tokens_offset: usize,
    // this is empty for top-level parser,
    // and the previous grm_bytes for sub-parsers
    previous_grm_bytes: Vec<u8>,
    mid_process_was_accepting: bool,

    max_tokens_total: usize,
    max_tokens_parser: usize,
    compiled_grammars: Vec<Arc<CGrammar>>,

    // tokens currently in KV cache
    llm_tokens: Vec<TokenId>,
    llm_bytes: Vec<u8>,
    grm_prefix: Vec<u8>,
}

#[derive(Clone)]
struct ParserStackEntry {
    parser: Parser,
    parser_llm_tokens_offset: usize,
    previous_grm_bytes_len: usize,
    symidx: CSymIdx,
    max_tokens_offset: usize,
    mask: Option<SimpleVob>,
    is_accepting: bool,
}

impl TokenParser {
    pub fn from_llguidance_json(
        token_env: Arc<dyn TokenizerEnv + Sync>,
        buf: TopLevelGrammar,
        log_level: isize,
    ) -> Result<Self> {
        let mid_process_start_time = std::time::Instant::now();
        let test_trace = buf.test_trace;
        let max_tokens = buf.max_tokens.unwrap_or(usize::MAX);
        let compiled_grammars = grammars_from_json(buf, log_level >= 2)?;
        let parser = Parser::new(
            Arc::clone(&compiled_grammars[0]),
            GenGrammarOptions::default(),
        )?;

        Ok(TokenParser {
            log_level,
            test_trace,
            token_env,
            mid_process_start_time,
            mid_process_was_accepting: false,
            pop_tokens: None,
            parser,
            parser_llm_tokens_offset: 0,
            parser_stack: Vec::new(),
            previous_grm_bytes: Vec::new(),
            compiled_grammars,
            llm_tokens: Vec::new(),
            llm_bytes: Vec::new(),
            grm_prefix: Vec::new(),
            max_tokens_total: max_tokens,
            max_tokens_parser: max_tokens,
        })
    }

    pub fn parser_stats(&self) -> &ParserStats {
        &self.parser.stats
    }

    pub fn num_tokens(&self) -> usize {
        self.llm_tokens.len()
    }

    pub fn final_bytes(&self) -> &[u8] {
        &self.llm_bytes[self.grm_prefix.len()..]
    }

    pub fn mid_process_was_accepting(&self) -> bool {
        self.mid_process_was_accepting
    }

    pub fn bytes_since(&mut self, mut idx: usize) -> &[u8] {
        idx += self.grm_prefix.len();
        let endp = std::cmp::min(
            self.llm_bytes.len(),
            self.previous_grm_bytes
                .len()
                .saturating_add(self.parser.hidden_start()),
        );
        if idx >= self.llm_bytes.len() || idx >= endp {
            return &[];
        }
        &self.llm_bytes[idx..endp]
    }

    pub fn process_prompt(&mut self, prompt: Vec<TokenId>) -> Vec<TokenId> {
        assert!(self.llm_tokens.is_empty());

        let trie = self.token_env.tok_trie();
        infoln!(self, "prompt: {}", trie.tokens_dbg(&prompt));
        let mut prompt_bytes = trie.decode(&prompt);
        self.parser.force_bytes();
        let grm_bytes = self.parser.get_bytes();
        prompt_bytes.extend_from_slice(&grm_bytes);
        let tokens = self.token_env.tokenize_bytes(&prompt_bytes);
        infoln!(self, "prompt+grm: {}", trie.tokens_dbg(&tokens));
        let (chop_tokens, chop_bytes) = trie.chop_tokens(&mut self.parser, &tokens);
        let res_prompt = tokens[..tokens.len() - chop_tokens].to_vec();

        // if we moved a bunch of grammar to the prompt, update llm_tokens to reflect that
        if chop_bytes <= grm_bytes.len() {
            self.llm_bytes = grm_bytes[0..grm_bytes.len() - chop_bytes].to_vec();
            self.llm_tokens = self.token_env.tokenize_bytes(&self.llm_bytes);
            let decoded = self.token_env.tok_trie().decode(&self.llm_tokens);
            if self.llm_bytes.len() > 0 && &decoded[1..] == &self.llm_bytes && decoded[0] == b' ' {
                infoln!(self, "applying <s>space hack");
                self.grm_prefix = decoded[0..1].to_vec();
                self.llm_bytes = decoded;
            }
            infoln!(self, "ini_tokens: {}", trie.tokens_dbg(&self.llm_tokens));
        } else {
            // pretend the final bit of prompt was the prefix of the grammar
            self.grm_prefix = prompt_bytes
                [prompt_bytes.len() - chop_bytes..prompt_bytes.len() - grm_bytes.len()]
                .to_vec();
            infoln!(
                self,
                "force_prefix: {:?}",
                String::from_utf8_lossy(&self.grm_prefix)
            );
        }

        infoln!(self, "res_prompt: {}", trie.tokens_dbg(&res_prompt));
        if self.test_trace {
            self.test_trace_json(&json!({
                "prompt": trie.test_trace_tokens(&prompt),
                "res_prompt": trie.test_trace_tokens(&res_prompt),
            }));
        }
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

    fn test_trace_json(&self, j: &serde_json::Value) {
        if self.test_trace {
            infoln!(self, "TEST: {}", serde_json::to_string(j).unwrap());
        }
    }

    pub fn mid_process(&mut self, arg: MidProcessArg) -> MidProcessResult {
        self.mid_process_start_time = std::time::Instant::now();
        if self.max_tokens_total == 0 {
            warn!(self, "max_tokens_total reached, stopping");
            return MidProcessResult::stop();
        }
        self.max_tokens_total -= 1;
        self.max_tokens_parser = self.max_tokens_parser.saturating_sub(1);

        let trace = if self.test_trace {
            let tokens = self.token_env.tok_trie().test_trace_tokens(&arg.tokens);
            Some(json!({
                "backtrack": arg.backtrack,
                "tokens": tokens,
            }))
        } else {
            None
        };

        infoln!(self, "\n");

        let r = self.mid_process_inner(arg);

        if self.test_trace {
            let res = if r.is_stop() {
                json!("stop")
            } else {
                let b = &r.branches[0];
                json!({
                    "sample_mask": b.sample_mask.is_some(),
                    "temperature": b.temperature,
                    "splices": b.splices.iter().map(|s| {
                        json!({
                            "when_sampled": s.when_sampled,
                            "backtrack": s.backtrack,
                            "tokens": self.token_env.tok_trie().test_trace_tokens(&s.ff_tokens),
                        })
                    }).collect::<Vec<_>>(),
                })
            };
            self.test_trace_json(&json!({
                "arg": trace.unwrap(),
                "res": res,
            }));
        }

        r
    }

    fn mid_process_inner(&mut self, mut arg: MidProcessArg) -> MidProcessResult {
        let start_time = std::time::Instant::now();

        self.mid_process_was_accepting = false;

        let trie = self.token_env.tok_trie();

        infoln!(
            self,
            "post tokens: bt={} {}",
            arg.backtrack,
            trie.tokens_dbg(&arg.tokens)
        );

        if arg.tokens.len() == 1 {
            if let Some(pop) = &self.pop_tokens {
                if pop.is_allowed(arg.tokens[0]) {
                    infoln!(self, "pop_tokens hit: {}", trie.token_set_dbg(pop));
                    let pentry = self.parser_stack.last().unwrap();
                    // if the top of parse stack allows this token, we should stop
                    // popping parsers in the next iteration - clear pop_tokens
                    if pentry.mask.as_ref().unwrap().is_allowed(arg.tokens[0]) {
                        self.pop_tokens = None;
                    }
                    self.pop_parser();
                    return self.mid_process_inner(arg);
                }
            }
        }
        self.pop_tokens = None;

        let mut has_eos = false;

        if arg.tokens.contains(&trie.eos_token()) {
            assert!(arg.tokens.len() == 1);
            if self.parser.scan_model_variable(ModelVariable::eos_token()) {
                // it got scanned correctly, so we remove it
                infoln!(self, "scanned eos_token");
                arg.tokens.clear();
            } else {
                infoln!(self, "didn't scan eos_token; saving");
                arg.save_tokens(&mut self.llm_tokens);
                has_eos = true;
            }
        } else {
            arg.save_tokens(&mut self.llm_tokens);
        }

        let new_bytes = trie.decode(&arg.tokens);
        self.llm_bytes.extend_from_slice(&new_bytes);

        // eprintln!(
        //     "llm_bytes: {:?}\nllm_tokens: {}\n{:?}",
        //     String::from_utf8_lossy(&self.llm_bytes),
        //     trie.tokens_dbg(&self.llm_tokens),
        //     self.llm_tokens
        // );

        // TODO maybe remove in future
        if self.llm_bytes != trie.decode(&self.llm_tokens) {
            panic!(
                "llm_bytes mismatch:\n    {:?}\n    {:?}",
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
            Ok(msg) => infoln!(self, "parser: {}", msg),
            Err(e) => {
                infoln!(self, "Parser Error: {}", e);
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
                        self,
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

        // if arg.tokens.contains(&trie.eos_token()) {
        //     return MidProcessResult::stop();
        // }

        let new_forced = grm_bytes[self.llm_bytes.len()..].to_vec();
        let mut token_prefix = Vec::new();

        if new_forced.len() > 0 || backtrack > 0 {
            let mut grm_tokens = self.token_env.tokenize_bytes(&new_forced);
            infoln!(
                self,
                "forced: {} bytes:{:?} tokens:{:?}",
                trie.tokens_dbg(&grm_tokens),
                new_forced,
                grm_tokens
            );
            let (chop_tokens, chop_bytes) = trie.chop_tokens(&mut self.parser, &grm_tokens);
            infoln!(self, "chop: {} tokens, {} bytes", chop_tokens, chop_bytes);
            token_prefix = new_forced[new_forced.len() - chop_bytes..].to_vec();
            // here we remove a suffix from grm_tokens that could be possibly tokenized differently
            grm_tokens.truncate(grm_tokens.len() - chop_tokens);

            if grm_tokens.len() > 0 || backtrack > 0 {
                infoln!(
                    self,
                    "fixed_tokens: {} bt={}",
                    trie.tokens_dbg(&grm_tokens),
                    backtrack
                );
                return MidProcessResult::splice(backtrack as u32, grm_tokens);
            } else {
                infoln!(self, "no fixed tokens");
            }
        }

        if token_prefix.is_empty() {
            if let Err(e) = self.maybe_push_parser() {
                warn!(self, "Error creating nested parser: {}", e);
                return MidProcessResult::stop();
            }
        }

        let (inner_done, inner_accepting) = {
            let empty_token_prefix = token_prefix.is_empty();
            let lexer_bytes = self.parser.has_pending_lexeme_bytes();
            let is_accepting = self.parser.is_accepting();
            let can_advance = self.parser.can_advance();
            let inner_done = empty_token_prefix && is_accepting && (!can_advance || has_eos);
            infoln!(
                self,
                "inner_done: {inner_done}; lexer_bytes: {lexer_bytes}; \
                can_advance: {can_advance} (eos:{has_eos}); \
                accept: {is_accepting}; \
                empty_token_prefix: {empty_token_prefix}"
            );
            let inner_accepting = is_accepting && empty_token_prefix;
            (inner_done, inner_accepting)
        };

        let trie = self.token_env.tok_trie();
        // self.parser.print_row(self.parser.num_rows() - 1);
        let mut set = self.parser.compute_bias(trie, &token_prefix);

        if inner_done || self.max_tokens_parser == 0 {
            if self.parser_stack.is_empty() {
                self.mid_process_was_accepting = inner_accepting;
                infoln!(
                    self,
                    "only eos token allowed, stopping; accepting: {}",
                    inner_accepting
                );
                return MidProcessResult::stop();
            } else {
                infoln!(self, "pop_parser; tokens left {}", self.max_tokens_parser);
                self.pop_parser();
                // re-start the whole process with a nice tail-recursion
                return self.mid_process_inner(if has_eos {
                    arg
                } else {
                    MidProcessArg {
                        backtrack: 0,
                        tokens: Vec::new(),
                        fork_group: Vec::new(),
                    }
                });
            }
        }

        if inner_accepting {
            let mut all_accepting = true;
            if self.parser_stack.len() > 0 {
                let mut pop_tokens = trie.alloc_token_set();
                for pentry in self.parser_stack.iter_mut() {
                    if pentry.mask.is_none() {
                        assert!(token_prefix.is_empty());
                        let (is_accepting, mask) = pentry
                            .parser
                            .compute_bias_after_gen_grammar(trie, pentry.symidx);
                        infoln!(self, "bias for upper parser: {}", trie.token_set_dbg(&mask));
                        pentry.mask = Some(mask);
                        pentry.is_accepting = is_accepting;
                    }
                    let m = pentry.mask.as_ref().unwrap();
                    pop_tokens.or_minus(m, &set);
                    set.or(m);
                    if !pentry.is_accepting {
                        all_accepting = false;
                        break;
                    }
                }
                infoln!(self, "pop_tokens: {}", trie.token_set_dbg(&pop_tokens));
                self.pop_tokens = Some(pop_tokens);
            }
            self.mid_process_was_accepting = all_accepting;
            if all_accepting {
                set.allow_token(trie.eos_token());
            }
        }

        infoln!(
            self,
            "bias: (pref: {:?}; accpt: {}) {:?} {}",
            String::from_utf8_lossy(&token_prefix),
            self.mid_process_was_accepting,
            start_time.elapsed(),
            self.token_env.tok_trie().token_set_dbg(&set)
        );

        if set.num_set() == 0 {
            infoln!(self, "no tokens allowed, stopping");
            return MidProcessResult::stop();
        }

        return MidProcessResult::sample_with_temp(set, Some(self.parser.temperature()));
    }

    fn maybe_push_parser(&mut self) -> Result<()> {
        if let Some((msg, symidx, gen_grammar)) = self.parser.maybe_gen_grammar() {
            if msg.len() > 0 {
                warn!(self, "{}", msg);
            }
            let grm = Arc::clone(&self.compiled_grammars[gen_grammar.grammar.0]);
            let max_tokens = self.parser.grammar().sym_data(symidx).props.max_tokens;
            let parser = Parser::new(grm, gen_grammar)?;
            let old_parser = std::mem::replace(&mut self.parser, parser);
            self.parser.stats = old_parser.stats.clone();
            let mut entry = ParserStackEntry {
                parser: old_parser,
                parser_llm_tokens_offset: self.parser_llm_tokens_offset,
                previous_grm_bytes_len: self.previous_grm_bytes.len(),
                symidx,
                max_tokens_offset: self.max_tokens_total.saturating_sub(self.max_tokens_parser),
                mask: None,
                is_accepting: false, // computed with mask
            };
            self.max_tokens_parser = std::cmp::min(self.max_tokens_parser, max_tokens);
            self.parser_llm_tokens_offset = self.llm_tokens.len();
            self.previous_grm_bytes
                .extend_from_slice(&entry.parser.get_bytes());
            self.parser_stack.push(entry);
        }
        Ok(())
    }

    fn pop_parser(&mut self) {
        let inner_bytes = self.parser.get_bytes();
        let entry = self.parser_stack.pop().unwrap();
        let stats = self.parser.stats.clone();
        self.parser = entry.parser;
        self.parser.stats = stats;
        self.parser_llm_tokens_offset = entry.parser_llm_tokens_offset;
        self.previous_grm_bytes
            .truncate(entry.previous_grm_bytes_len);
        infoln!(
            self,
            "pop_parser: {} tokens left; new {} - {} = {}",
            self.max_tokens_parser,
            self.max_tokens_total,
            entry.max_tokens_offset,
            self.max_tokens_total
                .saturating_sub(entry.max_tokens_offset)
        );
        self.max_tokens_parser = self
            .max_tokens_total
            .saturating_sub(entry.max_tokens_offset);
        self.parser.scan_gen_grammar(entry.symidx, inner_bytes);
    }
}
