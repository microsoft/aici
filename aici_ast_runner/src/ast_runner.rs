/*
- the tokenization algorithm is not simply the greedy longest prefix - it breaks string into "words", splits words
  into single-byte tokens and then merges adjacent pairs of tokens in order of token number, see
  https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py
- models are also trained on sub-optimal tokenizations, via "subword regularization": https://arxiv.org/abs/1804.10959
- in all tokenizers (gpt4, llama, phi, ...), all tokens fit one of these 3 categories:
  - only whitespace (not only ' ', but also '\n', '\t' etc)
  - start with a ' '
  - have no ' '
*/

mod cfg;
mod lex;
mod rx;

use std::{fmt::Debug, rc::Rc};

use cfg::CfgParser;
use rx::RxStackRecognizer;
use serde::{Deserialize, Serialize};

use crate::rx::RecRx;

use aici_abi::{
    aici_expose_all,
    host::{self, tokenize},
    svob::SimpleVob,
    toktree::{Recognizer, SpecialToken, TokTrie},
    wprintln, AiciVm, AiciVmHelper, TokenId,
};

// The JSON AST
#[derive(Serialize, Deserialize, Clone)]
pub enum Step {
    // Generate exactly the provided string
    Fixed {
        text: String,
    },
    // Generate exactly one of the provided strings
    Choose {
        options: Vec<String>,
    },
    // Generate text. It can be constrained with a regex or a yacc grammar.
    // The length can be constrained in several ways.
    Gen {
        rx: Option<String>,
        yacc: Option<String>,
        stop_at: Option<String>,
        max_tokens: Option<usize>,
        max_words: Option<usize>,
        max_bytes: Option<usize>,
    },
}

fn limit_len(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", String::from_utf8_lossy(&s.as_bytes()[0..max]))
    } else {
        s.to_string()
    }
}

impl Debug for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Step::Fixed { text } => write!(f, "Fixed({:?})", text),
            Step::Choose { options } => write!(f, "Choose({:?})", options),
            Step::Gen {
                rx,
                yacc,
                stop_at,
                max_tokens,
                max_words,
                max_bytes,
            } => {
                write!(f, "Gen(")?;
                if let Some(rx) = rx {
                    write!(f, "/{:?}/ ", rx)?;
                }
                if let Some(yacc) = yacc {
                    write!(f, "yacc:{:?} ", limit_len(yacc, 200))?;
                }
                if let Some(stop_at) = stop_at {
                    write!(f, "stop_at:{:?}, ", stop_at)?;
                }
                if let Some(max_tokens) = max_tokens {
                    write!(f, "max_tokens:{:?}, ", max_tokens)?;
                }
                if let Some(max_words) = max_words {
                    write!(f, "max_words:{:?}, ", max_words)?;
                }
                if let Some(max_bytes) = max_bytes {
                    write!(f, "max_bytes:{:?}, ", max_bytes)?;
                }
                write!(f, ")")
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Program {
    pub steps: Vec<Step>,
}

enum StepSpecific {
    Options { tokens: Vec<Vec<TokenId>> },
    Gen { rx: RxStackRecognizer },
    Cfg { cfg: CfgParser },
    Stop,
}
struct StepState {
    ast: Step,
    specific: StepSpecific,

    // stop conditions
    stop_at: Option<String>,
    max_tokens: usize,
    max_words: usize,
    max_bytes: usize,

    // state so far for this step
    tokens: Vec<TokenId>,
    bytes: Vec<u8>,
    word_idx: usize,
}
pub struct Runner {
    helper: AiciVmHelper,
    state_idx: usize,
    states: Vec<StepState>,
}

impl Debug for StepState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tok:{}/", self.tokens.len())?;
        if self.max_tokens > 10000 {
            write!(f, "inf")?;
        } else {
            write!(f, "{}", self.max_tokens)?;
        }
        if self.max_words < 10000 {
            write!(f, " word:{}/{}", self.word_idx, self.max_words)?;
        }
        if self.max_bytes < 10000 {
            write!(f, " byte:{}/{}", self.bytes.len(), self.max_bytes)?;
        }
        Ok(())
    }
}

fn has_token_at(t: TokenId, idx: usize) -> impl for<'a> Fn(&'a Vec<TokenId>) -> bool {
    move |v: &Vec<TokenId>| idx < v.len() && v[idx] == t
}

impl StepState {
    #[allow(dead_code)]
    fn pp(&self) -> String {
        format!("{:?}", self.ast)
    }

    fn from_specific(ast: &Step, specific: StepSpecific) -> StepState {
        StepState {
            ast: ast.clone(),
            specific,
            word_idx: 0,
            stop_at: None,
            tokens: Vec::new(),
            bytes: Vec::new(),
            max_words: usize::MAX,
            max_bytes: usize::MAX,
            max_tokens: usize::MAX,
        }
    }

    fn from_ast(s: &Step) -> StepState {
        match s {
            Step::Fixed { text } => Self::from_specific(
                s,
                StepSpecific::Options {
                    tokens: vec![tokenize(&text)],
                },
            ),

            Step::Choose { options } => Self::from_specific(
                s,
                StepSpecific::Options {
                    tokens: options.iter().map(|s| tokenize(s)).collect(),
                },
            ),

            Step::Gen {
                rx,
                yacc,
                stop_at,
                max_tokens,
                max_bytes,
                max_words,
            } => {
                let spec = match (yacc, rx) {
                    (Some(_), Some(_)) => {
                        panic!("can't have both yacc= and rx=")
                    }
                    (Some(yacc), None) => StepSpecific::Cfg {
                        cfg: CfgParser::from_yacc(yacc),
                    },
                    _ => {
                        let defl = "(.|\n)+".to_string();
                        let rx = rx.as_deref().unwrap_or(&defl);
                        StepSpecific::Gen {
                            rx: RecRx::from_rx(&rx).to_stack_recognizer(),
                        }
                    }
                };
                let mut r = Self::from_specific(s, spec);
                r.max_bytes = max_bytes.unwrap_or(usize::MAX);
                r.max_words = max_words.unwrap_or(usize::MAX);
                r.max_tokens = max_tokens.unwrap_or(usize::MAX);
                r.stop_at = stop_at.clone();
                r
            }
        }
    }

    fn check_eos(&mut self, optional: bool) -> bool {
        self.tokens.len() >= self.max_tokens
            || self.bytes.len() >= self.max_bytes
            || self.word_idx >= self.max_words
            || (self.stop_at.is_some() && self.stop_at.as_ref().unwrap().is_empty())
            || match &mut self.specific {
                StepSpecific::Stop => false,
                StepSpecific::Options { tokens } => {
                    if optional {
                        tokens.iter().any(|t| self.tokens.len() >= t.len())
                    } else {
                        tokens.iter().all(|t| self.tokens.len() >= t.len())
                    }
                }
                StepSpecific::Cfg { cfg } => {
                    cfg.special_allowed(SpecialToken::EndOfSentence)
                        && (optional || (0..=255).all(|byte| !cfg.byte_allowed(byte)))
                }
                StepSpecific::Gen { rx } => {
                    rx.special_allowed(SpecialToken::EndOfSentence)
                        && (optional || (0..=255).all(|byte| !rx.byte_allowed(byte)))
                }
            }
    }

    fn allows_eos(&mut self) -> bool {
        self.check_eos(true)
    }

    fn forces_eos(&mut self) -> bool {
        self.check_eos(false)
    }

    fn advance(&mut self, helper: &AiciVmHelper, token: TokenId) {
        self.tokens.push(token);

        let bytes = helper.trie.token(token);
        let sidx = self.bytes.len();
        self.bytes.extend_from_slice(bytes);
        for idx in sidx.saturating_sub(1)..self.bytes.len().saturating_sub(1) {
            if !is_boundry(self.bytes[idx]) && is_boundry(self.bytes[idx + 1]) {
                self.word_idx += 1;
                break;
            }
        }

        if let Some(stop) = &self.stop_at {
            let slen = stop.len();
            if slen > 0 {
                let pos = self.bytes[sidx.saturating_sub(slen)..]
                    .windows(stop.len())
                    .position(|w| w == stop.as_bytes());
                if pos.is_some() {
                    self.stop_at = Some("".to_string())
                }
            }
        }

        match &mut self.specific {
            StepSpecific::Stop => {}
            StepSpecific::Options { tokens } => {
                tokens.retain(has_token_at(token, self.tokens.len() - 1))
            }
            StepSpecific::Cfg { cfg } => helper.trie.append_token(cfg, token),
            StepSpecific::Gen { rx } => helper.trie.append_token(rx, token),
        }

        fn is_boundry(b: u8) -> bool {
            b == b' ' || b == b'\n' || b == b'\t'
        }
    }

    // the 'mut' on self is bogus - the state of the 'rx' doesn't change
    fn allows_token(&mut self, helper: &AiciVmHelper, token: TokenId) -> bool {
        if token == helper.trie.special_token(SpecialToken::EndOfSentence) {
            return self.allows_eos();
        }
        if self.forces_eos() {
            return false;
        }
        match &mut self.specific {
            StepSpecific::Stop => false,
            StepSpecific::Options { tokens } => {
                tokens.iter().any(has_token_at(token, self.tokens.len()))
            }
            StepSpecific::Cfg { cfg } => helper.trie.token_allowed(cfg, token),
            StepSpecific::Gen { rx } => helper.trie.token_allowed(rx, token),
        }
    }

    fn apply_to(&mut self, trie: Rc<Box<TokTrie>>, toks: &mut SimpleVob) {
        match &mut self.specific {
            StepSpecific::Stop => {
                toks.allow_token(trie.special_token(SpecialToken::EndOfSentence));
            }
            StepSpecific::Options { tokens } => {
                for v in tokens {
                    if self.tokens.len() < v.len() {
                        toks.allow_token(v[self.tokens.len()]);
                    }
                }
            }
            StepSpecific::Gen { rx } => {
                trie.add_bias(rx, toks);
            }
            StepSpecific::Cfg { cfg } => {
                trie.add_bias(cfg, toks);
            }
        }
    }
}

impl Runner {
    pub fn new(program: Program) -> Self {
        let mut states = program
            .steps
            .iter()
            .map(StepState::from_ast)
            .collect::<Vec<_>>();
        let stop_ast = Step::Fixed {
            text: "[STOP]".to_string(),
        };
        states.push(StepState::from_specific(&stop_ast, StepSpecific::Stop));

        for (idx, state) in states.iter().enumerate() {
            wprintln!("[{}] {} {:?}", idx, state.pp(), state);
        }

        Self {
            helper: AiciVmHelper::new(),
            state_idx: 0,
            states,
        }
    }

    fn stop(&mut self, info: &str) {
        self.state_idx = self.states.len() - 1;
        wprintln!("stop: {}", info)
    }

    fn advance(&mut self, token: TokenId) {
        let bytes = self.helper.trie.token(token);
        wprintln!(
            "advance {} '{}' [{}] {:?}",
            token,
            String::from_utf8_lossy(bytes),
            self.state_idx,
            self.states[self.state_idx]
        );

        // skip as many states as we can (that allow EOS), and find the last one that allows the token
        let mut last_idx = usize::MAX;
        for idx in self.state_idx..self.states.len() {
            if self.states[idx].allows_token(&self.helper, token) {
                last_idx = idx;
            }
            if !self.states[idx].allows_eos() {
                break;
            }
        }

        if last_idx == usize::MAX {
            self.stop("no state allows token");
            return;
        }

        if self.state_idx != last_idx {
            self.state_idx = last_idx;
        }

        self.states[last_idx].advance(&mut self.helper, token);
        wprintln!(" => [{}] {:?}", self.state_idx, self.states[self.state_idx]);
    }

    fn compute(&mut self) {
        self.helper.all_disallowed();
        for state in &mut self.states[self.state_idx..] {
            if state.forces_eos() {
                continue;
            }
            state.apply_to(self.helper.trie.clone(), &mut self.helper.allowed_tokens);
            if !state.allows_eos() {
                break;
            }
        }

        self.helper.compute_biases();
    }

    #[allow(dead_code)]
    fn print_prob(&self, tok: &str) {
        if let Some(id) = self.helper.trie.token_id(tok.as_bytes()) {
            wprintln!(
                "prob '{}' {} = {}",
                tok,
                id,
                self.helper.logit_biases[id as usize]
            );
        } else {
            wprintln!("prob '{}' -> no token", tok)
        }
    }
}

impl AiciVm for Runner {
    fn aici_process_prompt(&mut self) {
        wprintln!("prompt, {} tokens", self.helper.prompt_length);
        // ignore the prompt (for now)
        self.compute();
    }

    fn aici_append_token(&mut self, token: u32) {
        // save the token, just in case
        let toks = &mut self.helper.tokens;
        toks.push(token);

        self.advance(token);
        self.compute();
        // self.print_prob(" a");
        // self.print_prob(" about");
    }

    fn get_helper(&mut self) -> &mut AiciVmHelper {
        &mut self.helper
    }
}

fn main() {
    cfg::cfg_test().unwrap();
    //    let _run = sample_prog();
}

fn runner_from_env() -> Runner {
    let a = host::arg_bytes();
    let p: Program = serde_json::from_slice(&a).unwrap();
    Runner::new(p)
}

aici_expose_all!(Runner, runner_from_env());
