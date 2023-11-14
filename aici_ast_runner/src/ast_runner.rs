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
    bytes::limit_str,
    host::{self, tokenize},
    svob::SimpleVob,
    toktree::{Recognizer, SpecialToken, TokTrie},
    wprintln, AiciVm, AiciVmHelper, InitPromptArg, PreProcessArg, PreProcessResult, ProcessArg,
    ProcessResult, TokenId,
};

// The JSON AST
#[derive(Serialize, Deserialize, Clone)]
pub enum Step {
    // Generate exactly the provided string
    Fixed {
        text: String,
        tag: Option<String>,
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
        mask_tags: Option<Vec<String>>,
    },

    /// Generate one sequence for each of the branches.
    Fork {
        branches: Vec<Vec<Step>>,
    },
}

impl Debug for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Step::Fork { branches } => {
                write!(f, "Fork {{")?;
                for branch in branches {
                    write!(f, "  branch:\n")?;
                    for step in branch {
                        write!(f, "      {:?}\n", step)?;
                    }
                }
                write!(f, "}}")
            }
            Step::Fixed { text, tag } => {
                write!(f, "Fixed({}: {:?})", tag.as_deref().unwrap_or(""), text)
            }
            Step::Choose { options } => write!(f, "Choose({:?})", options),
            Step::Gen {
                rx,
                yacc,
                stop_at,
                max_tokens,
                max_words,
                max_bytes,
                mask_tags,
            } => {
                write!(f, "Gen(")?;
                if let Some(rx) = rx {
                    write!(f, "/{:?}/ ", rx)?;
                }
                if let Some(yacc) = yacc {
                    write!(f, "yacc:{:?} ", limit_str(yacc, 200))?;
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
                if let Some(mask_tags) = mask_tags {
                    write!(f, "mask_tags:{:?}, ", mask_tags)?;
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
    Fork { branches: Vec<Vec<Step>> },
    Stop,
}
struct StepState {
    idx: usize,
    ast: Step,
    specific: StepSpecific,

    // stop conditions
    stop_at: Option<String>,
    max_tokens: usize,
    max_words: usize,
    max_bytes: usize,

    tag: String, // can be empty
    mask_tags: Vec<String>,

    // state so far for this step
    tokens: Vec<TokenId>,
    bytes: Vec<u8>,
    word_idx: usize,
}

struct TokenInfo {
    id: TokenId,
    tag: String,
}

pub struct Runner {
    helper: AiciVmHelper,
    tokens: Vec<TokenInfo>,
    state_idx: usize,
    states: Vec<StepState>,
}

impl Debug for StepState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] tok:{}/", self.idx, self.tokens.len())?;
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
            idx: 0,
            ast: ast.clone(),
            specific,
            word_idx: 0,
            stop_at: None,
            tokens: Vec::new(),
            bytes: Vec::new(),
            max_words: usize::MAX,
            max_bytes: usize::MAX,
            max_tokens: usize::MAX,
            tag: "".to_string(),
            mask_tags: Vec::new(),
        }
    }

    fn from_ast(s: &Step) -> StepState {
        match s {
            Step::Fork { branches } => {
                assert!(branches.len() > 1, "more than one branch required in fork");
                assert!(
                    branches.iter().all(|b| b.len() > 0),
                    "fork branches cannot be empty"
                );
                Self::from_specific(
                    s,
                    StepSpecific::Fork {
                        branches: branches.clone(),
                    },
                )
            }

            Step::Fixed { text, tag } => {
                let mut r = Self::from_specific(
                    s,
                    StepSpecific::Options {
                        tokens: vec![tokenize(&text)],
                    },
                );
                if let Some(tag) = tag {
                    r.tag = tag.clone();
                }
                r
            }

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
                mask_tags,
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
                r.mask_tags = mask_tags.clone().unwrap_or_default();
                r
            }
        }
    }

    pub fn ff_state_tokens(&self) -> Option<Vec<TokenId>> {
        match &self.specific {
            StepSpecific::Options { tokens } => {
                let tt = tokens
                    .iter()
                    .filter(|t| t.len() >= self.tokens.len() + 2)
                    .collect::<Vec<_>>();
                if tt.len() == 1 {
                    Some(tt[0][self.tokens.len()..].to_vec())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn check_eos(&mut self, optional: bool) -> bool {
        self.tokens.len() >= self.max_tokens
            || self.bytes.len() >= self.max_bytes
            || self.word_idx >= self.max_words
            || (self.stop_at.is_some() && self.stop_at.as_ref().unwrap().is_empty())
            || match &mut self.specific {
                StepSpecific::Fork { .. } => false,
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

    fn attention_mask(&self, helper: &AiciVmHelper, all_tokens: &Vec<TokenInfo>) -> Vec<f32> {
        if self.mask_tags.len() == 0 {
            vec![]
        } else {
            let mut mask = vec![1.0f32; all_tokens.len()];
            for (idx, tok) in all_tokens.iter().enumerate() {
                if self.mask_tags.contains(&tok.tag) {
                    wprintln!(
                        "masking t[{}] = {:?} tag={}",
                        idx,
                        helper.trie.token_str(tok.id),
                        tok.tag
                    );
                    mask[idx] = 0.0;
                }
            }
            mask
        }
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
            StepSpecific::Fork { .. } => {
                panic!("advance on fork")
            }
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
            StepSpecific::Fork { .. } => false,
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
            StepSpecific::Fork { .. } => {
                // don't allow anything else
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
            tag: None,
        };
        states.push(StepState::from_specific(&stop_ast, StepSpecific::Stop));

        for (idx, state) in states.iter_mut().enumerate() {
            state.idx = idx;
            wprintln!("[{}] {} {:?}", idx, state.pp(), state);
        }

        Self {
            helper: AiciVmHelper::new(),
            state_idx: 0,
            tokens: Vec::new(),
            states,
        }
    }

    fn stop(&mut self, info: &str) {
        self.state_idx = self.states.len() - 1;
        wprintln!("stop: {}", info)
    }

    fn should_fork(&mut self) -> bool {
        if !self.states[self.state_idx].allows_eos() {
            return false;
        }
        let idx = self.state_idx + 1;
        if idx < self.states.len() {
            if let StepSpecific::Fork { .. } = &self.states[idx].specific {
                return true;
            }
        }
        false
    }

    fn advance(&mut self, token: TokenId) {
        let bytes = self.helper.trie.token(token);
        wprintln!(
            "advance {} {:?} {:?}",
            token,
            String::from_utf8_lossy(bytes),
            self.curr_state()
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

        self.states[self.state_idx].advance(&mut self.helper, token);

        self.tokens.push(TokenInfo {
            id: token,
            tag: self.curr_state().tag.clone(),
        });

        wprintln!(" => {:?}", self.curr_state());
    }

    fn curr_state(&self) -> &StepState {
        &self.states[self.state_idx]
    }

    fn compute(&mut self) -> ProcessResult {
        self.helper.all_disallowed();
        let mut ff_tokens = None;
        let mut can_ff = true;
        for state in &mut self.states[self.state_idx..] {
            if state.forces_eos() {
                continue;
            }
            state.apply_to(self.helper.trie.clone(), &mut self.helper.allowed_tokens);
            if can_ff {
                ff_tokens = state.ff_state_tokens();
            } else {
                ff_tokens = None;
            }
            can_ff = false;
            if !state.allows_eos() {
                break;
            }
        }

        if let Some(ff_tokens) = ff_tokens {
            ProcessResult::Splice {
                backtrack: 0,
                ff_tokens,
            }
        } else {
            self.helper.return_logit_bias()
        }
    }

    #[allow(dead_code)]
    fn print_prob(&self, tok: &str) {
        if let Some(id) = self.helper.trie.token_id(tok.as_bytes()) {
            wprintln!(
                "prob {:?} {} = {}",
                tok,
                id,
                self.helper.allowed_tokens.is_allowed(id)
            );
        } else {
            wprintln!("prob {:?} -> no token", tok)
        }
    }
}

impl AiciVm for Runner {
    fn init_prompt(&mut self, arg: InitPromptArg) {
        wprintln!("prompt: {:?}", arg.prompt);
        for t in arg.prompt {
            self.tokens.push(TokenInfo {
                id: t,
                tag: "prompt".to_string(),
            })
        }
    }

    fn pre_process(&mut self, arg: PreProcessArg) -> PreProcessResult {
        let tokens = arg.tokens;
        let ntok = tokens.len();
        if ntok > 1 {
            wprintln!("<<< {} tokens", ntok);
        }
        for token in tokens {
            self.advance(token);
        }
        if ntok > 1 {
            wprintln!(">>>");
        }

        if self.should_fork() {
            self.state_idx += 1;
            if let StepSpecific::Fork { branches } = &self.curr_state().specific {
                let attention_masks = branches
                    .iter()
                    .map(|b| StepState::from_ast(&b[0]).attention_mask(&self.helper, &self.tokens))
                    .collect::<Vec<_>>();
                PreProcessResult { attention_masks }
            } else {
                panic!();
            }
        } else {
            let mask = self.curr_state().attention_mask(&self.helper, &self.tokens);
            PreProcessResult {
                attention_masks: vec![mask],
            }
        }
    }

    fn process(&mut self, arg: ProcessArg) -> ProcessResult {
        if arg.fork_group.len() > 1 {
            if let StepSpecific::Fork { branches } = &self.curr_state().specific {
                assert!(arg.fork_group.len() == branches.len());
                let my_id = host::self_seq_id();
                let idx = arg.fork_group.iter().position(|id| *id == my_id).unwrap();
                let branch = branches[idx]
                    .iter()
                    .map(StepState::from_ast)
                    .collect::<Vec<_>>();
                self.states
                    .splice(self.state_idx..(self.state_idx + 1), branch);
            } else {
                panic!("current step not a fork");
            }
        }
        self.compute()
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
