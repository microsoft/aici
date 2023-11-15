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

use std::fmt::Debug;

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
    wprintln, AiciVm, InitPromptArg, PreProcessArg, PreProcessResult, ProcessArg, ProcessResult,
    TokenId,
};

//
// The JSON AST
//

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct VarName(String);

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TagName(String);

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct StepAttributes {
    /// Append the result of this generation to named variable.
    append_to_var: Option<VarName>,

    /// Set named variable to the result of this generation.
    set_var: Option<VarName>,

    /// For attention masking
    tag: Option<TagName>,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Step {
    // Generate exactly the provided string
    Fixed {
        /// Text to generate.
        text: String,

        /// Expand variables `{{var_name}}` in the `text`.
        /// Expansion occurs atomically, when the step is executed.
        #[serde(default)]
        expand_vars: bool,

        /// Common attributes
        #[serde(flatten)]
        attrs: StepAttributes,
    },

    // Generate exactly one of the provided strings
    Choose {
        options: Vec<String>,

        /// Common attributes
        #[serde(flatten)]
        attrs: StepAttributes,
    },

    // Generate text. It can be constrained with a regex or a yacc grammar.
    // The length can be constrained in several ways.
    Gen {
        /// Generate string that matches the regex.
        rx: Option<String>,

        /// Generate string that matches the yacc grammar.
        yacc: Option<String>,

        /// Stop generation when specific string is generated.
        /// It is still included in the output.
        stop_at: Option<String>,

        /// Do not generate more than this many tokens.
        max_tokens: Option<usize>,

        /// Do not generate more than this many words (separator is ' ').
        max_words: Option<usize>,

        /// Do not generate more than this many bytes.
        max_bytes: Option<usize>,

        /// Don't pay attention to text tagged with any of these names.
        mask_tags: Option<Vec<TagName>>,

        /// Common attributes
        #[serde(flatten)]
        attrs: StepAttributes,
    },

    /// Generate one sequence for each of the branches.
    /// If there are steps after the fork, they are executed for each branch.
    /// You can use `Stop` on some branches to prevent this.
    Fork { branches: Vec<Vec<Step>> },

    /// Stop the sequence (makes most sense in a Fork).
    Stop {},
}

impl Debug for StepAttributes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(var) = &self.append_to_var {
            write!(f, ", ${} += out", var.0)?;
        }
        if let Some(var) = &self.set_var {
            write!(f, ", ${} := out", var.0)?;
        }
        if let Some(tag) = &self.tag {
            write!(f, ", tag:{:?}", tag)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Step::Fork { branches } => write!(f, "Fork({})", branches.len()),
            Step::Stop {} => write!(f, "Stop"),
            Step::Fixed {
                text, expand_vars, ..
            } => {
                write!(
                    f,
                    "Fixed({}{:?})",
                    if *expand_vars { "f" } else { "" },
                    limit_str(text, 30)
                )
            }
            Step::Choose { options, .. } => write!(f, "Choose({})", options.len()),
            Step::Gen { .. } => {
                write!(f, "Gen()")
            }
        }
    }
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
            Step::Stop {} => write!(f, "Stop"),
            Step::Fixed {
                text,
                expand_vars,
                attrs,
            } => {
                write!(
                    f,
                    "Fixed({}{text:?}{attrs:?})",
                    if *expand_vars { "f" } else { "" }
                )
            }
            Step::Choose { options, attrs } => write!(f, "Choose({options:?}{attrs:?})"),
            Step::Gen {
                rx,
                yacc,
                stop_at,
                max_tokens,
                max_words,
                max_bytes,
                mask_tags,
                attrs,
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
                write!(f, "{attrs:?})")
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
    ExpandOptions { texts: Vec<String> },
    Gen { rx: RxStackRecognizer },
    Cfg { cfg: CfgParser },
    Fork { branches: Vec<Vec<Step>> },
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

    mask_tags: Vec<TagName>,
    attrs: StepAttributes,

    // state so far for this step
    tokens: Vec<TokenId>,
    bytes: Vec<u8>,
    word_idx: usize,
}

struct TokenInfo {
    id: TokenId,
    tag: TagName,
}

pub struct Runner {
    trie: TokTrie,
    vars: host::VariableStorage,
    tokens: Vec<TokenInfo>,
    prev_state_idx: usize,
    state_idx: usize,
    states: Vec<StepState>,
    log_advance: bool,
}

impl Debug for StepState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] tok:{}/", self.ast, self.tokens.len())?;
        if self.max_tokens > 10000 {
            write!(f, "inf")?;
        } else {
            write!(f, "{}", self.max_tokens)?;
        }
        if self.max_words < 10000 {
            write!(f, " word:{}/{}", self.word_idx, self.max_words)?;
        }
        if self.max_bytes < 100000 {
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

    fn from_specific(ast: &Step, attrs: &StepAttributes, specific: StepSpecific) -> StepState {
        StepState {
            ast: ast.clone(),
            attrs: attrs.clone(),
            specific,
            word_idx: 0,
            stop_at: None,
            tokens: Vec::new(),
            bytes: Vec::new(),
            max_words: usize::MAX,
            max_bytes: usize::MAX,
            max_tokens: usize::MAX,
            mask_tags: Vec::new(),
        }
    }

    fn from_ast(s: &Step) -> StepState {
        match s {
            Step::Stop {} => Self::from_specific(s, &StepAttributes::default(), StepSpecific::Stop),

            Step::Fork { branches } => {
                assert!(branches.len() > 1, "more than one branch required in fork");
                assert!(
                    branches.iter().all(|b| b.len() > 0),
                    "fork branches cannot be empty"
                );
                Self::from_specific(
                    s,
                    &StepAttributes::default(),
                    StepSpecific::Fork {
                        branches: branches.clone(),
                    },
                )
            }

            Step::Fixed {
                text,
                expand_vars,
                attrs,
            } => Self::from_specific(
                s,
                attrs,
                if *expand_vars {
                    StepSpecific::ExpandOptions {
                        texts: vec![text.clone()],
                    }
                } else {
                    StepSpecific::Options {
                        tokens: vec![tokenize(&text)],
                    }
                },
            ),

            Step::Choose { options, attrs } => Self::from_specific(
                s,
                attrs,
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
                attrs,
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
                let mut r = Self::from_specific(s, attrs, spec);
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
                StepSpecific::ExpandOptions { texts } => {
                    assert!(self.tokens.len() == 0);
                    if optional {
                        texts.iter().any(|t| t.len() == 0)
                    } else {
                        texts.iter().all(|t| t.len() == 0)
                    }
                }
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

    fn attention_mask(&self, trie: &TokTrie, all_tokens: &Vec<TokenInfo>) -> Vec<f32> {
        if self.mask_tags.len() == 0 {
            vec![]
        } else {
            let mut mask = vec![1.0f32; all_tokens.len()];
            for (idx, tok) in all_tokens.iter().enumerate() {
                if self.mask_tags.contains(&tok.tag) {
                    wprintln!(
                        "masking t[{}] = {:?} tag={:?}",
                        idx,
                        trie.token_str(tok.id),
                        tok.tag
                    );
                    mask[idx] = 0.0;
                }
            }
            mask
        }
    }

    fn advance(&mut self, trie: &TokTrie, token: TokenId) {
        self.tokens.push(token);

        let bytes = trie.token(token);
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
            StepSpecific::ExpandOptions { .. } => {
                panic!("advance on ExpandOptions")
            }
            StepSpecific::Fork { .. } => {
                panic!("advance on fork")
            }
            StepSpecific::Stop => {}
            StepSpecific::Options { tokens } => {
                tokens.retain(has_token_at(token, self.tokens.len() - 1))
            }
            StepSpecific::Cfg { cfg } => trie.append_token(cfg, token),
            StepSpecific::Gen { rx } => trie.append_token(rx, token),
        }

        fn is_boundry(b: u8) -> bool {
            b == b' ' || b == b'\n' || b == b'\t'
        }
    }

    // the 'mut' on self is bogus - the state of the 'rx' doesn't change
    fn allows_token(&mut self, trie: &TokTrie, token: TokenId) -> bool {
        if token == trie.special_token(SpecialToken::EndOfSentence) {
            return self.allows_eos();
        }
        if self.forces_eos() {
            return false;
        }
        match &mut self.specific {
            StepSpecific::ExpandOptions { .. } => false,
            StepSpecific::Fork { .. } => false,
            StepSpecific::Stop => false,
            StepSpecific::Options { tokens } => {
                tokens.iter().any(has_token_at(token, self.tokens.len()))
            }
            StepSpecific::Cfg { cfg } => trie.token_allowed(cfg, token),
            StepSpecific::Gen { rx } => trie.token_allowed(rx, token),
        }
    }

    fn finish(&mut self, vars: &host::VariableStorage) {
        wprintln!(
            "finish: {self:?} {:?}",
            String::from_utf8_lossy(&self.bytes)
        );
        if let Some(v) = self.attrs.append_to_var.as_ref() {
            vars.append(&v.0, self.bytes.clone());
        }
        if let Some(v) = self.attrs.set_var.as_ref() {
            vars.set(&v.0, self.bytes.clone());
        }
    }

    fn concretize(&mut self, vars: &host::VariableStorage) {
        match &mut self.specific {
            StepSpecific::ExpandOptions { texts } => {
                let re = regex_automata::meta::Regex::new(r"\{\{[a-zA-Z0-9_]+\}\}").unwrap();
                let tokens = texts
                    .iter()
                    .map(|text| {
                        let mut new_text = String::with_capacity(text.len());
                        let mut last_match = 0;
                        for mtch in re.find_iter(text) {
                            new_text.push_str(&text[last_match..mtch.start()]);
                            let var = &text[(mtch.start() + 2)..(mtch.end() - 2)];
                            let val = vars.get(var).unwrap_or(b"???".to_vec());
                            let val = String::from_utf8(val).unwrap();
                            new_text.push_str(&val);
                            last_match = mtch.end();
                        }
                        new_text.push_str(&text[last_match..]);
                        wprintln!("exp: {text} -> {new_text}");
                        return tokenize(&new_text);
                    })
                    .collect();
                self.specific = StepSpecific::Options { tokens }
            }
            _ => {}
        }
    }

    fn apply_to(&mut self, trie: &TokTrie, toks: &mut SimpleVob) {
        match &mut self.specific {
            StepSpecific::Stop => {
                toks.allow_token(trie.special_token(SpecialToken::EndOfSentence));
            }
            StepSpecific::ExpandOptions { .. } => {}
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
        states.push(StepState::from_ast(&Step::Stop {}));

        for (idx, state) in states.iter_mut().enumerate() {
            wprintln!("[{}] {} {:?}", idx, state.pp(), state);
        }

        Self {
            trie: TokTrie::from_host(),
            state_idx: 0,
            prev_state_idx: 0,
            tokens: Vec::new(),
            vars: host::VariableStorage::new(),
            states,
            log_advance: false,
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
        let bytes = self.trie.token(token);
        if self.log_advance {
            wprintln!(
                "advance {} {:?} {:?}",
                token,
                String::from_utf8_lossy(bytes),
                self.curr_state()
            );
        }

        // skip as many states as we can (that allow EOS), and find the last one that allows the token
        let mut last_idx = usize::MAX;
        for idx in self.state_idx..self.states.len() {
            if self.states[idx].allows_token(&self.trie, token) {
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

        self.states[self.state_idx].advance(&self.trie, token);

        self.tokens.push(TokenInfo {
            id: token,
            tag: self
                .curr_state()
                .attrs
                .tag
                .clone()
                .unwrap_or(TagName("other".to_string())),
        });

        if self.log_advance {
            wprintln!(" => {:?}", self.curr_state());
        }
    }

    fn curr_state(&self) -> &StepState {
        &self.states[self.state_idx]
    }

    fn compute(&mut self) -> ProcessResult {
        let mut allowed_tokens = self.trie.alloc_token_set();
        let mut ff_tokens = None;
        let mut can_ff = true;
        let mut all_eos = true;

        for state in &mut self.states[self.state_idx..] {
            state.concretize(&self.vars);
            if state.forces_eos() {
                if all_eos {
                    self.state_idx += 1;
                }
                continue;
            }
            all_eos = false;
            state.apply_to(&self.trie, &mut allowed_tokens);
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

        self.finish_states();

        if let Some(ff_tokens) = ff_tokens {
            ProcessResult::Splice {
                backtrack: 0,
                ff_tokens,
            }
        } else {
            host::return_logit_bias(&allowed_tokens);
            ProcessResult::SampleWithBias
        }
    }

    fn finish_states(&mut self) {
        while self.prev_state_idx < self.state_idx {
            self.states[self.prev_state_idx].finish(&self.vars);
            self.prev_state_idx += 1;
        }
    }
}

impl AiciVm for Runner {
    fn init_prompt(&mut self, arg: InitPromptArg) {
        wprintln!("prompt: {:?}", arg.prompt);
        for t in arg.prompt {
            self.tokens.push(TokenInfo {
                id: t,
                tag: TagName("prompt".to_string()),
            })
        }
    }

    fn pre_process(&mut self, arg: PreProcessArg) -> PreProcessResult {
        let tokens = arg.tokens;
        let ntok = tokens.len();
        if ntok > 1 && self.log_advance {
            wprintln!("<<< {} tokens", ntok);
        }
        for token in tokens {
            self.advance(token);
        }
        if ntok > 1 && self.log_advance {
            wprintln!(">>>");
        }

        if self.should_fork() {
            self.state_idx += 1;
            if let StepSpecific::Fork { branches } = &self.curr_state().specific {
                let attention_masks = branches
                    .iter()
                    .map(|b| StepState::from_ast(&b[0]).attention_mask(&self.trie, &self.tokens))
                    .collect::<Vec<_>>();
                PreProcessResult { attention_masks }
            } else {
                panic!();
            }
        } else {
            let mask = self.curr_state().attention_mask(&self.trie, &self.tokens);
            PreProcessResult {
                attention_masks: vec![mask],
            }
        }
    }

    fn process(&mut self, arg: ProcessArg) -> ProcessResult {
        self.finish_states();

        if arg.fork_group.len() > 1 {
            wprintln!("fork group: {:?}", arg.fork_group);
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
}

fn main() {
    cfg::cfg_test().unwrap();
    //    let _run = sample_prog();
}

fn runner_from_env() -> Runner {
    let a = host::arg_bytes();
    match serde_json::from_slice(&a) {
        Ok(p) => Runner::new(p),
        Err(e) => {
            let mut col = e.column().saturating_sub(1);
            let mut line = e.line().saturating_sub(1);
            for off in 0..a.len() {
                if line == 0 {
                    col -= 1;
                    if col == 0 {
                        wprintln!(
                            "at: {:?} <HERE> {:?}",
                            String::from_utf8_lossy(&a[off.saturating_sub(30)..off]),
                            String::from_utf8_lossy(&a[off..std::cmp::min(a.len(), off + 30)]),
                        );
                        break;
                    }
                }
                if a[off] == b'\n' {
                    line -= 1;
                }
            }
            wprintln!("JSON AST parsing error: {:?}", e);
            panic!()
        }
    }
}

aici_expose_all!(Runner, runner_from_env());
