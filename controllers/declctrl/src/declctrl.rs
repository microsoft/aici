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

use aici_abi::{
    aici_expose_all,
    bytes::limit_str,
    cfg::CfgParser,
    rx::RecRx,
    rx::RxStackRecognizer,
    svob::SimpleVob,
    tokenize, tokenize_bytes,
    toktree::{Recognizer, SpecialToken, TokTrie},
    AiciCtrl, InitPromptArg, InitPromptResult, MidProcessArg, MidProcessResult, PostProcessArg,
    PostProcessResult, PreProcessArg, PreProcessResult, TokenId, VariableStorage,
};
use core::panic;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

const LOG_ADVANCE: bool = false;

//
// The JSON AST
//

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub struct VarName(String);

impl Debug for VarName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "${}", self.0)
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub struct LabelName(String);

impl Debug for LabelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:", self.0)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TagName(String);

pub const SEP: &str = "\u{FF0C}"; // this is '，' - full-width Comma
pub const SEP_REPL: &str = ", ";

#[derive(Serialize, Deserialize, Clone)]
pub enum Expr {
    /// Literal string
    String { str: String },
    /// The current value of this variable, or empty string if not set.
    Var { var: VarName },
    /// The result of the current step (typically Gen, but can be anything).
    Current {},
    Concat {
        /// Concatenate these expressions.
        parts: Vec<Expr>,
        /// When set, add SEP between elements.
        #[serde(default)]
        list: bool,
    },
    /// Evaluates to `eq` if `a == b`, otherwise to `neq`.
    IfEq {
        a: Box<Expr>,
        b: Box<Expr>,
        eq: Box<Expr>,
        neq: Box<Expr>,
    },
    Extract {
        /// Extract from where?
        from: Box<Expr>,
        /// Regular expression to search for.
        rx: String,
        /// The template for the result. Defaults to "$1".
        template: Option<String>,
        /// When set, find all occurrences of `rx` and add SEP between them.
        /// Also, if SEP occurs in any element, replace it with SEP_REPL.
        #[serde(default)]
        list: bool,
    },
}

fn write_list<T: Debug>(lst: &[T], sep: &str, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for (idx, part) in lst.iter().enumerate() {
        if idx > 0 {
            write!(f, "{}", sep)?;
        }
        write!(f, "{:?}", part)?;
    }
    Ok(())
}

fn extract_template(t: &Option<String>) -> String {
    if let Some(t) = t {
        t.clone()
    } else {
        "$1".to_string()
    }
}

impl Debug for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::String { str } => write!(f, "{:?}", str),
            Expr::Var { var } => write!(f, "{:?}", var),
            Expr::Current {} => write!(f, "$current"),
            Expr::Concat { parts, list } => {
                if *list {
                    write!(f, "[")?;
                    write_list(parts, SEP, f)?;
                    write!(f, "]")?;
                } else {
                    write!(f, "(")?;
                    write_list(parts, " + ", f)?;
                    write!(f, ")")?;
                }
                Ok(())
            }
            Expr::IfEq { a, b, eq, neq } => {
                write!(f, "(if {:?} == {:?} then {:?} else {:?})", a, b, eq, neq)
            }
            Expr::Extract {
                from,
                rx,
                list,
                template,
            } => {
                if *list {
                    write!(f, "extract_list(")?;
                } else {
                    write!(f, "extract(")?;
                }
                write!(
                    f,
                    "/{rx:?}/ -> {repl:?} from {from:?})",
                    repl = extract_template(template)
                )?;
                Ok(())
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Stmt {
    /// Set named variable
    Set { var: VarName, expr: Expr },
}

impl Debug for Stmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stmt::Set { var, expr } => write!(f, "{:?} := {:?}", var, expr),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct StepAttributes {
    /// What to do with the output of the generation (if any).
    #[serde(default)]
    stmts: Vec<Stmt>,

    /// For attention masking
    tag: Option<TagName>,

    /// Label this step, so that it can be backtracked to later.
    label: Option<LabelName>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct InnerConstraint {
    /// After this strings is generated,
    pub after: String,
    /// chose one of these options.
    pub options: Expr,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Step {
    // Generate exactly the provided string
    Fixed {
        /// Text to generate.
        text: Expr,

        /// First backtrack to this label, and then generate text.
        following: Option<LabelName>,

        /// Common attributes
        #[serde(flatten)]
        attrs: StepAttributes,
    },

    /// Generate exactly one of the provided strings
    Choose {
        /// This is typically a Expr::Concat([...], list=true)
        options: Expr,

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

        /// Constraints to apply in the middle of the generation.
        #[serde(default)]
        inner: Vec<InnerConstraint>,

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

    /// Wait for all listed variables to be set.
    Wait { vars: Vec<VarName> },

    /// Stop the sequence (makes most sense in a Fork).
    Stop {},
}

impl Debug for StepAttributes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(tag) = &self.tag {
            write!(f, ", tag:{:?}", tag)?;
        }
        if let Some(LabelName(label)) = &self.label {
            write!(f, ", label:{label}")?;
        }
        if self.stmts.len() == 1 {
            write!(f, ", stmt: {:?}", self.stmts[0])?;
        } else if self.stmts.len() > 0 {
            write!(f, ", stmts:\n")?;
            for s in &self.stmts {
                write!(f, "    {:?}\n", s)?;
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Step::Fork { branches } => write!(f, "Fork({})", branches.len()),
            Step::Wait { vars } => write!(f, "Wait({})", vars.len()),
            Step::Stop {} => write!(f, "Stop"),
            Step::Fixed { .. } => write!(f, "Fixed()"),
            Step::Choose { .. } => write!(f, "Choose()"),
            Step::Gen { .. } => write!(f, "Gen()"),
        }
    }
}

impl Debug for InnerConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} -> {:?}", self.after, self.options)
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
            Step::Wait { vars } => write!(f, "Wait({:?})", vars),
            Step::Stop {} => write!(f, "Stop"),
            Step::Fixed {
                text,
                attrs,
                following,
            } => {
                write!(
                    f,
                    "Fixed({text:?}{attrs:?}){}",
                    following
                        .as_ref()
                        .map(|l| format!(" following:{}", l.0))
                        .unwrap_or_default(),
                )
            }
            Step::Choose { options, attrs } => write!(f, "Choose({options:?}{attrs:?})"),
            Step::Gen {
                rx,
                yacc,
                inner,
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
                if inner.len() > 0 {
                    write!(f, "inner:")?;
                    for ic in inner {
                        write!(f, " /{:?}/ -> {:?}, ", ic.after, ic.options)?;
                    }
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
    ExpandOptions { text: Expr, many: bool },
    Inner { constraints: Vec<InnerConstraint> },
    Rx { rx: RxStackRecognizer },
    Cfg { cfg: CfgParser },
    Fork { branches: Vec<Vec<StepState>> },
    Wait { vars: Vec<VarName> },
    Stop,
}
struct StepState {
    ast: Step,
    specific: StepSpecific,
    following: Option<LabelName>,

    // stop conditions
    stop_at: Option<String>,
    max_tokens: usize,
    max_words: usize,
    max_bytes: usize,

    // if true, this step was derived from the next step
    is_derived: bool,

    mask_tags: Vec<TagName>,
    attrs: StepAttributes,

    // state so far for this step
    num_tokens: usize,
    num_bytes: usize,
    num_words: usize,
}

struct TokenInfo {
    id: TokenId,
    tag: TagName,
    labels: Vec<LabelName>,
}

struct RunnerCtx {
    trie: TokTrie,
    vars: VariableStorage,
    tokens: Vec<TokenInfo>,
    bytes: Vec<u8>,
}

impl RunnerCtx {
    pub fn string_position(&self, sidx: usize, str: &str) -> Option<usize> {
        let slen = str.len();
        let start = sidx.saturating_sub(slen);
        self.bytes[start..]
            .windows(slen)
            .position(|w| w == str.as_bytes())
            .map(|x| start + x)
    }

    fn do_expand(&self, expr: &Expr, curr_ctx: Option<&StepState>) -> Vec<u8> {
        match expr {
            Expr::String { str } => str.as_bytes().to_vec(),
            Expr::Var { var } => match self.vars.get(&var.0) {
                Some(r) => r,
                None => Vec::new(),
            },
            Expr::Current {} => match curr_ctx {
                Some(ctx) => self.bytes[self.bytes.len() - ctx.num_bytes..].to_vec(),
                None => panic!("$current used outside of stmts:..."),
            },
            Expr::Concat { parts, list } => {
                let parts = parts
                    .iter()
                    .map(|p| self.do_expand(p, curr_ctx))
                    .collect::<Vec<_>>();
                if *list {
                    parts.join(SEP.as_bytes()).to_vec()
                } else {
                    parts.join("".as_bytes()).to_vec()
                }
            }
            Expr::IfEq { a, b, eq, neq } => {
                let a = self.do_expand(a, curr_ctx);
                let b = self.do_expand(b, curr_ctx);
                if a == b {
                    self.do_expand(eq, curr_ctx)
                } else {
                    self.do_expand(neq, curr_ctx)
                }
            }
            Expr::Extract {
                from,
                rx,
                template,
                list,
            } => {
                let from = self.do_expand(from, curr_ctx);
                let t = extract_template(template);
                let template = t.as_bytes();
                let re = regex_automata::meta::Regex::new(rx).unwrap();
                let mut res = Vec::new();
                for cap in re.captures_iter(&from) {
                    res.push(cap.interpolate_bytes(&from, template));
                    if !list {
                        break;
                    }
                }
                res.join(SEP.as_bytes())
            }
        }
    }

    pub fn expand(&self, expr: &Expr) -> Vec<u8> {
        self.do_expand(expr, None)
    }

    pub fn expand_with_curr(&self, expr: &Expr, curr_ctx: &StepState) -> Vec<u8> {
        self.do_expand(expr, Some(curr_ctx))
    }
}

pub struct Runner {
    ctx: RunnerCtx,
    prev_state_idx: usize,
    state_idx: usize,
    states: Vec<StepState>,
}

impl Debug for StepState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] tok:{}/", self.ast, self.num_tokens)?;
        if self.max_tokens > 10000 {
            write!(f, "inf")?;
        } else {
            write!(f, "{}", self.max_tokens)?;
        }
        if self.max_words < 10000 {
            write!(f, " word:{}/{}", self.num_words, self.max_words)?;
        }
        if self.max_bytes < 100000 {
            write!(f, " byte:{}/{}", self.num_bytes, self.max_bytes)?;
        }
        Ok(())
    }
}

fn has_token_at(t: TokenId, idx: usize) -> impl for<'a> Fn(&'a Vec<TokenId>) -> bool {
    move |v: &Vec<TokenId>| idx < v.len() && v[idx] == t
}

fn split_vec(vec: &[u8], sep: &[u8]) -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    let mut last = 0;
    for (i, window) in vec.windows(sep.len()).enumerate() {
        if window == sep {
            result.push(vec[last..i].to_vec());
            last = i + sep.len();
        }
    }
    if last < vec.len() {
        result.push(vec[last..].to_vec());
    }
    result
}

fn val_to_list(val: &[u8]) -> Vec<Vec<u8>> {
    split_vec(val, SEP.as_bytes())
}

impl StepSpecific {
    fn is_fork(&self) -> bool {
        match self {
            StepSpecific::Fork { .. } => true,
            _ => false,
        }
    }
}

impl StepState {
    #[allow(dead_code)]
    fn pp(&self) -> String {
        format!("{:?}", self.ast)
    }

    fn new_with_attrs(ast: &Step, attrs: &StepAttributes, specific: StepSpecific) -> StepState {
        StepState {
            ast: ast.clone(),
            attrs: attrs.clone(),
            specific,
            stop_at: None,
            max_words: usize::MAX,
            max_bytes: usize::MAX,
            max_tokens: usize::MAX,
            mask_tags: Vec::new(),
            num_words: 0,
            num_tokens: 0,
            num_bytes: 0,
            following: None,
            is_derived: false,
        }
    }

    fn new(ast: &Step, specific: StepSpecific) -> StepState {
        Self::new_with_attrs(ast, &StepAttributes::default(), specific)
    }

    fn with<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut StepState) -> (),
    {
        f(&mut self);
        self
    }

    fn from_ast(s: &Step) -> StepState {
        match s {
            Step::Stop {} => Self::new(s, StepSpecific::Stop),

            Step::Wait { vars } => Self::new(s, StepSpecific::Wait { vars: vars.clone() }),

            Step::Fork { branches } => {
                assert!(branches.len() > 1, "more than one branch required in fork");
                assert!(
                    branches.iter().all(|b| b.len() > 0),
                    "fork branches cannot be empty"
                );
                Self::new(
                    s,
                    StepSpecific::Fork {
                        branches: branches
                            .iter()
                            .map(|b| b.iter().map(|s| Self::from_ast(s)).collect())
                            .collect(),
                    },
                )
            }

            Step::Fixed {
                text,
                attrs,
                following,
            } => Self::new_with_attrs(
                s,
                attrs,
                StepSpecific::ExpandOptions {
                    text: text.clone(),
                    many: false,
                },
            )
            .with(|s| s.following = following.clone()),

            Step::Choose { options, attrs } => Self::new_with_attrs(
                s,
                attrs,
                StepSpecific::ExpandOptions {
                    text: options.clone(),
                    many: true,
                },
            ),

            Step::Gen {
                rx,
                yacc,
                stop_at,
                inner,
                max_tokens,
                max_bytes,
                max_words,
                mask_tags,
                attrs,
            } => {
                let spec = match (yacc, rx) {
                    (None, None) if inner.len() > 0 => StepSpecific::Inner {
                        constraints: inner.clone(),
                    },
                    _ if inner.len() > 0 => {
                        panic!("can't have inner= and either yacc= or rx=")
                    }
                    (Some(_), Some(_)) => {
                        panic!("can't have both yacc= and rx=")
                    }
                    (Some(yacc), None) => StepSpecific::Cfg {
                        cfg: CfgParser::from_yacc(yacc).expect("invalid grammar"),
                    },
                    _ => {
                        let defl = "(.|\n)+".to_string();
                        let rx = rx.as_deref().unwrap_or(&defl);
                        StepSpecific::Rx {
                            rx: RecRx::from_rx(&rx).to_stack_recognizer(),
                        }
                    }
                };
                let mut r = Self::new_with_attrs(s, attrs, spec);
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
                    .filter(|t| t.len() >= self.num_tokens + 2)
                    .collect::<Vec<_>>();
                if tt.len() == 1 {
                    Some(tt[0][self.num_tokens..].to_vec())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn check_eos(&mut self, optional: bool) -> bool {
        if self.num_tokens >= self.max_tokens
            || self.num_bytes >= self.max_bytes
            || self.num_words >= self.max_words
        {
            return true;
        }

        match &self.stop_at {
            Some(s) => return s.len() == 0,
            None => {}
        }

        match &mut self.specific {
            StepSpecific::Fork { .. } => false,
            StepSpecific::Wait { .. } => false,
            StepSpecific::Stop => false,
            StepSpecific::ExpandOptions { .. } => {
                assert!(self.num_tokens == 0);
                false
                // if optional {
                //     texts.iter().any(|t| t.len() == 0)
                // } else {
                //     texts.iter().all(|t| t.len() == 0)
                // }
            }
            StepSpecific::Options { tokens } => {
                if optional {
                    tokens.iter().any(|t| self.num_tokens >= t.len())
                } else {
                    tokens.iter().all(|t| self.num_tokens >= t.len())
                }
            }
            StepSpecific::Cfg { cfg } => {
                cfg.special_allowed(SpecialToken::EndOfSentence)
                    && (optional || (0..=255).all(|byte| !cfg.byte_allowed(byte)))
            }
            StepSpecific::Inner { .. } => optional,
            StepSpecific::Rx { rx } => {
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

    fn attention_mask(&self, ctx: &RunnerCtx) -> Vec<f32> {
        if self.mask_tags.len() == 0 {
            vec![]
        } else {
            let mut mask = vec![1.0f32; ctx.tokens.len()];
            for (idx, tok) in ctx.tokens.iter().enumerate() {
                if self.mask_tags.contains(&tok.tag) {
                    println!(
                        "masking t[{}] = {:?} tag={:?}",
                        idx,
                        ctx.trie.token_str(tok.id),
                        tok.tag
                    );
                    mask[idx] = 0.0;
                }
            }
            mask
        }
    }

    fn advance(&mut self, runner: &RunnerCtx, token: TokenId) -> Option<StepState> {
        let nbytes = runner.trie.token(token).len();
        self.num_tokens += 1;
        self.num_bytes += nbytes;
        let sidx = runner.bytes.len() - nbytes;

        for idx in sidx.saturating_sub(1)..runner.bytes.len().saturating_sub(1) {
            if !is_boundary(runner.bytes[idx]) && is_boundary(runner.bytes[idx + 1]) {
                self.num_words += 1;
                break;
            }
        }

        if let Some(stop) = &self.stop_at {
            if stop.len() > 0 && runner.string_position(sidx, stop).is_some() {
                self.stop_at = Some("".to_string())
            }
        }

        match &mut self.specific {
            StepSpecific::ExpandOptions { .. } => panic!("advance on ExpandOptions"),
            StepSpecific::Fork { .. } => panic!("advance on fork"),
            StepSpecific::Wait { .. } => panic!("advance on wait"),
            StepSpecific::Stop => {}
            StepSpecific::Options { tokens } => {
                tokens.retain(has_token_at(token, self.num_tokens - 1))
            }
            StepSpecific::Cfg { cfg } => runner.trie.append_token(cfg, token),
            StepSpecific::Rx { rx } => runner.trie.append_token(rx, token),
            StepSpecific::Inner { constraints } => {
                for c in constraints {
                    let pos = runner.string_position(sidx, &c.after);
                    if let Some(p) = pos {
                        let pref = &runner.bytes[p + c.after.len()..];
                        let expanded = runner.expand(&c.options);
                        let tokens = val_to_list(&expanded)
                            .iter()
                            .filter_map(|e| {
                                if e.starts_with(pref) {
                                    Some(tokenize_bytes(&e[pref.len()..]))
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>();
                        let mut new_state = StepState::from_ast(&self.ast);
                        // println!("tokens: {p} {pref:?} {tokens:?}");
                        new_state.specific = StepSpecific::Options { tokens };
                        new_state.max_tokens -= self.num_tokens;
                        new_state.max_bytes -= self.num_bytes;
                        new_state.max_words -= self.num_words;
                        new_state.is_derived = true;
                        // println!("here: {new_state:?}");
                        return Some(new_state);
                    }
                }
            }
        }

        return None;

        fn is_boundary(b: u8) -> bool {
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
            StepSpecific::Wait { .. } => false,
            StepSpecific::Stop => false,
            StepSpecific::Inner { .. } => true,
            StepSpecific::Options { tokens } => {
                tokens.iter().any(has_token_at(token, self.num_tokens))
            }
            StepSpecific::Cfg { cfg } => trie.token_allowed(cfg, token),
            StepSpecific::Rx { rx } => trie.token_allowed(rx, token),
        }
    }

    fn finish(&mut self, runner: &RunnerCtx) {
        let sidx = runner.bytes.len() - self.num_bytes;
        let my_bytes = runner.bytes[sidx..].to_vec();
        println!("finish: {self:?} {:?}", String::from_utf8_lossy(&my_bytes));
        for s in &self.attrs.stmts {
            match s {
                Stmt::Set { var, expr } => {
                    let val = runner.expand_with_curr(&expr, self);
                    println!("  set {:?} := {:?}", var, String::from_utf8_lossy(&val));
                    runner.vars.set(&var.0, val);
                }
            }
        }
    }

    fn concretize(&mut self, runner: &RunnerCtx) {
        match &mut self.specific {
            StepSpecific::ExpandOptions { text, many } => {
                let expanded = runner.expand(text);
                let options = if *many {
                    val_to_list(&expanded)
                } else {
                    vec![expanded]
                };
                let tokens = options
                    .iter()
                    .map(|v| tokenize_bytes(v))
                    .collect::<Vec<_>>();
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
            StepSpecific::Wait { .. } => {}
            StepSpecific::Fork { .. } => {}
            StepSpecific::Inner { .. } => {
                // anything goes, until one of constraint strings is generated
                toks.set_all(true);
                toks.disallow_token(trie.special_token(SpecialToken::EndOfSentence));
            }
            StepSpecific::Options { tokens } => {
                for v in tokens {
                    if self.num_tokens < v.len() {
                        toks.allow_token(v[self.num_tokens]);
                    }
                }
            }
            StepSpecific::Rx { rx } => {
                trie.add_bias(rx, toks, &[]);
            }
            StepSpecific::Cfg { cfg } => {
                trie.add_bias(cfg, toks, &[]);
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
            println!("[{}] {} {:?}", idx, state.pp(), state);
        }

        Self {
            ctx: RunnerCtx {
                trie: TokTrie::from_host(),
                tokens: Vec::new(),
                bytes: Vec::new(),
                vars: VariableStorage::new(),
            },
            state_idx: 0,
            prev_state_idx: 0,
            states,
        }
    }

    fn stop(&mut self, info: &str) {
        println!("stop: {}", info);
        self.finish_states();
        self.state_idx = self.states.len() - 1;
        // don't finish the states
        self.prev_state_idx = self.state_idx;
    }

    fn can_move_to_next_state(&mut self) -> bool {
        self.states[self.state_idx].allows_eos()
    }

    fn next_state(&self) -> Option<&StepSpecific> {
        let idx = self.state_idx + 1;
        if idx < self.states.len() {
            Some(&self.states[idx].specific)
        } else {
            None
        }
    }

    fn advance(&mut self, token: TokenId) {
        let bytes = self.ctx.trie.token(token);
        if LOG_ADVANCE {
            println!(
                "advance {} {:?} {:?}",
                token,
                String::from_utf8_lossy(bytes),
                self.curr_state()
            );
        }

        if token == self.ctx.trie.special_token(SpecialToken::EndOfSentence) {
            if self.state_idx < self.states.len() - 1 {
                println!("EOS: advancing to next state");
                self.state_idx += 1;
            }
        }

        // skip as many states as we can (that allow EOS), and find the last one that allows the token
        let mut last_idx = usize::MAX;
        let mut labels = Vec::new();
        for idx in self.state_idx..self.states.len() {
            if self.states[idx].allows_token(&self.ctx.trie, token) {
                last_idx = idx;
                if let Some(lbl) = &self.states[idx].attrs.label {
                    labels.push(lbl.clone());
                }
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

        // we only want first-time labels
        let labels = if let Some(t) = self.ctx.tokens.last() {
            labels
                .into_iter()
                .filter(|l| !t.labels.contains(l))
                .collect()
        } else {
            labels
        };

        self.ctx.tokens.push(TokenInfo {
            id: token,
            labels,
            tag: self
                .curr_state()
                .attrs
                .tag
                .clone()
                .unwrap_or(TagName("other".to_string())),
        });

        self.ctx.bytes.extend_from_slice(bytes);

        if let Some(new_state) = self.states[self.state_idx].advance(&self.ctx, token) {
            self.states.insert(self.state_idx, new_state)
        }

        while self.states[self.state_idx].forces_eos() {
            self.state_idx += 1;
        }

        if LOG_ADVANCE {
            println!(" => {:?}", self.curr_state());
        }
    }

    fn curr_state(&self) -> &StepState {
        &self.states[self.state_idx]
    }

    fn find_label(&self, label: &LabelName) -> Option<usize> {
        self.ctx
            .tokens
            .iter()
            .position(|t| t.labels.contains(label))
    }

    fn try_backtrack(&mut self) -> MidProcessResult {
        for idx in self.state_idx..self.states.len() {
            self.states[idx].concretize(&self.ctx);

            let state = &self.states[idx];
            if let Some(label) = &state.following {
                if let StepSpecific::Options { tokens } = &state.specific {
                    assert!(tokens.len() == 1);
                    let lbl_idx = self.find_label(label);
                    if lbl_idx.is_none() {
                        panic!("label not found: {label:?}");
                    }
                    // lbl_idx is index of the first token with the label
                    // we want to pop that token and everything that follows
                    let backtrack = (self.ctx.tokens.len() - lbl_idx.unwrap()) as u32;

                    let t0 = self.ctx.tokens.iter().map(|t| t.id).collect::<Vec<_>>();
                    println!(
                        "slice: {t0:?} -> {:?} + {:?}",
                        &t0[..lbl_idx.unwrap()],
                        tokens[0]
                    );

                    return MidProcessResult::Splice {
                        backtrack,
                        ff_tokens: tokens[0].clone(),
                    };
                } else {
                    panic!("following on non-options");
                }
            }

            if !self.states[idx].allows_eos() {
                break;
            }
        }

        self.compute()
    }

    fn compute(&mut self) -> MidProcessResult {
        let mut allowed_tokens = self.ctx.trie.alloc_token_set();
        let mut ff_tokens = None;
        let mut can_ff = true;
        let mut all_eos = true;

        for state in &mut self.states[self.state_idx..] {
            state.concretize(&self.ctx);
            if state.forces_eos() {
                if all_eos {
                    self.state_idx += 1;
                }
                continue;
            }
            all_eos = false;
            state.apply_to(&self.ctx.trie, &mut allowed_tokens);
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
            MidProcessResult::Splice {
                backtrack: 0,
                ff_tokens,
            }
        } else {
            MidProcessResult::SampleWithBias { allowed_tokens }
        }
    }

    fn maybe_wait(&mut self) -> bool {
        if let StepSpecific::Wait { vars } = &self.curr_state().specific {
            if vars.iter().any(|name| self.ctx.vars.get(&name.0).is_none()) {
                println!("wait {vars:?} suspend");
                true
            } else {
                println!("wait {vars:?} done");
                self.state_idx += 1;
                false
            }
        } else {
            false
        }
    }

    fn finish_states(&mut self) {
        while self.prev_state_idx < self.state_idx {
            self.states[self.prev_state_idx].finish(&self.ctx);
            self.prev_state_idx += 1;
        }
    }
}

impl AiciCtrl for Runner {
    fn init_prompt(&mut self, arg: InitPromptArg) -> InitPromptResult {
        println!("prompt: {:?}", arg.prompt);
        for t in arg.prompt {
            self.ctx.tokens.push(TokenInfo {
                id: t,
                tag: TagName("prompt".to_string()),
                labels: vec![LabelName("prompt".to_string())],
            })
        }
        InitPromptResult::default()
    }

    fn post_process(&mut self, arg: PostProcessArg) -> PostProcessResult {
        self.finish_states();

        if arg.backtrack > 0 {
            self.ctx
                .tokens
                .drain(self.ctx.tokens.len() - arg.backtrack as usize..);
            let state = self.curr_state();
            assert!(state.following.is_some());
            assert!(state.num_tokens == 0);
        }

        // if in wait state, don't do anything...
        if let StepSpecific::Wait { .. } = &self.curr_state().specific {
            return PostProcessResult::continue_();
        }

        let tokens = arg.tokens;
        let ntok = tokens.len();
        if ntok > 1 && LOG_ADVANCE {
            println!("<<< {} tokens", ntok);
        }
        for token in tokens {
            self.advance(token);
        }
        if ntok > 1 && LOG_ADVANCE {
            println!(">>>");
        }

        self.finish_states();

        if let StepSpecific::Stop = &self.curr_state().specific {
            PostProcessResult::stop()
        } else {
            PostProcessResult::continue_()
        }
    }

    fn pre_process(&mut self, _arg: PreProcessArg) -> PreProcessResult {
        self.finish_states();

        if self.maybe_wait() {
            return PreProcessResult::suspend();
        }

        // moving to Fork state is greedy
        if self.can_move_to_next_state() {
            if let Some(StepSpecific::Fork { .. }) = self.next_state() {
                self.state_idx += 1;
            }
        }

        if let StepSpecific::Fork { branches } = &self.curr_state().specific {
            let attention_masks = branches
                .iter()
                .map(|b| b[0].attention_mask(&self.ctx))
                .collect::<Vec<_>>();
            PreProcessResult::new(attention_masks.len())
        } else {
            let mask = self.curr_state().attention_mask(&self.ctx);
            PreProcessResult::new(vec![mask].len())
        }
    }

    fn mid_process(&mut self, arg: MidProcessArg) -> MidProcessResult {
        self.finish_states();

        if arg.fork_group.len() > 1 {
            println!("fork group: {:?}", arg.fork_group);
            if self.state_idx == 0 && !self.curr_state().specific.is_fork() {
                println!("initial fork; nothing to see here");
            } else {
                let st = self.states.remove(self.state_idx);
                if let StepSpecific::Fork { mut branches } = st.specific {
                    assert!(arg.fork_group.len() == branches.len());
                    let my_id = aici_abi::self_seq_id();
                    let idx = arg.fork_group.iter().position(|id| *id == my_id).unwrap();
                    let branch = branches.remove(idx);
                    self.states.splice(self.state_idx..self.state_idx, branch);
                } else {
                    panic!("current step not a fork");
                }
            }
        }

        if self.maybe_wait() {
            // this is a bit late in the game, but it's the best we can do
            MidProcessResult::Splice {
                backtrack: 0,
                ff_tokens: tokenize(" "),
            }

            // // we pop the useless generated token
            // MidProcessResult::Splice {
            //     backtrack: 1,
            //     ff_tokens: vec![],
            // }
        } else {
            self.try_backtrack()
        }
    }
}

fn main() {
    aici_abi::cfg::cfg_test().unwrap();
    //    let _run = sample_prog();
}

fn runner_from_env() -> Runner {
    let a = aici_abi::arg_bytes();
    match serde_json::from_slice(&a) {
        Ok(p) => Runner::new(p),
        Err(e) => {
            let mut col = e.column().saturating_sub(1);
            let mut line = e.line().saturating_sub(1);
            for off in 0..a.len() {
                if line == 0 {
                    col -= 1;
                    if col == 0 {
                        println!(
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
            println!("JSON AST parsing error: {:?}", e);
            panic!()
        }
    }
}

aici_expose_all!(Runner, runner_from_env());
