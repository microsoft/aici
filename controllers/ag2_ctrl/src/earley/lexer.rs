use aici_abi::{bytes::limit_str, svob::SimpleVob};
use anyhow::Result;
use derivre::{ExprRef, RegexAst, RegexBuilder, RegexVec, StateDesc};
use std::{fmt::Debug, hash::Hash, rc::Rc};

use super::vobset::VobSet;

const DEBUG: bool = true;

macro_rules! debug {
    ($($arg:tt)*) => {
        if DEBUG {
            println!($($arg)*);
        }
    }
}

pub struct Lexer {
    dfa: RegexVec,
    spec: LexerSpec,
    vobset: Rc<VobSet>,
}

#[derive(Clone)]
pub struct LexerSpec {
    pub greedy: bool,
    pub lexemes: Vec<LexemeSpec>,
}

#[derive(Clone)]
pub struct LexemeSpec {
    pub(crate) idx: LexemeIdx,
    name: String,
    rx: RegexAst,
    compiled_rx: ExprRef,
    ends_at_eos_only: bool,
    contextual: bool,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct LexemeIdx(pub usize);
pub type StateID = derivre::StateID;

#[derive(Debug, Clone, Copy)]
pub struct PreLexeme {
    pub idx: LexemeIdx,
    pub byte: Option<u8>,
    pub hidden_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum LexerResult {
    Lexeme(PreLexeme),
    State(StateID, u8),
    Error,
}

#[derive(Clone)]
pub struct Lexeme {
    pub idx: LexemeIdx,
    bytes: Vec<u8>,
    hidden_len: usize,
}

impl LexemeSpec {
    // The first byte of EOS_MARKER should not occur in any token,
    // other than the token representing this byte itself.
    // Once we switch regex engines, we can also use 0xFF,
    // as it is not a valid UTF-8 byte, but for now we stick to 0x02,
    // which is OK for all tokenizers we use.
    pub const EOS_MARKER: &'static str = "\u{02}-EoS";

    pub fn key(&self) -> String {
        format!("{}:{:?}", self.contextual, self.compiled_rx)
    }

    pub fn compile_rx(&mut self, builder: &mut RegexBuilder) -> Result<()> {
        self.compiled_rx = builder.mk(&self.rx)?;
        Ok(())
    }

    pub fn from_rx_and_stop(name: String, body_rx: &str, stop_rx: &str) -> Result<Self> {
        let ends_at_eos_only = stop_rx.is_empty();
        let rx = RegexAst::Concat(vec![
            RegexAst::Regex(body_rx.to_string()),
            RegexAst::LookAhead(Box::new(RegexAst::Regex(if ends_at_eos_only {
                Self::EOS_MARKER.to_string()
            } else {
                stop_rx.to_string()
            }))),
        ]);
        let info = LexemeSpec {
            idx: LexemeIdx(0),
            name,
            rx,
            compiled_rx: ExprRef::INVALID,
            ends_at_eos_only,
            contextual: false,
        };
        Ok(info)
    }

    pub fn from_simple_literal(name: String, literal: &str) -> Self {
        let info = LexemeSpec {
            idx: LexemeIdx(0),
            name,
            rx: RegexAst::Literal(literal.to_string()),
            compiled_rx: ExprRef::INVALID,
            ends_at_eos_only: false,
            contextual: false,
        };
        info
    }

    pub fn from_greedy_lexeme(name: String, rx: &str, contextual: bool) -> Self {
        let info = LexemeSpec {
            idx: LexemeIdx(0),
            name,
            rx: RegexAst::Regex(rx.to_string()),
            compiled_rx: ExprRef::INVALID,
            ends_at_eos_only: false,
            contextual,
        };
        info
    }

    /// Check if the lexeme always matches bytes, and has at least one more byte to spare.
    pub fn has_forced_bytes(&self, bytes: &[u8]) -> bool {
        match &self.rx {
            RegexAst::Literal(s) if s.len() > bytes.len() => &s.as_bytes()[0..bytes.len()] == bytes,
            _ => false,
        }
    }
}

impl Debug for LexemeSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {} {:?}", self.idx.0, self.name, self.rx)?;
        if self.ends_at_eos_only {
            write!(f, " eos-only")?;
        }
        if self.contextual {
            write!(f, " allow-others")?;
        }
        Ok(())
    }
}

impl Lexeme {
    fn new(spec: &LexemeSpec, bytes: Vec<u8>, hidden_len: usize) -> Result<Self> {
        Ok(Lexeme {
            idx: spec.idx,
            bytes,
            hidden_len,
        })
    }

    pub fn just_idx(idx: LexemeIdx) -> Self {
        Lexeme {
            idx,
            hidden_len: 0,
            bytes: Vec::new(),
        }
    }

    pub fn bogus() -> Self {
        Lexeme::just_idx(LexemeIdx(0))
    }

    pub fn is_bogus(&self) -> bool {
        // TODO?
        self.idx.0 == 0 && self.bytes.is_empty()
    }

    pub fn num_hidden_bytes(&self) -> usize {
        self.hidden_len
    }

    pub fn num_visible_bytes(&self) -> usize {
        self.bytes.len() - self.hidden_len
    }

    pub fn visible_bytes(&self) -> &[u8] {
        &self.bytes[0..self.num_visible_bytes()]
    }

    pub fn hidden_bytes(&self) -> &[u8] {
        &self.bytes[self.num_visible_bytes()..]
    }
}

impl LexerSpec {
    pub fn dbg_lexeme(&self, lex: &Lexeme) -> String {
        let str = String::from_utf8_lossy(&lex.bytes).to_string();
        let info = &self.lexemes[lex.idx.0];
        if matches!(info.rx, RegexAst::Literal(_)) && lex.hidden_len == 0 {
            format!("[{}]", info.name)
        } else {
            format!(
                "[{}] match={:?} hidden={}",
                info.name,
                limit_str(&str, 32),
                lex.hidden_len
            )
        }
    }

    pub fn dbg_lexeme_set(&self, vob: &SimpleVob) -> String {
        format!(
            "Lexemes: [{}]",
            vob.iter()
                .map(|idx| self.lexemes[idx as usize].name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    pub fn new_lexeme(&self, idx: LexemeIdx, bytes: Vec<u8>, hidden_len: usize) -> Result<Lexeme> {
        Lexeme::new(&self.lexemes[idx.0], bytes, hidden_len)
    }

    pub fn lexeme_spec(&self, idx: LexemeIdx) -> &LexemeSpec {
        &self.lexemes[idx.0]
    }

    pub fn eos_lexemes(&self) -> SimpleVob {
        let mut v = SimpleVob::alloc(self.lexemes.len());
        for (idx, lex) in self.lexemes.iter().enumerate() {
            if lex.ends_at_eos_only {
                v.set(idx, true);
            }
        }
        v
    }
}

impl Lexer {
    pub fn from(spec: LexerSpec) -> Result<Self> {
        let patterns = &spec.lexemes;
        let vobset = VobSet::new(patterns.len());
        let mut builder = RegexBuilder::new();
        let refs = patterns
            .iter()
            .map(|p| builder.mk(&p.rx))
            .collect::<Result<Vec<_>>>()?;
        let dfa = builder.to_regex_vec(&refs);

        println!("dfa: {:?}", dfa);
        for p in patterns {
            debug!("  pattern: {:?}", p)
        }

        let lex = Lexer {
            dfa,
            vobset: Rc::new(vobset),
            spec,
        };

        Ok(lex)
    }

    pub fn vobset(&self) -> &VobSet {
        &self.vobset
    }

    pub fn start_state(&mut self, allowed_lexemes: &SimpleVob, first_byte: Option<u8>) -> StateID {
        let s = self.dfa.initial_state(allowed_lexemes);
        first_byte.map(|b| self.dfa.transition(s, b)).unwrap_or(s)
    }

    pub fn a_dead_state(&self) -> StateID {
        StateID::DEAD
    }

    pub fn possible_hidden_len(&mut self, state: StateID) -> usize {
        self.dfa.possible_lookahead_len(state)
    }

    fn state_info(&self, state: StateID) -> &StateDesc {
        self.dfa.state_desc(state)
    }

    pub fn allows_eos(&mut self, state: StateID, allowed_eos_lexemes: &SimpleVob) -> bool {
        if allowed_eos_lexemes.is_zero() {
            return false;
        }

        let state = self
            .dfa
            .transition_bytes(state, LexemeSpec::EOS_MARKER.as_bytes());

        let accepting = &self.dfa.state_desc(state).accepting;
        if accepting.and_is_zero(allowed_eos_lexemes) {
            false
        } else {
            true
        }
    }

    pub fn force_lexeme_end(&self, prev: StateID) -> LexerResult {
        let info = self.state_info(prev);
        let idx = info.possible.first_bit_set().expect("no allowed lexemes");
        LexerResult::Lexeme(PreLexeme {
            idx: LexemeIdx(idx),
            byte: None,
            hidden_len: 0,
        })
    }

    #[inline(always)]
    pub fn advance(&mut self, prev: StateID, byte: u8, enable_logging: bool) -> LexerResult {
        let state = self.dfa.transition(prev, byte);

        if enable_logging {
            let info = self.state_info(state);
            debug!(
                "lex: {:?} -{:?}-> {:?}, acpt={}",
                prev, byte as char, state, info.lowest_accepting
            );
        }

        if state.is_dead() {
            if !self.spec.greedy {
                return LexerResult::Error;
            }

            let info = self.dfa.state_desc(prev);
            // we take the first token that matched
            // (eg., "while" will match both keyword and identifier, but keyword is first)
            if info.is_accepting() {
                LexerResult::Lexeme(PreLexeme {
                    idx: LexemeIdx::from_state_desc(info),
                    byte: Some(byte),
                    hidden_len: self.dfa.possible_lookahead_len(prev),
                })
            } else {
                LexerResult::Error
            }
        } else {
            let info = self.state_info(state);
            if !self.spec.greedy && info.is_accepting() {
                LexerResult::Lexeme(PreLexeme {
                    idx: LexemeIdx::from_state_desc(info),
                    byte: Some(byte),
                    hidden_len: self.dfa.possible_lookahead_len(state),
                })
            } else {
                LexerResult::State(state, byte)
            }
        }
    }
}

fn is_regex_special(b: char) -> bool {
    match b {
        '\\' | '+' | '*' | '?' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '.' | '|' => true,
        _ => false,
    }
}

pub fn quote_regex(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if is_regex_special(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

impl LexemeIdx {
    fn from_state_desc(desc: &StateDesc) -> Self {
        assert!(desc.lowest_accepting >= 0);
        LexemeIdx(desc.lowest_accepting as usize)
    }
}

impl LexerResult {
    #[inline(always)]
    pub fn is_error(&self) -> bool {
        matches!(self, LexerResult::Error)
    }
}
