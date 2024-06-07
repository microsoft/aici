use aici_abi::{bytes::limit_str, svob::SimpleVob};
use anyhow::Result;
use derivre::{RegexVec, StateDesc};
use regex::bytes::Regex;
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
    rx: String,
    simple_text: Option<String>,
    ends_at_eos_only: bool,
    allow_others: bool,
    hidden: HiddenLexeme,
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

#[derive(Debug, Clone)]
enum HiddenLexeme {
    Regex(Regex),
    Fixed(usize),
}

impl HiddenLexeme {
    pub fn from_rx(full_rx: &str, stop_rx: &str) -> Result<Self> {
        match guess_regex_len(stop_rx.as_bytes()) {
            Some(len) => Ok(HiddenLexeme::Fixed(len)),
            None => Ok(HiddenLexeme::Regex(Regex::new(full_rx)?)),
        }
    }
}

impl Default for HiddenLexeme {
    fn default() -> Self {
        HiddenLexeme::Fixed(0)
    }
}

impl LexemeSpec {
    // The first byte of EOS_MARKER should not occur in any token,
    // other than the token representing this byte itself.
    // Once we switch regex engines, we can also use 0xFF,
    // as it is not a valid UTF-8 byte, but for now we stick to 0x02,
    // which is OK for all tokenizers we use.
    pub const EOS_MARKER: &'static str = "\u{02}-EoS";

    pub fn key(&self) -> &str {
        &self.rx
    }

    pub fn from_rx_and_stop(name: String, body_rx: &str, stop_rx: &str) -> Result<Self> {
        let ends_at_eos_only = stop_rx.is_empty();
        let rx = format!(
            "({})({})",
            body_rx,
            if ends_at_eos_only {
                Self::EOS_MARKER
            } else {
                stop_rx
            }
        );
        let hidden = HiddenLexeme::from_rx(&rx, stop_rx)?;
        let info = LexemeSpec {
            idx: LexemeIdx(0),
            name,
            rx,
            simple_text: None,
            ends_at_eos_only,
            allow_others: false,
            hidden,
        };
        Ok(info)
    }

    pub fn from_simple_literal(name: String, literal: &str) -> Self {
        let info = LexemeSpec {
            idx: LexemeIdx(0),
            name,
            rx: quote_regex(literal),
            simple_text: Some(literal.to_string()),
            ends_at_eos_only: false,
            allow_others: false,
            hidden: HiddenLexeme::default(),
        };
        info
    }

    pub fn from_greedy_lexeme(name: String, rx: &str, allow_others: bool) -> Self {
        let info = LexemeSpec {
            idx: LexemeIdx(0),
            name,
            rx: rx.to_string(),
            simple_text: None,
            ends_at_eos_only: false,
            allow_others,
            hidden: HiddenLexeme::default(),
        };
        info
    }

    pub fn has_hidden_len(&self) -> bool {
        match &self.hidden {
            HiddenLexeme::Fixed(0) => false,
            _ => true,
        }
    }

    /// Check if the lexeme always matches bytes, and has at least one more byte to spare.
    pub fn has_forced_bytes(&self, bytes: &[u8]) -> bool {
        match &self.simple_text {
            Some(s) if s.len() > bytes.len() => &s.as_bytes()[0..bytes.len()] == bytes,
            _ => false,
        }
    }
}

impl Debug for LexemeSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {} ", self.idx.0, self.name)?;
        match self.simple_text {
            Some(ref s) => write!(f, "{:?}", limit_str(s, 32))?,
            None => write!(f, "rx:{:?}", limit_str(&self.rx, 32))?,
        }
        if self.ends_at_eos_only {
            write!(f, " eos-only")?;
        }
        if self.allow_others {
            write!(f, " allow-others")?;
        }
        match &self.hidden {
            HiddenLexeme::Fixed(len) => write!(f, " hidden={}", len),
            HiddenLexeme::Regex(_) => write!(f, " hidden=regex"),
        }
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
        if str == info.rx && lex.hidden_len == 0 {
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
        // TIME: 4ms
        let patterns = &spec.lexemes;
        let vobset = VobSet::new(patterns.len());
        let parser = regex_syntax::ParserBuilder::new()
            .dot_matches_new_line(false)
            .unicode(true)
            .utf8(true)
            .build();
        let dfa = RegexVec::new_with_parser(
            parser,
            &patterns.iter().map(|x| x.rx.as_str()).collect::<Vec<_>>(),
        )?;

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

fn guess_regex_len(bytes: &[u8]) -> Option<usize> {
    let mut len = 0;
    let mut idx = 0;
    while idx < bytes.len() {
        let c = bytes[idx] as char;
        match c {
            '\\' => {
                idx += 2;
                len += 1;
            }
            // TODO we could do char classes too; watch for unicode though!
            '.' => {
                // dot is OK
            }
            _ if is_regex_special(c) => {
                return None;
            }
            _ => {}
        }

        len += 1;
        idx += 1;
    }
    Some(len)
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
