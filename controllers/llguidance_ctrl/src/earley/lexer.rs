use aici_abi::svob::SimpleVob;
use anyhow::Result;
use derivre::{NextByte, RegexVec, StateDesc};
use std::fmt::Debug;

use super::lexerspec::{LexemeIdx, LexerSpec, EOS_MARKER};

const DEBUG: bool = true;

macro_rules! debug {
    ($($arg:tt)*) => {
        if cfg!(feature = "logging") && DEBUG {
            println!($($arg)*);
        }
    }
}

#[derive(Clone)]
pub struct Lexer {
    dfa: RegexVec,
    spec: LexerSpec,
}

pub type StateID = derivre::StateID;

#[derive(Debug, Clone, Copy)]
pub struct PreLexeme {
    pub idx: LexemeIdx,
    pub byte: Option<u8>,
    pub byte_next_row: bool,
    pub hidden_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum LexerResult {
    Lexeme(PreLexeme),
    State(StateID, u8),
    Error,
}

impl Lexer {
    pub fn from(spec: &LexerSpec) -> Result<Self> {
        let dfa = spec.to_regex_vec();

        debug!("lexer: {:?}\n  ==> dfa: {:?}", spec, dfa);

        let lex = Lexer {
            dfa,
            spec: spec.clone(), // TODO check perf of Rc<> ?
        };

        Ok(lex)
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

        let state = self.dfa.transition_bytes(state, EOS_MARKER);

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
            idx: LexemeIdx::new(idx),
            byte: None,
            byte_next_row: false,
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
                    byte_next_row: true,
                    hidden_len: self.dfa.possible_lookahead_len(prev),
                })
            } else {
                LexerResult::Error
            }
        } else {
            let can_stop_now = !self.spec.greedy || self.dfa.next_byte(state) == NextByte::ForcedEOI;
            let info = self.state_info(state);
            if can_stop_now && info.is_accepting() {
                LexerResult::Lexeme(PreLexeme {
                    idx: LexemeIdx::from_state_desc(info),
                    byte: Some(byte),
                    byte_next_row: false,
                    hidden_len: self.dfa.possible_lookahead_len(state),
                })
            } else {
                LexerResult::State(state, byte)
            }
        }
    }
}

impl LexemeIdx {
    fn from_state_desc(desc: &StateDesc) -> Self {
        assert!(desc.lowest_accepting >= 0);
        LexemeIdx::new(desc.lowest_accepting as usize)
    }
}

impl LexerResult {
    #[inline(always)]
    pub fn is_error(&self) -> bool {
        matches!(self, LexerResult::Error)
    }
}
