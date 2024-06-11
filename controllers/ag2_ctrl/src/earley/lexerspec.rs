use aici_abi::{bytes::limit_str, svob::SimpleVob};
use anyhow::Result;
use derivre::{ExprRef, RegexAst, RegexBuilder, RegexVec};
use std::{fmt::Debug, hash::Hash};

#[derive(Clone)]
pub struct LexerSpec {
    pub greedy: bool,
    pub lexemes: Vec<LexemeSpec>,
    regex_builder: RegexBuilder,
}

#[derive(Clone)]
pub struct LexemeSpec {
    pub(crate) idx: LexemeIdx,
    name: String,
    pub(crate) rx: RegexAst,
    compiled_rx: ExprRef,
    ends_at_eos_only: bool,
    contextual: bool,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct LexemeIdx(usize);

impl LexemeIdx {
    pub const SKIP: LexemeIdx = LexemeIdx(0);

    pub fn new(idx: usize) -> Self {
        LexemeIdx(idx)
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }

    pub fn as_u16(&self) -> u16 {
        self.0 as u16
    }
}

// The first byte of EOS_MARKER should not occur in any token,
// other than the token representing this byte itself.
// Once we switch regex engines, we can also use 0xFF,
// as it is not a valid UTF-8 byte, but for now we stick to 0x02,
// which is OK for all tokenizers we use.
pub const EOS_MARKER: &'static str = "\u{02}-EoS";

impl LexemeSpec {
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

impl LexerSpec {
    pub fn new(greedy: bool, skip: RegexAst) -> Result<Self> {
        let mut r = LexerSpec {
            greedy,
            lexemes: Vec::new(),
            regex_builder: RegexBuilder::new(),
        };
        let skip = r.add_lexeme_spec(LexemeSpec {
            name: "SKIP".to_string(),
            rx: skip,
            ..r.empty_spec()
        })?;
        assert!(skip == LexemeIdx::SKIP);
        Ok(r)
    }

    pub fn to_regex_vec(&self) -> RegexVec {
        // TODO
        // Find all non-contextual lexemes that are literals (we call them 'keywords')
        // This assumes that this is the only possible conflict in the lexer that we want to catch.
        // For every non literals lexeme, find all keywords that match it.
        // Replace the regex R for the lexeme with (R & ~(K1|K2|...)) where K1...
        // are the conflicting keywords.
        let rx_list: Vec<_> = self.lexemes.iter().map(|lex| lex.compiled_rx).collect();
        self.regex_builder.to_regex_vec(&rx_list)
    }

    fn add_lexeme_spec(&mut self, mut spec: LexemeSpec) -> Result<LexemeIdx> {
        let compiled = self.regex_builder.mk(&spec.rx)?;
        if let Some(idx) = self
            .lexemes
            .iter()
            .position(|lex| lex.compiled_rx == compiled)
        {
            return Ok(LexemeIdx(idx));
        }
        let idx = LexemeIdx(self.lexemes.len());
        spec.idx = idx;
        spec.compiled_rx = compiled;
        self.lexemes.push(spec);
        Ok(idx)
    }

    fn empty_spec(&self) -> LexemeSpec {
        LexemeSpec {
            idx: LexemeIdx(0),
            name: "".to_string(),
            rx: RegexAst::NoMatch,
            compiled_rx: ExprRef::INVALID,
            ends_at_eos_only: false,
            contextual: false,
        }
    }

    pub fn add_rx_and_stop(
        &mut self,
        name: String,
        body_rx: &str,
        stop_rx: &str,
    ) -> Result<LexemeIdx> {
        let ends_at_eos_only = stop_rx.is_empty();
        let rx = RegexAst::Concat(vec![
            RegexAst::Regex(body_rx.to_string()),
            RegexAst::LookAhead(Box::new(RegexAst::Regex(if ends_at_eos_only {
                EOS_MARKER.to_string()
            } else {
                stop_rx.to_string()
            }))),
        ]);
        self.add_lexeme_spec(LexemeSpec {
            name,
            rx,
            ends_at_eos_only,
            ..self.empty_spec()
        })
    }

    pub fn add_simple_literal(&mut self, name: String, literal: &str) -> Result<LexemeIdx> {
        self.add_lexeme_spec(LexemeSpec {
            name,
            rx: RegexAst::Literal(literal.to_string()),
            ..self.empty_spec()
        })
    }

    pub fn add_greedy_lexeme(
        &mut self,
        name: String,
        rx: &str,
        contextual: bool,
    ) -> Result<LexemeIdx> {
        self.add_lexeme_spec(LexemeSpec {
            name,
            rx: RegexAst::Regex(rx.to_string()),
            contextual,
            ..self.empty_spec()
        })
    }

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

impl Debug for LexerSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LexerSpec {{ greedy: {}, lexemes: [", self.greedy)?;
        for lex in &self.lexemes {
            writeln!(f, "  {:?}", lex)?;
        }
        write!(f, "] }}")
    }
}

#[derive(Clone)]
pub struct Lexeme {
    pub idx: LexemeIdx,
    bytes: Vec<u8>,
    hidden_len: usize,
}

impl Lexeme {
    pub fn new(idx: LexemeIdx, bytes: Vec<u8>, hidden_len: usize) -> Self {
        Lexeme {
            idx,
            bytes,
            hidden_len,
        }
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
