use aici_abi::{
    bytes::{limit_bytes, limit_str},
    svob::SimpleVob,
};
use anyhow::{bail, Result};
use regex::bytes::Regex;
use regex_automata::{
    dfa::{dense, Automaton},
    util::syntax,
};
use rustc_hash::FxHashMap;
use std::{fmt::Debug, hash::Hash, rc::Rc, vec};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct LexemeIdx(pub usize);
pub type StateID = regex_automata::util::primitives::StateID;

#[derive(Clone)]
pub struct Lexeme {
    pub idx: LexemeIdx,
    bytes: Vec<u8>,
    hidden_len: usize,
}

#[derive(Debug, Clone)]
pub enum HiddenLexeme {
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

#[derive(Clone)]
pub struct LexemeSpec {
    pub idx: LexemeIdx,
    pub name: String,
    pub rx: String,
    pub ends_at_eos_only: bool,
    pub allow_others: bool,
    pub hidden: HiddenLexeme,
}

impl Debug for LexemeSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} {:?}",
            self.idx.0,
            self.name,
            limit_str(&self.rx, 32),
        )?;
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
    fn new(spec: &LexemeSpec, bytes: Vec<u8>) -> Result<Self> {
        let hidden_len = match &spec.hidden {
            HiddenLexeme::Fixed(len) => *len,
            HiddenLexeme::Regex(re) => {
                if let Some(c) = re.captures(&bytes) {
                    let visible_len = c.get(1).unwrap().end();
                    bytes.len() - visible_len
                } else {
                    bail!(
                        "no match for {} matching {}",
                        spec.name,
                        limit_bytes(&bytes, 100)
                    );
                }
            }
        };
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

    pub fn num_visible_bytes(&self) -> usize {
        self.bytes.len() - self.hidden_len
    }

    pub fn visible_bytes(&self) -> &[u8] {
        &self.bytes[0..self.num_visible_bytes()]
    }
}

const DEBUG: bool = true;

macro_rules! debug {
    ($($arg:tt)*) => {
        if DEBUG {
            println!($($arg)*);
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub struct VobIdx {
    v: u32,
}

impl VobIdx {
    pub fn new(v: usize) -> Self {
        VobIdx { v: v as u32 }
    }

    pub fn all_zero() -> Self {
        VobIdx { v: 0 }
    }

    pub fn as_usize(&self) -> usize {
        self.v as usize
    }

    pub fn is_zero(&self) -> bool {
        self.v == 0
    }
}

pub struct VobSet {
    vobs: Vec<SimpleVob>,
    by_vob: FxHashMap<SimpleVob, VobIdx>,
}

impl VobSet {
    pub fn new(single_vob_size: usize) -> Self {
        let mut r = VobSet {
            vobs: Vec::new(),
            by_vob: FxHashMap::default(),
        };
        let v = SimpleVob::alloc(single_vob_size);
        r.insert_or_get(&v);
        r.insert_or_get(&v.negated());
        r
    }

    pub fn insert_or_get(&mut self, vob: &SimpleVob) -> VobIdx {
        if let Some(idx) = self.by_vob.get(vob) {
            return *idx;
        }
        let len = self.vobs.len();
        if len == 0 && !vob.is_zero() {
            panic!("first vob must be empty");
        }
        let idx = VobIdx::new(len);
        self.vobs.push(vob.clone());
        self.by_vob.insert(vob.clone(), idx);
        idx
    }

    pub fn resolve(&self, idx: VobIdx) -> &SimpleVob {
        &self.vobs[idx.as_usize()]
    }

    pub fn and_is_zero(&self, a: VobIdx, b: VobIdx) -> bool {
        self.vobs[a.as_usize()].and_is_zero(&self.vobs[b.as_usize()])
    }
}

struct StateInfo {
    reachable: VobIdx,
    accepting: VobIdx,
}

impl Default for StateInfo {
    fn default() -> Self {
        StateInfo {
            reachable: VobIdx::all_zero(),
            accepting: VobIdx::all_zero(),
        }
    }
}

pub struct Lexer {
    dfa: dense::DFA<Vec<u32>>,
    initial: StateID,
    info_by_state_off: Vec<StateInfo>,
    spec: LexerSpec,
    vobset: Rc<VobSet>,
}

#[derive(Clone)]
pub struct LexerSpec {
    pub greedy: bool,
    pub lexemes: Vec<LexemeSpec>,
}

impl LexerSpec {
    pub fn dbg_lexeme(&self, lex: &Lexeme) -> String {
        let str = String::from_utf8_lossy(&lex.bytes).to_string();
        let info = &self.lexemes[lex.idx.0];
        if str == info.rx {
            format!("[{}]", info.name)
        } else {
            format!("[{}] match={:?}", info.name, limit_str(&str, 32))
        }
    }

    pub fn dbg_lexeme_set(&self, vob: &SimpleVob) -> String {
        vob.iter()
            .map(|idx| self.lexemes[idx as usize].name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn new_lexeme(&self, idx: LexemeIdx, bytes: Vec<u8>) -> Result<Lexeme> {
        Lexeme::new(&self.lexemes[idx.0], bytes)
    }
}

impl Lexer {
    pub fn from(spec: LexerSpec) -> Self {
        // TIME: 4ms
        let patterns = &spec.lexemes;
        let mut vobset = VobSet::new(patterns.len());
        let dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .start_kind(regex_automata::dfa::StartKind::Anchored)
                    .match_kind(regex_automata::MatchKind::All),
            )
            .syntax(syntax::Config::new().unicode(false).utf8(false))
            .build_many(&patterns.iter().map(|x| &x.rx).collect::<Vec<_>>())
            .unwrap();

        println!(
            "dfa: {} bytes, {} patterns",
            dfa.memory_usage(),
            patterns.len(),
        );
        for p in patterns {
            debug!("  pattern: {:?}", p)
        }

        let mut incoming = FxHashMap::default();
        let initial = dfa
            .start_state(&anchored_start())
            .expect("dfa has no start state");
        let mut todo = vec![initial];
        incoming.insert(initial, Vec::new());

        // TIME: 1.5ms
        while todo.len() > 0 {
            let s = todo.pop().unwrap();
            for b in 0..=255 {
                let s2 = dfa.next_state(s, b);
                if !incoming.contains_key(&s2) {
                    todo.push(s2);
                    incoming.insert(s2, Vec::new());
                }
                incoming.get_mut(&s2).unwrap().push(s);
            }
        }

        let states = incoming.keys().map(|x| *x).collect::<Vec<_>>();
        let mut reachable_patterns = FxHashMap::default();

        for s in &states {
            let mut v = SimpleVob::alloc(patterns.len());
            let s2 = dfa.next_eoi_state(*s);
            if dfa.is_match_state(s2) {
                for idx in 0..dfa.match_len(s2) {
                    let idx = dfa.match_pattern(s2, idx).as_usize();
                    v.set(idx, true);
                    debug!("  match: {:?} {:?}", *s, patterns[idx].rx)
                }
            }
            reachable_patterns.insert(*s, v);
        }

        // TIME: 20ms
        loop {
            let mut num_set = 0;

            for s in &states {
                let ours = reachable_patterns.get(s).unwrap().clone();
                for o in &incoming[s] {
                    let theirs = reachable_patterns.get(o).unwrap();
                    let mut tmp = ours.clone();
                    tmp.or(theirs);
                    if tmp != *theirs {
                        num_set += 1;
                        reachable_patterns.insert(*o, tmp);
                    }
                }
            }

            debug!("iter {} {}", num_set, states.len());
            if num_set == 0 {
                break;
            }
        }

        let mut states_idx = states.iter().map(|x| x.as_usize()).collect::<Vec<_>>();
        states_idx.sort();

        let shift = dfa.stride2();
        let mut info_by_state_off = (0..1 + (states_idx.iter().max().unwrap() >> shift))
            .map(|_| StateInfo::default())
            .collect::<Vec<_>>();
        for (k, v) in reachable_patterns.iter() {
            let state_idx = k.as_usize() >> shift;
            info_by_state_off[state_idx].reachable = vobset.insert_or_get(v);

            let state = dfa.next_eoi_state(*k);
            if dfa.is_match_state(state) {
                let mut accepting = SimpleVob::alloc(patterns.len());
                for pat_idx in 0..dfa.match_len(state) {
                    accepting.set(dfa.match_pattern(state, pat_idx).as_usize(), true)
                }
                info_by_state_off[state_idx].accepting = vobset.insert_or_get(&accepting);
            }
        }

        println!("initial: {:?}; {} states", initial, states.len());

        let lex = Lexer {
            dfa,
            info_by_state_off,
            initial,
            vobset: Rc::new(vobset),
            spec,
        };

        if DEBUG {
            for s in &states {
                if lex.is_dead(*s) {
                    debug!("dead: {:?} {}", s, lex.dfa.is_dead_state(*s));
                }
            }

            // debug!("reachable: {:#?}", reachable_patterns);
        }

        lex
    }

    pub fn vobset(&self) -> &VobSet {
        &self.vobset
    }

    pub fn file_start_state(&self) -> StateID {
        self.initial
        // pretend we've just seen a newline at the beginning of the file
        // TODO: this should be configurable
        // self.dfa.next_state(self.initial.state, b'\n')
    }

    fn state_info(&self, state: StateID) -> &StateInfo {
        &self.info_by_state_off[state.as_usize() >> self.dfa.stride2()]
    }

    fn is_dead(&self, state: StateID) -> bool {
        self.reachable_tokens(state).is_zero()
    }

    fn reachable_tokens(&self, state: StateID) -> VobIdx {
        self.state_info(state).reachable
    }

    fn get_token(&self, prev: StateID, allowed_lexems: &SimpleVob) -> Option<LexemeIdx> {
        let state = self.dfa.next_eoi_state(prev);
        if !self.dfa.is_match_state(state) {
            return None;
        }

        // we take the first token that matched
        // (eg., "while" will match both keyword and identifier, but keyword is first)
        let accepting = self.state_info(state).accepting;

        if accepting.is_zero() {
            return None;
        }

        if let Some(idx) = self
            .vobset
            .resolve(accepting)
            .first_bit_set_here_and_in(allowed_lexems)
        {
            Some(LexemeIdx(idx))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn advance_eof(
        &self,
        prev: StateID,
        allowed_lexems: &SimpleVob,
    ) -> Option<(StateID, Option<LexemeIdx>)> {
        let tok = self.get_token(prev, allowed_lexems);
        if tok.is_none() {
            None
        } else {
            Some((self.initial, tok))
        }
    }

    #[inline(always)]
    pub fn advance(
        &self,
        allowed_lexems: &SimpleVob,
        prev: StateID,
        byte: u8,
        max_tokens_reached: bool,
        definitive: bool,
    ) -> Option<(StateID, Option<LexemeIdx>)> {
        let dfa = &self.dfa;

        if max_tokens_reached {
            let info = self.state_info(prev);
            let idx = if let Some(idx) = self
                .vobset
                .resolve(info.reachable)
                .first_bit_set_here_and_in(allowed_lexems)
            {
                idx
            } else {
                allowed_lexems
                    .first_bit_set()
                    .expect("empty allowed lexemes")
            };
            return Some((self.initial, Some(LexemeIdx(idx))));
        }

        let state = dfa.next_state(prev, byte);
        let info = self.state_info(state);
        if definitive {
            debug!(
                "lex: {:?} -{:?}-> {:?} d={}, acpt={:?}",
                prev,
                byte as char,
                state,
                self.is_dead(state),
                self.vobset
                    .resolve(info.accepting)
                    .first_bit_set_here_and_in(allowed_lexems)
                    .map(|x| self.spec.lexemes[x].name.as_str())
            );
        }
        if self
            .vobset
            .resolve(info.reachable)
            .and_is_zero(allowed_lexems)
        {
            if !self.spec.greedy {
                return None;
            }

            // if final_state is a match state, find the token that matched
            let tok = self.get_token(prev, allowed_lexems);
            if tok.is_none() {
                None
            } else {
                let state = dfa.next_state(self.initial, byte);
                if definitive {
                    debug!("lex0: {:?} -{:?}-> {:?}", self.initial, byte as char, state);
                }
                Some((state, tok))
            }
        } else {
            if !self.spec.greedy && !info.accepting.is_zero() {
                if let Some(idx) = self
                    .vobset
                    .resolve(info.accepting)
                    .first_bit_set_here_and_in(allowed_lexems)
                {
                    return Some((self.initial, Some(LexemeIdx(idx))));
                }
            }

            Some((state, None))
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

fn anchored_start() -> regex_automata::util::start::Config {
    regex_automata::util::start::Config::new().anchored(regex_automata::Anchored::Yes)
}
