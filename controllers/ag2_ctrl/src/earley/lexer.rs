use aici_abi::svob::SimpleVob;
use regex_automata::{
    dfa::{dense, Automaton},
    util::syntax,
};
use rustc_hash::FxHashMap;
use std::{hash::Hash, rc::Rc, vec};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct LexemeIdx(pub usize);
pub type StateID = regex_automata::util::primitives::StateID;

#[derive(Clone)]
pub struct Lexeme {
    pub idx: LexemeIdx,
    pub bytes: Vec<u8>,
    pub hidden_len: usize,
}

impl Lexeme {
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

const DEBUG: bool = false;

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
    pub fn new() -> Self {
        VobSet {
            vobs: Vec::new(),
            by_vob: FxHashMap::default(),
        }
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

#[derive(Debug, Clone)]
pub struct LexemeSpec {
    pub rx: String,
    pub allow_others: bool,
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
    #[allow(dead_code)]
    patterns: Vec<LexemeSpec>,
    initial: StateID,
    info_by_state_off: Vec<StateInfo>,
    spec: LexerSpec,
    vobset: Rc<VobSet>,
}

pub struct LexerSpec {
    pub greedy: bool,
}

impl Lexer {
    pub fn from(spec: LexerSpec, patterns: Vec<LexemeSpec>, mut vobset: VobSet) -> Self {
        // TIME: 4ms
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
        for p in &patterns {
            debug!("  {}", p.rx)
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
                    debug!("  match: {:?} {}", *s, patterns[idx].rx)
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

            let state = *k;
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
            patterns,
        };

        if DEBUG {
            for s in &states {
                if lex.is_dead(*s) {
                    debug!("dead: {:?} {}", s, lex.dfa.is_dead_state(*s));
                }
            }

            debug!("reachable: {:#?}", reachable_patterns);
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
    ) -> Option<(StateID, Option<LexemeIdx>)> {
        let dfa = &self.dfa;
        let state = dfa.next_state(prev, byte);
        debug!(
            "lex: {:?} -{:?}-> {:?} d={}",
            prev,
            byte as char,
            state,
            self.is_dead(state),
        );
        let info = self.state_info(state);
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
                debug!("lex0: {:?} -{:?}-> {:?}", self.initial, byte as char, state);
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

pub fn quote_regex(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '\\' | '+' | '*' | '?' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '.' | '|' => {
                out.push_str("\\")
            }
            _ => {}
        }
        out.push(c);
    }
    out
}

fn anchored_start() -> regex_automata::util::start::Config {
    regex_automata::util::start::Config::new().anchored(regex_automata::Anchored::Yes)
}
