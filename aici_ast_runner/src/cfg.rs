use std::{cell::RefCell, hash::Hash, rc::Rc, time::Instant, vec};

use aici_abi::{
    toktree::{Recognizer, SpecialToken, TokTrie},
    wprint, wprintln,
};
use anyhow::Result;
use cfgrammar::{
    yacc::{YaccGrammar, YaccKind},
    Symbol, TIdx,
};
use lrtable::{from_yacc, Action, Minimiser, StIdx, StateTable};
use regex_automata::{
    dfa::{dense, Automaton},
    util::{primitives::StateID, syntax},
};
use rustc_hash::FxHashMap;
use vob::{vob, Vob};

type StorageT = u32;
type PatIdx = usize;
type PStack<StorageT> = Vec<StIdx<StorageT>>; // Parse stack

#[derive(Debug, Clone, Copy)]
enum ParseResult {
    Accept,
    Error,
    Continue,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
struct VobIdx(usize);

impl VobIdx {
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

struct VobSet {
    vobs: Vec<Vob>,
    by_vob: FxHashMap<Vob, VobIdx>,
    non_empty: Vob,
}

impl VobSet {
    pub fn new() -> Self {
        VobSet {
            vobs: Vec::new(),
            by_vob: FxHashMap::default(),
            non_empty: Vob::new(),
        }
    }

    pub fn get(&mut self, vob: &Vob) -> VobIdx {
        if let Some(idx) = self.by_vob.get(vob) {
            return *idx;
        }
        let len = self.vobs.len();
        if len == 0 && !vob_is_zero(vob) {
            panic!("first vob must be empty");
        }
        let idx = VobIdx(len);
        self.vobs.push(vob.clone());
        self.by_vob.insert(vob.clone(), idx);
        idx
    }

    pub fn and_is_zero(&self, a: VobIdx, b: VobIdx) -> bool {
        // vob_and_is_zero(&self.vobs[a.0], &self.vobs[b.0])
        !self.non_empty[a.0 * self.vobs.len() + b.0]
    }

    pub fn pre_compute(&mut self) {
        let l = self.vobs.len();
        self.non_empty.resize(l * l, false);
        for x in 0..self.vobs.len() {
            for y in 0..=x {
                if !vob_and_is_zero(&self.vobs[x], &self.vobs[y]) {
                    self.non_empty.set(x * l + y, true);
                    self.non_empty.set(y * l + x, true);
                }
            }
        }
        wprintln!(
            "vobset: {} vobs, {} nonempty",
            self.vobs.len(),
            self.non_empty.len()
        );
    }
}

struct Lexer {
    dfa: dense::DFA<Vec<u32>>,
    skip_patterns: Vob,
    friendly_pattern_names: Vec<String>,
    possible_by_state: FxHashMap<StateID, VobIdx>,
    initial: StateID,
    file_start: StateID,
    logging: bool,
    vobidx_by_state_off: Vec<VobIdx>,
}

impl Lexer {
    pub fn from(
        patterns: Vec<String>,
        skip_patterns: Vob,
        friendly_pattern_names: Vec<String>,
        vobset: &mut VobSet,
    ) -> Self {
        let logging = false;
        let dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .start_kind(regex_automata::dfa::StartKind::Anchored)
                    .match_kind(regex_automata::MatchKind::All),
            )
            .syntax(syntax::Config::new().unicode(false).utf8(false))
            .build_many(&patterns)
            .unwrap();

        wprintln!(
            "dfa: {} bytes, {} patterns",
            dfa.memory_usage(),
            patterns.len(),
        );
        if false {
            for p in &patterns {
                wprintln!("  {}", p)
            }
        }

        let anch = regex_automata::Anchored::Yes;

        let mut incoming = FxHashMap::default();
        let initial = dfa.universal_start_state(anch).unwrap();
        let mut todo = vec![initial];
        incoming.insert(initial, Vec::new());
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
        let mut tokenset_by_state = FxHashMap::default();

        for s in &states {
            let mut v = vob![false; patterns.len()];
            let s2 = dfa.next_eoi_state(*s);
            if dfa.is_match_state(s2) {
                for idx in 0..dfa.match_len(s2) {
                    let idx = dfa.match_pattern(s2, idx).as_usize();
                    v.set(idx, true);
                    if logging {
                        wprintln!("  match: {:?} {}", *s, patterns[idx])
                    }
                }
            }
            tokenset_by_state.insert(*s, v);
        }

        loop {
            let mut num_set = 0;

            for s in &states {
                let ours = tokenset_by_state.get(s).unwrap().clone();
                for o in &incoming[s] {
                    let theirs = tokenset_by_state.get(o).unwrap();
                    let mut tmp = ours.clone();
                    tmp |= theirs;
                    if tmp != *theirs {
                        num_set += 1;
                        tokenset_by_state.insert(*o, tmp);
                    }
                }
            }

            if logging {
                wprintln!("iter {} {}", num_set, states.len());
            }
            if num_set == 0 {
                break;
            }
        }

        let mut states_idx = states.iter().map(|x| x.as_usize()).collect::<Vec<_>>();
        states_idx.sort();

        let shift = dfa.stride2();
        let mut vobidx_by_state_off =
            vec![VobIdx(0); 1 + (states_idx.iter().max().unwrap() >> shift)];
        for (k, v) in tokenset_by_state.iter() {
            vobidx_by_state_off[k.as_usize() >> shift] = vobset.get(v);
        }

        // pretend we've just seen a newline at the beginning of the file
        // TODO: this should be configurable
        let file_start = dfa.next_state(initial, b'\n');
        wprintln!(
            "initial: {:?} {:?}; {} states",
            initial,
            file_start,
            states.len()
        );

        let lex = Lexer {
            dfa,
            skip_patterns,
            friendly_pattern_names,
            vobidx_by_state_off,
            possible_by_state: tokenset_by_state
                .iter()
                .map(|(k, v)| (k.clone(), vobset.get(v)))
                .collect(),
            initial,
            file_start,
            logging,
        };

        if logging {
            for s in &states {
                if lex.is_dead(*s) {
                    wprintln!("dead: {:?} {}", s, lex.dfa.is_dead_state(*s));
                }
            }

            wprintln!("possible_tokens: {:#?}", lex.possible_by_state);
        }

        lex
    }

    fn is_dead(&self, state: StateID) -> bool {
        self.possible_tokens(state).is_zero()
    }

    fn possible_tokens(&self, state: StateID) -> VobIdx {
        self.vobidx_by_state_off[state.as_usize() >> self.dfa.stride2()]
        // *self.possible_by_state.get(&state).unwrap()
    }

    fn get_token(&self, state: StateID) -> Option<PatIdx> {
        if !self.dfa.is_match_state(state) {
            return None;
        }

        // we take the first token that matched
        // (eg., "while" will match both keyword and identifier, but keyword is first)
        let pat_idx = (0..self.dfa.match_len(state))
            .map(|idx| self.dfa.match_pattern(state, idx).as_usize())
            .min()
            .unwrap();

        if self.logging {
            wprintln!("token: {}", self.friendly_pattern_names[pat_idx]);
        }

        Some(pat_idx)
    }

    fn advance(
        &self,
        prev: StateID,
        byte: Option<u8>,
    ) -> Option<(StateID, VobIdx, Option<PatIdx>)> {
        let dfa = &self.dfa;
        if let Some(byte) = byte {
            let state = dfa.next_state(prev, byte);
            if self.logging {
                wprintln!(
                    "lex: {:?} -{:?}-> {:?} d={}",
                    prev,
                    byte as char,
                    state,
                    self.is_dead(state),
                );
            }
            let v = self.possible_tokens(state);
            if v.is_zero() {
                let final_state = dfa.next_eoi_state(prev);
                // if final_state is a match state, find the token that matched
                let tok = self.get_token(final_state);
                if tok.is_none() {
                    None
                } else {
                    let state = dfa.next_state(self.initial, byte);
                    if self.logging {
                        wprintln!("lex0: {:?} -{:?}-> {:?}", self.initial, byte as char, state);
                    }
                    Some((state, self.possible_tokens(state), tok))
                }
            } else {
                Some((state, v, None))
            }
        } else {
            let final_state = dfa.next_eoi_state(prev);
            let tok = self.get_token(final_state);
            if tok.is_none() {
                None
            } else {
                Some((self.initial, self.possible_tokens(self.initial), tok))
            }
        }
    }
}

#[allow(dead_code)]
fn to_index<I, T>(iter: I) -> FxHashMap<T, usize>
where
    I: IntoIterator<Item = T>,
    T: Eq + Hash,
{
    let mut map = FxHashMap::default();
    for item in iter {
        let idx = map.len();
        map.entry(item).or_insert(idx);
    }
    map
}

struct CfgStats {
    yacc_actions: usize,
    states_pushed: usize,
}

pub struct CfgParser {
    grm: YaccGrammar<StorageT>,
    stable: StateTable<StorageT>,
    lexer: Lexer,
    byte_states: Vec<ByteState>,
    pat_idx_to_tidx: Vec<TIdx<u32>>,
    possible_tokens_by_state: RefCell<FxHashMap<StIdx<u32>, Rc<Vob>>>,
    possible_vob_idx_by_state: FxHashMap<StIdx<u32>, VobIdx>,
    vobset: VobSet,
    stats: RefCell<CfgStats>,
    tidx_to_pat_idx: FxHashMap<TIdx<u32>, usize>,
    logging: bool,
}

fn is_rx(name: &str) -> bool {
    name.len() > 2 && name.starts_with("/") && name.ends_with("/")
}

fn quote_rx(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ('0' <= ch && ch <= '9')
                || ('a' <= ch && ch <= 'z')
                || ('A' <= ch && ch <= 'Z')
                || '<' == ch
                || '>' == ch
            {
                ch.to_string()
            } else {
                format!("\\{}", ch)
            }
        })
        .collect::<String>()
}

impl CfgParser {
    pub fn from(yacc: &str) -> Self {
        let grmkind = YaccKind::Original(cfgrammar::yacc::YaccOriginalActionKind::NoAction);
        let grm = YaccGrammar::new(grmkind, yacc).unwrap();
        let (sgraph, stable) = from_yacc(&grm, Minimiser::Pager).unwrap();

        if false {
            wprintln!("core\n{}\n\n", sgraph.pp(&grm, true));
            for pidx in grm.iter_pidxs() {
                let prod = grm.prod(pidx);
                wprintln!("{:?} -> {}", prod, prod.len());
            }
        }

        let mut pat_idx_to_tidx = grm
            .iter_tidxs()
            .filter(|tidx| grm.token_name(*tidx).is_some())
            .collect::<Vec<_>>();

        pat_idx_to_tidx.sort_by_key(|tidx| {
            let name = grm.token_name(*tidx).unwrap();
            let l = name.len() as isize;
            if is_rx(name) {
                -l + 100000
            } else {
                -l
            }
        });

        let patterns = pat_idx_to_tidx
            .iter()
            .map(|tok| {
                let name = grm.token_name(*tok).unwrap();
                if is_rx(name) {
                    name[1..name.len() - 1].to_string()
                } else {
                    quote_rx(name)
                }
            })
            .collect::<Vec<_>>();

        let mut tidx_to_pat_idx = FxHashMap::default();
        for (idx, _tok) in patterns.iter().enumerate() {
            tidx_to_pat_idx.insert(pat_idx_to_tidx[idx], idx);
        }

        let mut skip_patterns = vob![false; patterns.len()];
        let mut friendly_pattern_names = pat_idx_to_tidx
            .iter()
            .map(|tok| grm.token_name(*tok).unwrap().to_string())
            .collect::<Vec<_>>();

        for ridx in grm.iter_rules() {
            let rname = grm.rule_name_str(ridx);
            if rname.to_uppercase() != rname {
                continue;
            }
            for pidx in grm.rule_to_prods(ridx) {
                let toks = grm.prod(*pidx);
                if let [Symbol::Token(tidx)] = toks {
                    let idx = *tidx_to_pat_idx.get(&tidx).unwrap();
                    friendly_pattern_names[idx] = rname.to_string();
                    if rname == "SKIP" {
                        skip_patterns.set(idx, true);
                    }
                }
            }
        }

        wprintln!("patterns: {:?}", friendly_pattern_names);

        let mut vobset = VobSet::new();
        // all-zero has to be inserted first
        let _all0 = vobset.get(&vob![false; patterns.len()]);
        let all1 = vobset.get(&vob![true; patterns.len()]);

        let dfa = Lexer::from(patterns, skip_patterns, friendly_pattern_names, &mut vobset);

        let byte_state = ByteState {
            lexer_state: dfa.file_start,
            parse_stack: Rc::new(vec![stable.start_state()]),
            viable: all1,
        };
        let mut cfg = CfgParser {
            grm,
            stable,
            lexer: dfa,
            byte_states: vec![byte_state],
            pat_idx_to_tidx,
            tidx_to_pat_idx,
            possible_tokens_by_state: RefCell::new(FxHashMap::default()),
            possible_vob_idx_by_state: FxHashMap::default(),
            vobset,
            stats: RefCell::new(CfgStats {
                yacc_actions: 0,
                states_pushed: 0,
            }),
            logging: false,
        };

        cfg.possible_vob_idx_by_state = sgraph
            .iter_stidxs()
            .map(|stidx| {
                (
                    stidx.clone(),
                    cfg.vobset.get(&(*cfg.viable_tokens(stidx)).clone()),
                )
            })
            .collect();

        cfg.vobset.pre_compute();

        cfg
    }

    fn viable_vobidx(&self, stidx: StIdx<StorageT>) -> VobIdx {
        *self.possible_vob_idx_by_state.get(&stidx).unwrap()
    }

    fn viable_tokens(&self, stidx: StIdx<StorageT>) -> Rc<Vob> {
        {
            let tmp = self.possible_tokens_by_state.borrow();
            let r = tmp.get(&stidx);
            if let Some(r) = r {
                return r.clone();
            }
        }

        // skip patterns (whitespace) are always viable
        let mut r = self.lexer.skip_patterns.clone();
        for tidx in self.stable.state_actions(stidx) {
            match self.stable.action(stidx, tidx) {
                Action::Error => {}
                _ => {
                    if let Some(pat_idx) = self.tidx_to_pat_idx.get(&tidx) {
                        r.set(*pat_idx, true);
                    }
                }
            }
        }

        let rr = Rc::new(r);
        self.possible_tokens_by_state
            .borrow_mut()
            .insert(stidx, rr.clone());
        rr
    }

    #[allow(dead_code)]
    fn friendly_token_name(&self, lexeme: TIdx<StorageT>) -> &str {
        if let Some(pidx) = self.tidx_to_pat_idx.get(&lexeme) {
            &self.lexer.friendly_pattern_names[*pidx]
        } else if self.grm.eof_token_idx() == lexeme {
            return "<EOF>";
        } else {
            return "<???>";
        }
    }

    fn parse_lexeme(&self, lexeme: TIdx<StorageT>, pstack: &mut PStack<StorageT>) -> ParseResult {
        loop {
            let stidx = *pstack.last().unwrap();

            let act = self.stable.action(stidx, lexeme);

            if self.logging {
                wprintln!(
                    "parse: {:?} {:?} -> {:?}",
                    pstack,
                    self.friendly_token_name(lexeme),
                    act
                );
            }

            match act {
                Action::Reduce(pidx) => {
                    let ridx = self.grm.prod_to_rule(pidx);
                    let pop_idx = pstack.len() - self.grm.prod(pidx).len();
                    pstack.drain(pop_idx..);
                    let prior = *pstack.last().unwrap();
                    pstack.push(self.stable.goto(prior, ridx).unwrap());
                }
                Action::Shift(state_id) => {
                    pstack.push(state_id);
                    return ParseResult::Continue;
                }
                Action::Accept => {
                    // only happens when lexeme is EOF
                    return ParseResult::Accept;
                }
                Action::Error => {
                    return ParseResult::Error;
                }
            }
        }
    }

    #[allow(dead_code)]
    fn print_viable(&self, lbl: &str, vob: &Vob) {
        wprintln!("viable tokens {}:", lbl);
        for (idx, b) in vob.iter().enumerate() {
            if b {
                wprintln!("  {}: {}", idx, self.lexer.friendly_pattern_names[idx]);
            }
        }
    }

    // None means EOF
    fn try_push(&self, byte: Option<u8>) -> Option<ByteState> {
        let top = self.byte_states.last().unwrap();
        if self.logging {
            wprint!("try_push: ");
            if let Some(b) = byte {
                wprint!("{:?}", b as char)
            } else {
                wprint!("<EOF>")
            }
        }
        let (info, res) = match self.lexer.advance(top.lexer_state, byte) {
            // Error?
            None => ("lex-err", None),
            // Just new state, no token - the hot path
            Some((state, v, None)) => (
                "lex",
                self.mk_byte_state(state, v, &top.parse_stack, top.viable),
            ),
            // New state and token generated
            Some((state, v, Some(pat_idx))) => ("parse", self.run_parser(pat_idx, top, state, v)),
        };
        if self.logging {
            wprintln!(
                " -> {} {}",
                info,
                if res.is_none() { "error" } else { "ok" }
            );
        }
        res
    }

    fn run_parser(
        &self,
        pat_idx: usize,
        top: &ByteState,
        state: StateID,
        v: VobIdx,
    ) -> Option<ByteState> {
        {
            let mut s = self.stats.borrow_mut();
            s.yacc_actions += 1;
        }
        if self.logging {
            wprintln!();
        }
        if self.lexer.skip_patterns[pat_idx] {
            let stidx = *top.parse_stack.last().unwrap();
            let viable = self.viable_vobidx(stidx);
            //self.print_viable("reset", &viable);
            if self.logging {
                wprintln!("parse: {:?} skip", top.parse_stack);
            }
            // reset viable states - they have been narrowed down to SKIP
            self.mk_byte_state(state, v, &top.parse_stack, viable)
        } else {
            let tidx = self.pat_idx_to_tidx[pat_idx];
            let mut pstack = (*top.parse_stack).clone();
            match self.parse_lexeme(tidx, &mut pstack) {
                ParseResult::Accept => panic!("accept non EOF?"),
                ParseResult::Continue => {
                    let stidx = *pstack.last().unwrap();
                    let viable = self.viable_vobidx(stidx);
                    self.mk_byte_state(state, v, &Rc::new(pstack), viable)
                }
                ParseResult::Error => None,
            }
        }
    }

    pub fn get_stats(&self) -> String {
        let mut s = self.stats.borrow_mut();
        let r = format!("yacc: {}/{}", s.yacc_actions, s.states_pushed);
        s.yacc_actions = 0;
        s.states_pushed = 0;
        r
    }

    fn mk_byte_state(
        &self,
        state: StateID,
        lextoks: VobIdx,
        pstack: &Rc<PStack<StorageT>>,
        viable: VobIdx,
    ) -> Option<ByteState> {
        {
            let mut s = self.stats.borrow_mut();
            s.states_pushed += 1;
        }
        // let lextoks = self.lexer.possible_tokens(state);
        if self.vobset.and_is_zero(viable, lextoks) {
            None
        } else {
            Some(ByteState {
                lexer_state: state,
                parse_stack: pstack.clone(),
                viable,
            })
        }
    }
}

fn vob_and_is_zero(a: &Vob, b: &Vob) -> bool {
    assert!(a.len() == b.len());
    for (a, b) in a.iter_storage().zip(b.iter_storage()) {
        if a & b != 0 {
            return false;
        }
    }
    return true;
}

fn vob_is_zero(v: &Vob) -> bool {
    for b in v.iter_storage() {
        if b != 0 {
            return false;
        }
    }
    true
}

struct ByteState {
    lexer_state: StateID,
    parse_stack: Rc<PStack<StorageT>>,
    viable: VobIdx,
}

impl Recognizer for CfgParser {
    fn push_byte(&mut self, byte: u8) {
        let st = self.try_push(Some(byte)).unwrap();
        self.byte_states.push(st)
    }

    fn pop_bytes(&mut self, num: usize) {
        self.byte_states.truncate(self.byte_states.len() - num);
    }

    fn collapse(&mut self) {
        let final_state = self.byte_states.pop().unwrap();
        self.byte_states.clear();
        self.byte_states.push(final_state);
    }

    fn byte_allowed(&self, byte: u8) -> bool {
        let st = self.try_push(Some(byte));
        st.is_some()
    }

    fn special_allowed(&self, tok: SpecialToken) -> bool {
        match tok {
            SpecialToken::EndOfSentence => {
                if let Some(st) = self.try_push(None) {
                    let tidx = self.grm.eof_token_idx();
                    let mut pstack = (*st.parse_stack).clone();
                    match self.parse_lexeme(tidx, &mut pstack) {
                        ParseResult::Accept => true,
                        _ => false,
                    }
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn trie_finished(&mut self) {
        assert!(self.byte_states.len() == 1);
    }

    fn try_push_byte(&mut self, byte: u8) -> bool {
        if let Some(st) = self.try_push(Some(byte)) {
            self.byte_states.push(st);
            true
        } else {
            false
        }
    }
}

pub fn cfg_test() -> Result<()> {
    let yacc_bytes = include_bytes!("../c.y");
    let mut cfg = CfgParser::from(&String::from_utf8_lossy(yacc_bytes));
    let mut rng = aici_abi::rng::Rng::new(0);

    let trie = TokTrie::from_host();
    let sample = include_bytes!("../sample.c");
    let toks = trie.greedy_tokenize(sample);

    let mut logits = trie.alloc_logits();
    let t0 = Instant::now();

    for tok in &toks[0..300] {
        let tok = *tok;
        trie.compute_bias(&mut cfg, &mut logits);
        if false {
            wprintln!(
                "tok: {:?} {}; {}",
                trie.token_str(tok),
                logits[tok as usize],
                cfg.get_stats()
            );
        }
        trie.append_token(&mut cfg, tok);
    }

    wprintln!("time: {:?} {}", t0.elapsed(), cfg.get_stats());

    if false {
        let mut ok = true;
        let mut idx = 0;
        while idx < sample.len() {
            let b = sample[idx];
            // wprintln!("idx {} {:?}", idx, b as char);
            let r = cfg.try_push_byte(b);
            if !r {
                ok = false;
                wprintln!(
                    "reject at\n{:?}\n{:?}",
                    String::from_utf8_lossy(&sample[idx.saturating_sub(50)..idx]),
                    String::from_utf8_lossy(&sample[idx..std::cmp::min(idx + 30, sample.len())])
                );
                break;
            }
            idx += 1;

            let max_pop = cfg.byte_states.len() - 1;
            if max_pop > 0 && rng.gen_up_to(4) == 0 {
                let num = rng.gen_up_to(max_pop - 1) + 1;
                // wprintln!("pop {} {}", num, cfg.byte_states.len());
                cfg.pop_bytes(num);
                idx -= num;
            }

            if rng.gen_up_to(10) == 0 {
                // wprintln!("collapse");
                cfg.collapse();
            }
        }

        if ok {
            if cfg.special_allowed(SpecialToken::EndOfSentence) {
                wprintln!("accept EOS");
            } else {
                wprintln!("reject EOS");
            }
        } else {
            wprintln!("reject");
        }
    }

    Ok(())
}
