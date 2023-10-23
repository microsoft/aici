use std::{cell::RefCell, hash::Hash, rc::Rc, vec};

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
use regex_automata::util::primitives::StateID;
use rustc_hash::FxHashMap;
use vob::{vob, Vob};

use crate::lex::{Lexer, VobIdx, VobSet};

type StorageT = u32;
type PStack<StorageT> = Vec<StIdx<StorageT>>; // Parse stack

const LOG_PARSER: bool = false;

#[derive(Debug, Clone, Copy)]
enum ParseResult {
    Accept,
    Error,
    Continue,
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
    parse_stacks: Vec<Vec<StIdx<u32>>>,
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

        let parse_stacks = vec![vec![stable.start_state()]];

        let byte_state = ByteState {
            lexer_state: dfa.file_start,
            parse_stack_idx: PStackIdx(0),
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
            parse_stacks,
            vobset,
            stats: RefCell::new(CfgStats {
                yacc_actions: 0,
                states_pushed: 0,
            }),
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

            if LOG_PARSER {
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
    #[inline(always)]
    fn try_push(&mut self, byte: Option<u8>) -> Option<ByteState> {
        let top = self.byte_states.last().unwrap().clone();
        if LOG_PARSER {
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
                self.mk_byte_state(state, v, top.parse_stack_idx, top.viable),
            ),
            // New state and token generated
            Some((state, v, Some(pat_idx))) => ("parse", self.run_parser(pat_idx, &top, state, v)),
        };
        if LOG_PARSER {
            wprintln!(
                " -> {} {}",
                info,
                if res.is_none() { "error" } else { "ok" }
            );
        }
        res
    }

    fn pstack_for(&self, top: &ByteState) -> &PStack<StorageT> {
        &self.parse_stacks[top.parse_stack_idx.0]
    }

    fn push_pstack(&mut self, top: &ByteState, pstack: Vec<StIdx<u32>>) -> PStackIdx {
        let new_idx = PStackIdx(top.parse_stack_idx.0 + 1);
        if self.parse_stacks.len() <= new_idx.0 {
            self.parse_stacks.push(Vec::new());
        }
        self.parse_stacks[new_idx.0] = pstack;
        new_idx
    }

    fn run_parser(
        &mut self,
        pat_idx: usize,
        top: &ByteState,
        state: StateID,
        v: VobIdx,
    ) -> Option<ByteState> {
        {
            let mut s = self.stats.borrow_mut();
            s.yacc_actions += 1;
        }
        if LOG_PARSER {
            wprintln!();
        }
        let pstack = self.pstack_for(top);
        if self.lexer.skip_patterns[pat_idx] {
            let stidx = *pstack.last().unwrap();
            let viable = self.viable_vobidx(stidx);
            //self.print_viable("reset", &viable);
            if LOG_PARSER {
                wprintln!("parse: {:?} skip", pstack);
            }
            // reset viable states - they have been narrowed down to SKIP
            self.mk_byte_state(state, v, top.parse_stack_idx, viable)
        } else {
            let tidx = self.pat_idx_to_tidx[pat_idx];
            let mut pstack = pstack.clone();
            match self.parse_lexeme(tidx, &mut pstack) {
                ParseResult::Accept => panic!("accept non EOF?"),
                ParseResult::Continue => {
                    let stidx = *pstack.last().unwrap();
                    let viable = self.viable_vobidx(stidx);
                    let new_idx = self.push_pstack(top, pstack);
                    self.mk_byte_state(state, v, new_idx, viable)
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
        pstack: PStackIdx,
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
                parse_stack_idx: pstack,
                viable,
            })
        }
    }
}

#[derive(Clone, Copy)]
struct PStackIdx(usize);

#[derive(Clone)]
struct ByteState {
    lexer_state: StateID,
    parse_stack_idx: PStackIdx,
    viable: VobIdx,
}

impl Recognizer for CfgParser {
    fn pop_bytes(&mut self, num: usize) {
        self.byte_states.truncate(self.byte_states.len() - num);
    }

    fn collapse(&mut self) {
        let final_state = self.byte_states.pop().unwrap();
        self.byte_states.clear();
        self.byte_states.push(final_state);
    }

    fn special_allowed(&mut self, tok: SpecialToken) -> bool {
        match tok {
            SpecialToken::EndOfSentence => {
                if let Some(st) = self.try_push(None) {
                    let tidx = self.grm.eof_token_idx();
                    let mut pstack = self.pstack_for(&st).clone();
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

    #[cfg(not(target_arch = "wasm32"))]
    let t0 = std::time::Instant::now();

    for tok in &toks[0..1000] {
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

    #[cfg(not(target_arch = "wasm32"))]
    wprintln!("time: {:?} ", t0.elapsed());

    wprintln!("stats:  {}", cfg.get_stats());

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
