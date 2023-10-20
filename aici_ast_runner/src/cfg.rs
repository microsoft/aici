use std::{cell::RefCell, collections::HashMap, rc::Rc, vec};

use aici_abi::{
    toktree::{Recognizer, SpecialToken},
    wprintln,
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

struct Lexer {
    dfa: dense::DFA<Vec<u32>>,
    patterns: Vec<String>,
    skip_patterns: Vob,
    friendly_pattern_names: Vec<String>,
    possible_by_state: HashMap<StateID, vob::Vob>,
    initial: StateID,
}

impl Lexer {
    pub fn from(
        patterns: Vec<String>,
        skip_patterns: Vob,
        friendly_pattern_names: Vec<String>,
    ) -> Self {
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
            patterns.len()
        );

        let anch = regex_automata::Anchored::Yes;

        let mut incoming = HashMap::new();
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
        let mut tokenset_by_state = HashMap::new();

        for s in &states {
            let mut v = vob![false; patterns.len()];
            if dfa.is_match_state(*s) {
                for idx in 0..dfa.match_len(*s) {
                    let idx = dfa.match_pattern(*s, idx).as_usize();
                    v.set(idx, true);
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

            wprintln!("iter {} {}", num_set, states.len());
            if num_set == 0 {
                break;
            }
        }

        if false {
            wprintln!(
                "tokenset_by_state: {:?}",
                tokenset_by_state.get(&dfa.next_state(initial, b'a'))
            );
        }

        println!("visited: {:?}", tokenset_by_state.len());

        Lexer {
            dfa,
            patterns,
            skip_patterns,
            friendly_pattern_names,
            possible_by_state: tokenset_by_state,
            initial,
        }
    }

    fn possible_tokens(&self, state: StateID) -> &Vob {
        self.possible_by_state.get(&state).unwrap()
    }

    fn get_token(&self, state: StateID) -> Option<PatIdx> {
        assert!(self.dfa.is_match_state(state));

        // we take the first token that matched
        // (eg., "while" will match both keyword and identifier, but keyword is first)
        let pat_idx = (0..self.dfa.match_len(state))
            .map(|idx| self.dfa.match_pattern(state, idx).as_usize())
            .min()
            .unwrap();

        if true {
            wprintln!("token: {}", self.friendly_pattern_names[pat_idx]);
        }

        if self.skip_patterns[pat_idx] {
            // whitespace, comment, etc.
            None
        } else {
            Some(pat_idx)
        }
    }

    fn advance(&self, prev: StateID, byte: Option<u8>) -> Option<(StateID, Option<PatIdx>)> {
        let dfa = &self.dfa;
        if let Some(byte) = byte {
            let state = dfa.next_state(prev, byte);
            wprintln!("state: {:?} {:?} {:?}", prev, byte as char, state);
            if dfa.is_dead_state(dfa.next_eoi_state(state)) {
                let final_state = dfa.next_eoi_state(prev);
                // if final_state is a match state, find the token that matched
                if dfa.is_match_state(final_state) {
                    let tok = self.get_token(final_state);
                    let state = dfa.next_state(self.initial, byte);
                    Some((state, tok))
                } else {
                    None
                }
            } else {
                Some((state, None))
            }
        } else {
            let final_state = dfa.next_eoi_state(prev);
            if dfa.is_match_state(final_state) {
                let tok = self.get_token(final_state);
                Some((self.initial, tok))
            } else {
                None
            }
        }
    }
}

pub struct CfgParser {
    grm: YaccGrammar<StorageT>,
    stable: StateTable<StorageT>,
    lexer: Lexer,
    byte_states: Vec<ByteState>,
    pat_idx_to_tidx: Vec<TIdx<u32>>,
    possible_tokens_by_state: RefCell<HashMap<StIdx<u32>, Vob>>,
    tidx_to_pat_idx: HashMap<TIdx<u32>, usize>,
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

        let mut tidx_to_pat_idx = HashMap::new();
        for (idx, _tok) in patterns.iter().enumerate() {
            tidx_to_pat_idx.insert(pat_idx_to_tidx[idx], idx);
            // wprintln!("tok: {:?} {:?}", tok, tidx[idx]);
        }

        let mut skip_patterns = vob![false; patterns.len()];
        let mut friendly_pattern_names = patterns.clone();

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

        let dfa = Lexer::from(patterns, skip_patterns, friendly_pattern_names);
        let byte_state = ByteState {
            lexer_state: dfa.initial,
            parse_stack: Rc::new(vec![stable.start_state()]),
            viable: vob![true; dfa.patterns.len()],
        };
        CfgParser {
            grm,
            stable,
            lexer: dfa,
            byte_states: vec![byte_state],
            pat_idx_to_tidx,
            tidx_to_pat_idx,
            possible_tokens_by_state: RefCell::new(HashMap::new()),
        }
    }

    fn viable_tokens(&self, stidx: StIdx<StorageT>) -> Vob {
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

        self.possible_tokens_by_state
            .borrow_mut()
            .insert(stidx, r.clone());
        r
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

            wprintln!(
                "tidx: {:?} {:?} {:?}",
                self.friendly_token_name(lexeme),
                pstack,
                act
            );

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
        wprintln!("advance: {:?} {:?}", top.lexer_state, byte,);
        let (info, res) = match self.lexer.advance(top.lexer_state, byte) {
            // Error?
            None => ("lex-err", None),
            // Just new state, no token
            Some((state, None)) => (
                "lex",
                self.mk_byte_state(state, top.parse_stack.clone(), top.viable.clone()),
            ),
            // New state and token generated
            Some((state, Some(pat_idx))) => ("parse", self.run_parser(pat_idx, top, state)),
        };
        wprintln!(
            "push: {:?} -> {} {}",
            if let Some(b) = byte {
                (b as char).to_string()
            } else {
                "<EOF>".to_string()
            },
            info,
            if res.is_none() { "error" } else { "ok" }
        );
        res
    }

    fn run_parser(&self, pat_idx: usize, top: &ByteState, state: StateID) -> Option<ByteState> {
        let tidx = self.pat_idx_to_tidx[pat_idx];
        let mut pstack = (*top.parse_stack).clone();
        match self.parse_lexeme(tidx, &mut pstack) {
            ParseResult::Accept => panic!("accept non EOF?"),
            ParseResult::Continue => {
                let stidx = *pstack.last().unwrap();
                let viable = self.viable_tokens(stidx);
                self.mk_byte_state(state, Rc::new(pstack), viable)
            }
            ParseResult::Error => None,
        }
    }

    fn mk_byte_state(
        &self,
        state: StateID,
        pstack: Rc<PStack<StorageT>>,
        mut viable: Vob,
    ) -> Option<ByteState> {
        let lextoks = self.lexer.possible_tokens(state);
        self.print_viable("v", &viable);
        self.print_viable("lex", lextoks);
        viable &= lextoks;
        if vob_is_zero(&viable) {
            None
        } else {
            Some(ByteState {
                lexer_state: state,
                parse_stack: pstack,
                viable,
            })
        }
    }
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
    viable: Vob,
}

impl Recognizer for CfgParser {
    /*

    state: DFA state, set of viable tokens, LR(1) stack

    push(byte):
        prev = state
        state = state.next(byte)
        if dead(state):
            tok = matches(prev)
            if tok != white space:
                LR(1) <- tok
            state = state0.next(byte)
            viable = possible_tokens(state) & (viable(LR(1)) | {white space})
        else
            viable = viable & possible_tokens(state)
            if viable is empty
                reject
            else
                continue

    */

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
    let grm = include_bytes!("../c.y");
    let mut cfg = CfgParser::from(&String::from_utf8_lossy(grm));

    let sample = include_bytes!("../sample.c");

    for b in sample {
        wprintln!("b: '{}'", *b as char);
        let r = cfg.try_push_byte(*b);
        if !r {
            wprintln!("error");
            break;
        }
    }

    if cfg.special_allowed(SpecialToken::EndOfSentence) {
        wprintln!("accept EOS");
    } else {
        wprintln!("reject EOS");
    }

    // let mut pstack = Vec::new();
    // pstack.push(stable.start_state());
    // let psr = ParserState {
    //     grm: &grm,
    //     stable: &stable,
    // };

    // let s = "(0+1)*Q2";
    // let mut tokens = s
    //     .char_indices()
    //     .map(|(index, ch)| &s[index..index + ch.len_utf8()])
    //     .map(|chstr| grm.token_idx(chstr).unwrap())
    //     .collect::<Vec<_>>();
    // tokens.push(grm.eof_token_idx());

    // // for tok in tokens {
    // //     let r = psr.parse_lexeme(tok.0, &mut pstack);
    // //     wprintln!("t: {:?} {:?} {:?}", tok, grm.token_name(tok), r);
    // // }

    // let patterns = vec![
    //     r#"foo"#, //
    //     r#"fob"#, //
    //     r#"\w+"#, //
    //     r#"\d+"#, //
    // ];
    // //wprintln!("dfa: {:?}", dfa);
    // let s = "fooXX";
    // let mut state = dfa.universal_start_state(anch).unwrap();
    // for b in s.as_bytes() {
    //     wprintln!("state: {:?} {:?}", state, b);
    //     let state2 = dfa.next_eoi_state(state);
    //     if dfa.is_match_state(state2) {
    //         for idx in 0..dfa.match_len(state2) {
    //             let pat = patterns[dfa.match_pattern(state2, idx).as_usize()];
    //             wprintln!("  match: {}", pat);
    //         }
    //     } else if dfa.is_dead_state(state) {
    //         wprintln!("dead");
    //         break;
    //     }
    //     state = dfa.next_state(state, *b);
    // }

    Ok(())
}
