use std::collections::HashMap;

use aici_abi::wprintln;
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
type PStack<StorageT> = Vec<StIdx<StorageT>>; // Parse stack

#[derive(Debug, Clone, Copy)]
enum ParseResult {
    Accept,
    Error,
    Continue,
}

struct DfaInfo {
    dfa: dense::DFA<Vec<u32>>,
    patterns: Vec<String>,
    skip_patterns: Vob,
    friendly_pattern_names: Vec<String>,
    tokenset_by_state: HashMap<StateID, vob::Vob>,
}

impl DfaInfo {
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
        let s0 = dfa.universal_start_state(anch).unwrap();
        let mut todo = vec![s0];
        incoming.insert(s0, Vec::new());
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

        wprintln!(
            "tokenset_by_state: {:?}",
            tokenset_by_state.get(&dfa.next_state(s0, b'a'))
        );

        println!("visited: {:?}", tokenset_by_state.len());

        DfaInfo {
            dfa,
            patterns,
            skip_patterns,
            friendly_pattern_names,
            tokenset_by_state,
        }
    }
}

struct GrammarInfo {
    grm: YaccGrammar<StorageT>,
    stable: StateTable<StorageT>,
    dfa: DfaInfo,
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

impl GrammarInfo {
    pub fn from(yacc: &str) -> Self {
        let grm = YaccGrammar::new(
            YaccKind::Original(cfgrammar::yacc::YaccOriginalActionKind::NoAction),
            yacc,
        )
        .unwrap();
        let (sgraph, stable) = from_yacc(&grm, Minimiser::Pager).unwrap();

        if false {
            wprintln!("core\n{}\n\n", sgraph.pp(&grm, true));
            for pidx in grm.iter_pidxs() {
                let prod = grm.prod(pidx);
                wprintln!("{:?} -> {}", prod, prod.len());
            }
        }

        let mut tidx = grm
            .iter_tidxs()
            .filter(|tidx| grm.token_name(*tidx).is_some())
            .collect::<Vec<_>>();

        tidx.sort_by_key(|tidx| {
            let name = grm.token_name(*tidx).unwrap();
            let l = name.len() as isize;
            if is_rx(name) {
                -l + 100000
            } else {
                -l
            }
        });

        let patterns = tidx
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

        let mut pat_idx_by_tidx = HashMap::new();
        for (idx, _tok) in patterns.iter().enumerate() {
            pat_idx_by_tidx.insert(tidx[idx], idx);
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
                    let idx = *pat_idx_by_tidx.get(&tidx).unwrap();
                    friendly_pattern_names[idx] = rname.to_string();
                    if rname == "SKIP" {
                        skip_patterns.set(idx, true);
                    }
                }
            }
        }

        wprintln!("patterns: {:?}", friendly_pattern_names);

        let dfa = DfaInfo::from(patterns, skip_patterns, friendly_pattern_names);
        GrammarInfo { grm, stable, dfa }
    }

    fn parse_lexeme(&self, lexeme: StorageT, pstack: &mut PStack<StorageT>) -> ParseResult {
        loop {
            let stidx = *pstack.last().unwrap();
            let la_tidx = TIdx(lexeme);

            match self.stable.action(stidx, la_tidx) {
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
}

pub fn cfg_test() -> Result<()> {
    let grm = include_bytes!("../c.y");
    let _ = GrammarInfo::from(&String::from_utf8_lossy(grm));

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
