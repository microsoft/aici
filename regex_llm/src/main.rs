mod jsonrx;
mod timelog;

use anyhow::Result;
use gvm_abi::rx::{StateOffset, TokRx, TokRxInfo};
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::{dense, Automaton};
use regex_automata::util::primitives::StateID;
use regex_automata::util::syntax;
use regex_automata::Input;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use tokenizers::Tokenizer;

const NO_TOKEN: TokenId = 0;
const NO_STATE: u32 = StateOffset::NONE.off;
const DEAD_STATE: u32 = StateOffset::DEAD.off;
const START_STATE: u32 = StateOffset::START.off;

fn main() -> Result<()> {
    let mut times = timelog::TimeLog::new();

    let tok = Tokenizer::from_file("tokenizer.json").unwrap();
    times.save("tokenizer");

    let nvocab = tok.get_vocab_size(true);

    let mut tokens = Vec::new();

    let repl = "\u{FFFD}";

    for idx in 0..nvocab {
        let arr = vec![idx as u32];
        if 3 <= idx && idx <= 258 {
            tokens.push(Vec::from(vec![(idx - 3) as u8]));
        } else {
            let str = tok.decode(&arr, false).unwrap();
            if str.contains(repl) {
                println!("{} {}", idx, str); // TODO
            }
            // assert!(!str.contains(repl));
            tokens.push(Vec::from(str.as_bytes()));
        }
    }

    times.save("prep");

    // println!("toks: {:?} {}", &tokens, c as u32);

    let rx = jsonrx::json_to_regex(&json!({
        "name": "",
        "valid": true,
        "type": "foo|bar|baz|something|else",
        "address": {
            "street": "",
            "city": "",
            "state": "[A-Z][A-Z]"
        },
        "age": 1
    }));
    println!("rx: {}", rx);

    let dfa = dense::Builder::new()
        .configure(dense::Config::new().start_kind(regex_automata::dfa::StartKind::Anchored))
        .syntax(syntax::Config::new().unicode(false).utf8(false))
        .build(&rx)?;

    times.save("rx");

    if false {
        let j1 = json!({
            "name": "123\u{0000}4",
            "valid": true,
            "type": "something",
            "address": {
                "street": "Foobar",
                "city": "Something",
                "state": "WA"
            },
            "age": 1
        });
        let s1 = serde_json::to_string(&j1)?;
        let s2 = serde_json::to_string_pretty(&j1)?;
        let s3 = format!("x{}", s2);
        let r1 = dfa.try_search_fwd(&Input::new(&s1).anchored(regex_automata::Anchored::Yes))?;
        let r2 = dfa.try_search_fwd(&Input::new(&s2).anchored(regex_automata::Anchored::Yes))?;
        let r3 = dfa.try_search_fwd(&Input::new(&s3).anchored(regex_automata::Anchored::Yes))?;
        println!("{} {:#?} {:#?} {:#?}", s2, r1, r2, r3);
    }

    let f = format!("{:#?}", dfa);
    let mut n = 0;
    for c in f.chars() {
        if c == '\n' {
            n += 1;
        }
    }

    times.save("rx-stats");

    println!("dfa: {} bytes, ~{} states", dfa.memory_usage(), n - 20);

    let state0 = dfa
        .universal_start_state(regex_automata::Anchored::Yes)
        .unwrap();

    let mut ctx = Ctx {
        tokens,
        dfa,
        token_sets: HashMap::new(),
        states: HashMap::new(),
        token_set_offsets: HashMap::new(),
        state_offsets: HashMap::new(),
    };
    compute_next_rec(&mut ctx, state0);

    times.save("compile");

    let ntransitions: usize = ctx.states.values().map(|v| v.transitions.len()).sum();
    let ntokens: usize = ctx.token_sets.keys().map(|t| t.len()).sum();

    println!(
        "size: {} transitions with {} tokens (over {} sets)",
        ntransitions,
        ntokens,
        ctx.token_sets.len()
    );

    let mut token_data = Vec::new();
    token_data.push(0);

    for (ts, id) in &ctx.token_sets {
        ctx.token_set_offsets.insert(*id, token_data.len() as u32);
        for t in ts {
            token_data.push(*t);
        }
        token_data.push(0);
    }

    let state0 = MyState::from(state0);
    let mut keys = ctx.states.keys().map(|x| x.clone()).collect::<Vec<_>>();
    keys.sort_by_key(|x| {
        if x.id == 0 {
            -2
        } else if *x == state0 {
            -1
        } else {
            x.id as i32
        }
    });

    assert!(keys[0].id == 0);
    // let dead = ctx.states.get(&keys[0]).unwrap();
    // assert!(dead.transitions.len() == 0);
    // assert!(dead.size() as u32 == START_STATE - DEAD_STATE);
    assert!(keys[1] == state0);

    let mut off = 1;
    for k in &keys {
        let s = ctx.states.get(&k).unwrap();
        ctx.state_offsets.insert(*k, off);
        off += s.size() as u32;
    }

    assert!(ctx.state_offset(keys[0]) == DEAD_STATE);
    assert!(ctx.state_offset(state0) == START_STATE);

    let mut state_data = Vec::with_capacity(off as usize);
    state_data.push(0);
    for k in &keys {
        let s = ctx.states.get(&k).unwrap();
        s.write(&ctx, &mut state_data);
    }

    let info = TokRxInfo {
        tok_eos: tok.token_to_id("</s>").unwrap() as u16, // TODO
    };
    let bytes = TokRx::serialize(&info, &token_data, &state_data);
    println!("size: {} bytes", bytes.len());
    std::fs::write("rx.bin", bytes)?;

    times.save("serialize");

    println!("\ntimes:\n{}", times);

    // for s in ctx.states.values() {
    //     for t in &s.transitions {
    //         println!("{:?} {}", t.target, t.tokens.id);
    //     }
    // }

    Ok(())
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
struct TokenSet {
    id: u32,
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
struct MyState {
    id: usize,
}

impl MyState {
    fn from(s: StateID) -> Self {
        MyState { id: s.as_usize() }
    }
}

struct Ctx {
    tokens: Vec<Vec<u8>>,
    dfa: DFA<Vec<u32>>,
    token_sets: HashMap<Vec<TokenId>, TokenSet>,
    states: HashMap<MyState, TokenState>,
    state_offsets: HashMap<MyState, u32>,
    token_set_offsets: HashMap<TokenSet, u32>,
}

type TokenId = u16;

#[derive(Debug)]
struct TokenTransition {
    target: MyState,
    tokens: TokenSet,
}

#[derive(Debug)]
struct TokenState {
    this_state: MyState,
    default_transition: MyState,
    transitions: Vec<TokenTransition>,
}

impl Ctx {
    fn state_offset(&self, s: MyState) -> u32 {
        *self.state_offsets.get(&s).unwrap()
    }
    fn token_offset(&self, s: TokenSet) -> u32 {
        *self.token_set_offsets.get(&s).unwrap()
    }
}

impl TokenState {
    fn size(&self) -> usize {
        1 + 2 * (self.transitions.len() + 1)
    }

    fn write(&self, ctx: &Ctx, target: &mut Vec<u32>) {
        let off = ctx.state_offset(self.this_state) as usize;
        assert!(off == target.len());
        target.push(ctx.state_offset(self.default_transition));
        for t in &self.transitions {
            target.push(ctx.state_offset(t.target));
            target.push(ctx.token_offset(t.tokens));
        }
        target.push(0);
        target.push(0);
        assert!(target.len() - off == self.size());
    }
}

fn compute_next_rec(ctx: &mut Ctx, state0: StateID) {
    let state0 = MyState::from(state0);
    let mut visited = HashSet::new();
    let mut pending = Vec::new();
    pending.push(state0);
    while pending.len() > 0 {
        let s = pending.pop().unwrap();
        let n = compute_next(ctx, StateID::new(s.id).unwrap());
        for t in &n.transitions {
            if !visited.contains(&t.target) {
                visited.insert(t.target.clone());
                pending.push(t.target);
            }
        }
        ctx.states.insert(s, n);
    }
}

fn compute_next(ctx: &mut Ctx, state0: StateID) -> TokenState {
    let mut tbystate = HashMap::<StateID, Vec<TokenId>>::new();
    for (idx, bytes) in ctx.tokens.iter().enumerate() {
        let token = idx as TokenId;
        if bytes.len() == 0 {
            continue;
        }
        let mut s = state0;
        for b in bytes.iter() {
            s = ctx.dfa.next_state(s, *b);
            if ctx.dfa.is_dead_state(s) {
                break;
            }
        }
        tbystate.entry(s).or_insert_with(Vec::new).push(token);
    }
    let default_transition = if let Some((state, _)) = tbystate.iter().max_by_key(|(_, v)| v.len())
    {
        let state = *state;
        tbystate.remove(&state);
        MyState::from(state)
    } else {
        panic!("no transitions in {:?}", state0)
    };
    let transitions = tbystate
        .iter()
        .map(|(k, v)| {
            let num = ctx.token_sets.len() as u32;
            let e = ctx
                .token_sets
                .entry(v.clone())
                .or_insert(TokenSet { id: num });
            TokenTransition {
                target: MyState::from(*k),
                tokens: e.clone(),
            }
        })
        .collect::<Vec<_>>();
    TokenState {
        this_state: MyState::from(state0),
        default_transition,
        transitions,
    }
}
