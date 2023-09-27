use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::time::{Duration, Instant};

use anyhow::Result;
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::{dense, Automaton};
use regex_automata::util::primitives::StateID;
use regex_automata::util::syntax;
use regex_automata::Input;
use serde::Serialize;
use serde_json::{json, Value};
use tokenizers::Tokenizer;

const NO_TOKEN: TokenId = 0;
const NO_STATE: u32 = 0;
const DEAD_STATE: u32 = 1;
const START_STATE: u32 = 4;

fn json_to_regex_inner(json: &Value) -> String {
    let strrx = r#""(\\(["\\\/bfnrt]|u[a-fA-F0-9]{4})|[^"\\\x00-\x1F\x7F]+)*""#;
    match json {
        Value::Bool(_) => r#"(true|false)"#.into(),
        Value::Number(_) => r#"\d+"#.into(),
        Value::String(s) => {
            if s == "" {
                strrx.into()
            } else {
                format!("\"({})\"", s)
            }
        }
        Value::Array(_) => r#"\[.*\]"#.into(),
        Value::Object(obj) => {
            String::from(r#"\{\s*"#)
                + &obj
                    .iter()
                    .map(|(k, v)| format!("\"{0}\"\\s*:\\s*{1}", k, json_to_regex_inner(v)))
                    .collect::<Vec<_>>()
                    .join("\\s*,\\s*")
                + r#"\s*\}"#
        }
        Value::Null => r#"null"#.into(),
    }
}

fn json_to_regex(json: &Value) -> String {
    format!("\\s*{}\\s*", json_to_regex_inner(json))
}

struct TimeLog {
    start: Instant,
    prev: Instant,
    times: Vec<(String, Duration)>,
}

impl TimeLog {
    pub fn new() -> Self {
        let now = Instant::now();
        TimeLog {
            start: now,
            prev: now,
            times: Vec::new(),
        }
    }
    pub fn save(&mut self, id: &str) {
        self.times.push((String::from(id), self.prev.elapsed()));
        self.prev = Instant::now();
    }
}

impl fmt::Display for TimeLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut copy = self.times.clone();
        copy.push((String::from("final"), self.prev.elapsed()));
        copy.push((String::from("TOTAL"), self.start.elapsed()));
        for (l, d) in &copy {
            write!(f, "{:8.1}ms {}\n", d.as_micros() as f64 / 100.0, l)?
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let mut times = TimeLog::new();

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

    let rx = json_to_regex(&json!({
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

    times.save("serialize");

    let compiled = TokenCompiled {
        token_data,
        state_data,
    };

    let mut logits = ctx.tokens.iter().map(|_| 0.0).collect::<Vec<_>>();
    compiled.compute_logit_bias(START_STATE, &mut logits);
    let new_state = compiled.advance(START_STATE, 123);
    compiled.compute_logit_bias(new_state, &mut logits);

    let s = serde_json::to_string_pretty(&ctx.states.values().collect::<Vec<_>>())?;
    std::fs::write("automaton.json", s)?;

    println!("\ntimes:\n{}", times);

    // for s in ctx.states.values() {
    //     for t in &s.transitions {
    //         println!("{:?} {}", t.target, t.tokens.id);
    //     }
    // }

    Ok(())
}

struct TokenCompiled {
    token_data: Vec<u16>,
    state_data: Vec<u32>,
}

impl TokenCompiled {
    fn token_in_token_set(self: &TokenCompiled, token: TokenId, set: u32) -> bool {
        assert!(token != NO_TOKEN);
        let mut idx = set as usize;
        loop {
            let v = self.token_data[idx];
            if v == token {
                return true;
            }
            if v == NO_TOKEN {
                return false;
            }
            idx = idx + 1;
        }
    }

    fn state_bias(state: u32) -> f32 {
        if state == DEAD_STATE {
            -100.0
        } else {
            0.0
        }
    }

    fn compute_logit_bias(self: &TokenCompiled, state_offset: u32, bias: &mut [f32]) {
        let mut p = state_offset as usize;
        let default_state = self.state_data[p];
        p += 1;

        let init_val = Self::state_bias(default_state);
        for idx in 0..bias.len() {
            bias[idx] = init_val;
        }

        loop {
            let state = self.state_data[p];
            if state == NO_STATE {
                break;
            }
            p += 1;
            let toks = self.state_data[p];
            p += 1;
            let val = Self::state_bias(state);

            let mut idx = toks as usize;
            loop {
                let tok = self.token_data[idx];
                if tok == NO_TOKEN {
                    break;
                }
                bias[tok as usize] = val;
                idx = idx + 1;
            }
        }
    }

    fn advance(self: &TokenCompiled, state_offset: u32, token: TokenId) -> u32 {
        let mut p = state_offset as usize;
        let default_state = self.state_data[p];
        p += 1;
        loop {
            let state = self.state_data[p];
            if state == NO_STATE {
                return default_state;
            }
            p += 1;
            let toks = self.state_data[p];
            p += 1;
            if self.token_in_token_set(token, toks) {
                return state;
            }
        }
    }
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
struct TokenSet {
    id: u32,
}

impl Serialize for TokenSet {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u32(self.id)
    }
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
struct MyState {
    id: usize,
}

impl Serialize for MyState {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u64(self.id as u64)
    }
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

#[derive(Debug, Serialize)]
struct TokenTransition {
    target: MyState,
    tokens: TokenSet,
}

#[derive(Debug, Serialize)]
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
