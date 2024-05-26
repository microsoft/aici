use std::fmt::Debug;

use aici_abi::svob::SimpleVob;
use anyhow::Result;

use crate::{
    ast::{ExprRef, ExprSet},
    bytecompress::ByteCompressor,
    deriv::DerivCache,
    hashcons::VecHashMap,
};

const DEBUG: bool = true;

macro_rules! debug {
    ($($arg:tt)*) => {
        if DEBUG {
            eprintln!($($arg)*);
        }
    };
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateID(u32);

impl StateID {
    // DEAD state corresponds to empty vector
    pub const DEAD: StateID = StateID(0);
    // MISSING state corresponds to yet not computed entries in the state table
    pub const MISSING: StateID = StateID(1);

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }

    pub fn is_valid(&self) -> bool {
        *self != Self::MISSING
    }
}

impl Debug for StateID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == StateID::DEAD {
            write!(f, "DEAD")
        } else if *self == StateID::MISSING {
            write!(f, "MISSING")
        } else {
            write!(f, "StateID({})", self.0)
        }
    }
}

pub struct RegexVec {
    cache: DerivCache,
    alphabet_mapping: Vec<u8>,
    alphabet_size: usize,
    rx_list: Vec<ExprRef>,
    rx_sets: VecHashMap,
    state_table: Vec<StateID>,
    state_descs: Vec<StateDesc>,
    num_transitions: usize,
    num_ast_nodes: usize,
}

#[derive(Clone)]
pub struct StateDesc {
    pub lowest_accepting: isize, // -1 if no accepting state
    pub accepting: SimpleVob,
    pub possible: SimpleVob,
}

impl StateDesc {
    pub fn is_accepting(&self) -> bool {
        self.lowest_accepting != -1
    }

    pub fn is_dead(&self) -> bool {
        self.possible.is_zero()
    }
}

// public implementation
impl RegexVec {
    pub fn new_single(rx: &str) -> Result<Self> {
        Self::new_vec(&[rx])
    }

    pub fn new_vec(rx_list: &[&str]) -> Result<Self> {
        let mut parser = regex_syntax::ParserBuilder::new().build();
        Self::new_with_parser(&mut parser, rx_list)
    }

    pub fn new_with_parser(parser: &mut regex_syntax::Parser, rx_list: &[&str]) -> Result<Self> {
        let mut exprset = ExprSet::new(256);
        let mut acc = Vec::new();
        for rx in rx_list {
            let ast = exprset.parse_expr(parser, rx)?;
            acc.push(ast);
        }
        Ok(Self::new_with_exprset(exprset, &acc))
    }

    pub fn initial_state(&mut self, selected: &SimpleVob) -> StateID {
        let mut vec_desc = vec![];
        for idx in selected.iter() {
            Self::push_rx(&mut vec_desc, idx as usize, self.rx_list[idx as usize]);
        }
        self.insert_state(vec_desc)
    }

    pub fn state_desc(&self, state: StateID) -> &StateDesc {
        &self.state_descs[state.as_usize()]
    }

    pub fn transition(&mut self, state: StateID, b: u8) -> StateID {
        let mapped = self.alphabet_mapping[b as usize] as usize;
        let idx = state.as_usize() * self.alphabet_size + mapped;
        let new_state = self.state_table[idx];
        if new_state != StateID::MISSING {
            new_state
        } else {
            let new_state = self.transition_inner(state, b);
            self.num_transitions += 1;
            self.state_table[idx] = new_state;
            new_state
        }
    }

    pub fn transition_bytes(&mut self, state: StateID, bytes: &[u8]) -> StateID {
        let mut state = state;
        for &b in bytes {
            state = self.transition(state, b);
        }
        state
    }

    pub fn is_match(&mut self, text: &str) -> bool {
        let selected = SimpleVob::alloc(self.rx_list.len());
        let mut state = self.initial_state(&selected.negated());
        for b in text.bytes() {
            let new_state = self.transition(state, b);
            debug!("b: {:?} --{:?}--> {:?}", state, b as char, new_state);
            state = new_state;
            if state == StateID::DEAD {
                return false;
            }
        }
        self.state_desc(state).is_accepting()
    }

    /// Estimate the size of the regex tables in bytes.
    pub fn num_bytes(&self) -> usize {
        self.cache.num_bytes()
            + self.state_descs.len() * 100
            + self.state_table.len() * std::mem::size_of::<StateID>()
            + self.rx_sets.num_bytes()
    }

    pub fn stats(&self) -> String {
        format!(
            "regexps: {} with {} nodes (+ {} derived), states: {}; transitions: {}; bytes: {}; alphabet size: {}",
            self.rx_list.len(),
            self.num_ast_nodes,
            self.cache.exprs.len() - self.num_ast_nodes,
            self.state_descs.len(),
            self.num_transitions,
            self.num_bytes(),
            self.alphabet_size
        )
    }
}

// private implementation
impl RegexVec {
    fn new_with_exprset(exprset: ExprSet, rx_list: &[ExprRef]) -> Self {
        assert!(exprset.alphabet_size() == 256);
        let compress = true;

        let ((exprset, rx_list), mapping, alphabet_size) = if compress {
            let mut compressor = ByteCompressor::new();
            (
                compressor.compress(&exprset, rx_list),
                compressor.mapping,
                compressor.alphabet_size,
            )
        } else {
            let alphabet_size = exprset.alphabet_size();
            (
                (exprset, rx_list.to_vec()),
                (0..=255).collect(),
                alphabet_size,
            )
        };

        let num_ast_nodes = exprset.len();

        let mut rx_sets = VecHashMap::new();
        let id = rx_sets.insert(vec![]);
        assert!(id == StateID::DEAD.as_u32());
        let id = rx_sets.insert(vec![0]);
        assert!(id == StateID::MISSING.as_u32());

        let mut r = RegexVec {
            cache: DerivCache::new(exprset),
            alphabet_mapping: mapping,
            rx_list,
            rx_sets,
            alphabet_size,
            state_table: vec![],
            state_descs: vec![],
            num_transitions: 0,
            num_ast_nodes,
        };
        r.insert_state(vec![]);
        // also append state for the "MISSING"
        r.append_state(r.state_descs[0].clone());
        // in fact, transition from MISSING and DEAD should both lead to DEAD
        r.state_table.fill(StateID::DEAD);
        // guard against corner-case
        if r.alphabet_size == 0 {
            r.state_table.push(StateID::DEAD);
        }
        r
    }

    fn append_state(&mut self, state_desc: StateDesc) {
        let mut new_states = vec![StateID::MISSING; self.alphabet_size];
        self.state_table.append(&mut new_states);
        self.state_descs.push(state_desc);
    }

    fn insert_state(&mut self, lst: Vec<u32>) -> StateID {
        // does this help?
        // if lst.len() == 0 {
        //     return StateID::DEAD;
        // }
        assert!(lst.len() % 2 == 0);
        let id = StateID(self.rx_sets.insert(lst));
        if id.as_usize() >= self.state_descs.len() {
            self.append_state(self.compute_state_desc(id));
        }
        id
    }

    fn iter_state(rx_sets: &VecHashMap, state: StateID, mut f: impl FnMut((usize, ExprRef))) {
        let lst = rx_sets.get(state.as_u32()).unwrap();
        for idx in (0..lst.len()).step_by(2) {
            f((lst[idx] as usize, ExprRef::new(lst[idx + 1])));
        }
    }

    fn exprs(&self) -> &ExprSet {
        &self.cache.exprs
    }

    fn compute_state_desc(&self, state: StateID) -> StateDesc {
        let mut res = StateDesc {
            lowest_accepting: -1,
            accepting: SimpleVob::alloc(self.rx_list.len()),
            possible: SimpleVob::alloc(self.rx_list.len()),
        };
        Self::iter_state(&self.rx_sets, state, |(idx, e)| {
            res.possible.set(idx, true);
            if self.exprs().is_nullable(e) {
                res.accepting.set(idx, true);
                if res.lowest_accepting == -1 {
                    res.lowest_accepting = idx as isize;
                }
            }
        });
        if res.possible.is_zero() {
            assert!(state == StateID::DEAD);
        }
        res
    }

    fn push_rx(vec_desc: &mut Vec<u32>, idx: usize, e: ExprRef) {
        vec_desc.push(idx as u32);
        vec_desc.push(e.as_u32());
    }

    fn transition_inner(&mut self, state: StateID, b: u8) -> StateID {
        assert!(state.is_valid());

        let mut vec_desc = vec![];

        Self::iter_state(&self.rx_sets, state, |(idx, e)| {
            let d = self.cache.derivative(e, b);
            if d != ExprRef::NO_MATCH {
                Self::push_rx(&mut vec_desc, idx, d);
            }
        });

        self.insert_state(vec_desc)
    }
}

impl Debug for RegexVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RegexVec({})", self.stats())
    }
}
