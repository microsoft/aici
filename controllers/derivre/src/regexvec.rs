use std::{collections::HashSet, fmt::Debug};

use anyhow::Result;

use crate::{
    ast::{ExprRef, ExprSet},
    bytecompress::ByteCompressor,
    deriv::DerivCache,
    hashcons::VecHashCons,
    pp::PrettyPrinter,
    SimpleVob,
};

const DEBUG: bool = false;

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

    #[inline(always)]
    pub fn is_dead(&self) -> bool {
        *self == Self::DEAD
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
    rx_sets: VecHashCons,
    state_table: Vec<StateID>,
    state_descs: Vec<StateDesc>,
    num_transitions: usize,
    num_ast_nodes: usize,
    max_states: usize,
    fuel: usize,
}

#[derive(Clone, Debug)]
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
        let parser = regex_syntax::ParserBuilder::new().build();
        Self::new_with_parser(parser, rx_list)
    }

    pub fn new_with_parser(parser: regex_syntax::Parser, rx_list: &[&str]) -> Result<Self> {
        let mut exprset = ExprSet::new(256);
        let mut acc = Vec::new();
        for rx in rx_list {
            let ast = exprset.parse_expr(parser.clone(), rx)?;
            acc.push(ast);
        }
        Ok(Self::new_with_exprset(exprset, &acc))
    }

    pub fn initial_state_all(&mut self) -> StateID {
        self.initial_state(&SimpleVob::all_true(self.rx_list.len()))
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

    pub fn lookahead_len_for_state(&self, state: StateID) -> Option<usize> {
        let state_desc = self.state_desc(state);
        let idx = state_desc.lowest_accepting;
        if idx < 0 {
            return None;
        }
        let mut res = None;

        Self::iter_state(&self.rx_sets, state, |(idx2, e)| {
            if res.is_none() && self.exprs().is_nullable(e) {
                assert!(idx == idx2 as isize);
                res = Some(self.exprs().lookahead_len(e).unwrap_or(0));
            }
        });

        res
    }

    pub fn alphabet_mapping(&self) -> &[u8] {
        &self.alphabet_mapping
    }

    pub fn alphabet_size(&self) -> usize {
        self.alphabet_size
    }

    #[inline(always)]
    pub fn transition(&mut self, state: StateID, b: u8) -> StateID {
        let mapped = self.alphabet_mapping[b as usize] as usize;
        let idx = state.as_usize() * self.alphabet_size + mapped;
        let new_state = self.state_table[idx];
        if new_state != StateID::MISSING {
            new_state
        } else {
            let new_state = self.transition_inner(state, mapped as u8);
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
        self.lookahead_len(text).is_some()
    }

    pub fn lookahead_len(&mut self, text: &str) -> Option<usize> {
        let selected = SimpleVob::alloc(self.rx_list.len());
        let mut state = self.initial_state(&selected.negated());
        for b in text.bytes() {
            let new_state = self.transition(state, b);
            debug!("b: {:?} --{:?}--> {:?}", state, b as char, new_state);
            state = new_state;
            if state == StateID::DEAD {
                return None;
            }
        }
        self.lookahead_len_for_state(state)
    }

    /// Estimate the size of the regex tables in bytes.
    pub fn num_bytes(&self) -> usize {
        self.cache.num_bytes()
            + self.state_descs.len() * 100
            + self.state_table.len() * std::mem::size_of::<StateID>()
            + self.rx_sets.num_bytes()
    }

    pub fn total_fuel_spent(&self) -> usize {
        self.cache.exprs.cost
    }

    pub fn set_max_states(&mut self, max_states: usize) {
        if !self.has_error() {
            self.max_states = max_states;
        }
    }

    // Each fuel point is on the order 100ns (though it varies).
    // So, for ~10ms limit, do a .set_fuel(100_000).
    pub fn set_fuel(&mut self, fuel: usize) {
        if !self.has_error() {
            self.fuel = fuel;
        }
    }

    pub fn has_error(&self) -> bool {
        self.alphabet_size == 0
    }

    pub fn get_error(&self) -> Option<String> {
        if self.has_error() {
            if self.fuel == 0 {
                Some("too many expressions constructed".to_string())
            } else if self.state_descs.len() >= self.max_states {
                Some(format!(
                    "too many states: {} >= {}",
                    self.state_descs.len(),
                    self.max_states
                ))
            } else {
                Some("unknown error".to_string())
            }
        } else {
            None
        }
    }

    pub fn stats(&self) -> String {
        format!(
            "regexps: {} with {} nodes (+ {} derived via {} derivatives with total fuel {}), states: {}; transitions: {}; bytes: {}; alphabet size: {} {}",
            self.rx_list.len(),
            self.num_ast_nodes,
            self.cache.exprs.len() - self.num_ast_nodes,
            self.cache.num_deriv,
            self.total_fuel_spent(),
            self.state_descs.len(),
            self.num_transitions,
            self.num_bytes(),
            self.alphabet_size,
            if self.alphabet_size == 0 {
                "ERROR"
            } else { "" }
        )
    }

    pub fn dfa(&mut self) -> Vec<u8> {
        let mut used = HashSet::new();
        let mut designated_bytes = vec![];
        for b in 0..=255 {
            let m = self.alphabet_mapping[b];
            if !used.contains(&m) {
                used.insert(m);
                designated_bytes.push(b as u8);
            }
        }

        let mut stack = vec![self.initial_state_all()];
        let mut visited = HashSet::new();
        while let Some(state) = stack.pop() {
            for b in &designated_bytes {
                let new_state = self.transition(state, *b);
                if !visited.contains(&new_state) {
                    stack.push(new_state);
                    visited.insert(new_state);
                    assert!(visited.len() < 250);
                }
            }
        }

        assert!(!self.state_table.contains(&StateID::MISSING));
        let mut res = self.alphabet_mapping.clone();
        res.extend(self.state_table.iter().map(|s| s.as_u32() as u8));
        res
    }

    pub fn print_state_table(&self) {
        let mut state = 0;
        for row in self.state_table.chunks(self.alphabet_size) {
            println!("state: {}", state);
            for (b, &new_state) in row.iter().enumerate() {
                println!("  s{:?} -> {:?}", b, new_state);
            }
            state += 1;
        }
    }
}

// private implementation
impl RegexVec {
    fn new_with_exprset(exprset: ExprSet, rx_list: &[ExprRef]) -> Self {
        assert!(exprset.alphabet_size() == 256);
        let compress = true;

        debug!("rx0: {}", exprset.expr_to_string_with_info(rx_list[0]));

        let ((exprset, rx_list), mapping, alphabet_size) = if compress {
            let mut compressor = ByteCompressor::new();
            let cost0 = exprset.cost;
            let (mut exprset, rx_list) = compressor.compress(&exprset, rx_list);
            exprset.cost += cost0;
            exprset.set_pp(PrettyPrinter::new(
                compressor.mapping.clone(),
                compressor.alphabet_size,
            ));
            (
                (exprset, rx_list),
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

        debug!(
            "compressed: {}",
            exprset.expr_to_string_with_info(rx_list[0])
        );

        let num_ast_nodes = exprset.len();

        let mut rx_sets = VecHashCons::new();
        let id = rx_sets.insert(&[]);
        assert!(id == StateID::DEAD.as_u32());
        let id = rx_sets.insert(&[0]);
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
            fuel: usize::MAX,
            max_states: usize::MAX,
        };

        // disable expensive optimizations after initial construction
        r.cache.exprs.optimize = false;

        r.insert_state(vec![]);
        // also append state for the "MISSING"
        r.append_state(r.state_descs[0].clone());
        // in fact, transition from MISSING and DEAD should both lead to DEAD
        r.state_table.fill(StateID::DEAD);
        assert!(r.alphabet_size > 0);
        r
    }

    fn append_state(&mut self, state_desc: StateDesc) {
        let mut new_states = vec![StateID::MISSING; self.alphabet_size];
        self.state_table.append(&mut new_states);
        self.state_descs.push(state_desc);
        if self.state_descs.len() >= self.max_states {
            self.enter_error_state();
        }
    }

    fn insert_state(&mut self, lst: Vec<u32>) -> StateID {
        // does this help?
        // if lst.len() == 0 {
        //     return StateID::DEAD;
        // }
        assert!(lst.len() % 2 == 0);
        let id = StateID(self.rx_sets.insert(&lst));
        if id.as_usize() >= self.state_descs.len() {
            self.append_state(self.compute_state_desc(id));
        }
        id
    }

    fn iter_state(rx_sets: &VecHashCons, state: StateID, mut f: impl FnMut((usize, ExprRef))) {
        let lst = rx_sets.get(state.as_u32());
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
        // debug!("state {:?} desc: {:?}", state, res);
        res
    }

    fn push_rx(vec_desc: &mut Vec<u32>, idx: usize, e: ExprRef) {
        vec_desc.push(idx as u32);
        vec_desc.push(e.as_u32());
    }

    fn transition_inner(&mut self, state: StateID, b: u8) -> StateID {
        assert!(state.is_valid());

        let mut vec_desc = vec![];

        let d0 = self.cache.num_deriv;
        let c0 = self.cache.exprs.cost;
        let t0 = std::time::Instant::now();

        Self::iter_state(&self.rx_sets, state, |(idx, e)| {
            let d = self.cache.derivative(e, b);
            if d != ExprRef::NO_MATCH {
                Self::push_rx(&mut vec_desc, idx, d);
            }
        });

        let num_deriv = self.cache.num_deriv - d0;
        let cost = self.cache.exprs.cost - c0;
        self.fuel = self.fuel.saturating_sub(cost);
        if self.fuel == 0 {
            self.enter_error_state();
        }
        if false && cost > 40 {
            println!(
                "cost: {:?} {} {}",
                t0.elapsed() / (cost as u32),
                num_deriv,
                cost
            );
        }
        self.insert_state(vec_desc)
    }

    fn enter_error_state(&mut self) {
        self.alphabet_size = 0;
    }
}

impl Debug for RegexVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RegexVec({})", self.stats())
    }
}
