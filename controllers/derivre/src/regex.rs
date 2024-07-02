use std::{collections::HashSet, fmt::Debug};

use anyhow::Result;

use crate::{
    ast::{ExprRef, ExprSet, NextByte},
    bytecompress::ByteCompressor,
    deriv::DerivCache,
    hashcons::VecHashCons,
    nextbyte::NextByteCache,
    pp::PrettyPrinter,
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

    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn new_hash_cons() -> VecHashCons {
        let mut rx_sets = VecHashCons::new();
        let id = rx_sets.insert(&[]);
        assert!(id == StateID::DEAD.as_u32());
        let id = rx_sets.insert(&[ExprRef::INVALID.as_u32()]);
        assert!(id == StateID::MISSING.as_u32());
        rx_sets
    }
}

impl Debug for StateID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == StateID::DEAD {
            write!(f, "StateID(DEAD)")
        } else if *self == StateID::MISSING {
            write!(f, "StateID(MISSING)")
        } else {
            write!(f, "StateID({})", self.0)
        }
    }
}

#[derive(Clone)]
pub struct AlphabetInfo {
    mapping: [u8; 256],
    inv_mapping: [Option<u8>; 256],
    size: usize,
}

#[derive(Clone)]
pub struct Regex {
    exprs: ExprSet,
    deriv: DerivCache,
    next_byte: NextByteCache,
    alpha: AlphabetInfo,
    initial: StateID,
    rx_sets: VecHashCons,
    state_table: Vec<StateID>,
    state_descs: Vec<StateDesc>,
    num_transitions: usize,
    num_ast_nodes: usize,
    max_states: usize,
}

#[derive(Clone, Debug, Default)]
struct StateDesc {
    lookahead_len: Option<Option<usize>>,
    next_byte: Option<NextByte>,
}

// public implementation
impl Regex {
    pub fn new(rx: &str) -> Result<Self> {
        let parser = regex_syntax::ParserBuilder::new().build();
        Self::new_with_parser(parser, rx)
    }

    pub fn new_with_parser(parser: regex_syntax::Parser, rx: &str) -> Result<Self> {
        let mut exprset = ExprSet::new(256);
        let rx = exprset.parse_expr(parser.clone(), rx)?;
        Ok(Self::new_with_exprset(&exprset, rx))
    }

    pub fn alpha(&self) -> &AlphabetInfo {
        &self.alpha
    }

    pub fn initial_state(&mut self) -> StateID {
        self.initial
    }

    pub fn is_accepting(&mut self, state: StateID) -> bool {
        self.lookahead_len_for_state(state).is_some()
    }

    fn resolve(rx_sets: &VecHashCons, state: StateID) -> ExprRef {
        ExprRef::new(rx_sets.get(state.as_u32())[0])
    }

    pub fn lookahead_len_for_state(&mut self, state: StateID) -> Option<usize> {
        if state == StateID::DEAD || state == StateID::MISSING {
            return None;
        }
        let desc = &mut self.state_descs[state.as_usize()];
        if let Some(len) = desc.lookahead_len {
            return len;
        }
        let expr = Self::resolve(&self.rx_sets, state);
        let mut res = None;
        if self.exprs.is_nullable(expr) {
            res = Some(self.exprs.lookahead_len(expr).unwrap_or(0));
        }
        desc.lookahead_len = Some(res);
        res
    }

    #[inline(always)]
    pub fn transition(&mut self, state: StateID, b: u8) -> StateID {
        let mapped = self.alpha.map(b);
        let idx = state.as_usize() * self.alpha.len() + mapped;
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
        let mut state = self.initial_state();
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
        self.exprs.num_bytes()
            + self.deriv.num_bytes()
            + self.next_byte.num_bytes()
            + self.state_descs.len() * 100
            + self.state_table.len() * std::mem::size_of::<StateID>()
            + self.rx_sets.num_bytes()
    }

    /// Check if the there is only one transition out of state.
    /// This is an approximation - see docs for NextByte.
    pub fn next_byte(&mut self, state: StateID) -> NextByte {
        if state == StateID::DEAD || state == StateID::MISSING {
            return NextByte::Dead;
        }

        let desc = &mut self.state_descs[state.as_usize()];
        if let Some(next_byte) = desc.next_byte {
            return next_byte;
        }

        let e = Self::resolve(&self.rx_sets, state);
        let next_byte = self.next_byte.next_byte(&self.exprs, e);
        let next_byte = match next_byte {
            NextByte::ForcedByte(b) => {
                if let Some(b) = self.alpha.inv_map(b as usize) {
                    NextByte::ForcedByte(b)
                } else {
                    NextByte::SomeBytes
                }
            }
            _ => next_byte,
        };
        desc.next_byte = Some(next_byte);
        next_byte
    }

    pub fn stats(&self) -> String {
        format!(
            "regexp: {} nodes (+ {} derived via {} derivatives), states: {}; transitions: {}; bytes: {}; alphabet size: {}",
            self.num_ast_nodes,
            self.exprs.len() - self.num_ast_nodes,
            self.deriv.num_deriv,
            self.state_descs.len(),
            self.num_transitions,
            self.num_bytes(),
            self.alpha.len(),
        )
    }

    pub fn dfa(&mut self) -> Vec<u8> {
        let mut used = HashSet::new();
        let mut designated_bytes = vec![];
        for b in 0..=255 {
            let m = self.alpha.map(b);
            if !used.contains(&m) {
                used.insert(m);
                designated_bytes.push(b);
            }
        }

        let mut stack = vec![self.initial_state()];
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
        let mut res = self.alpha.mapping.to_vec();
        res.extend(self.state_table.iter().map(|s| s.as_u32() as u8));
        res
    }

    pub fn print_state_table(&self) {
        let mut state = 0;
        for row in self.state_table.chunks(self.alpha.len()) {
            println!("state: {}", state);
            for (b, &new_state) in row.iter().enumerate() {
                println!("  s{:?} -> {:?}", b, new_state);
            }
            state += 1;
        }
    }
}

impl AlphabetInfo {
    pub fn from_exprset(exprset: &ExprSet, rx_list: &[ExprRef]) -> (Self, ExprSet, Vec<ExprRef>) {
        assert!(exprset.alphabet_size() == 256);
        let compress = true;

        debug!("rx0: {}", exprset.expr_to_string_with_info(rx_list[0]));

        let ((mut exprset, rx_list), mapping, alphabet_size) = if compress {
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
                (exprset.clone(), rx_list.to_vec()),
                (0..=255).collect(),
                alphabet_size,
            )
        };

        // disable expensive optimizations after initial construction
        exprset.disable_optimizations();

        let mut inv_alphabet_mapping = [None; 256];
        let mut num_mappings = [0; 256];
        for (i, &b) in mapping.iter().enumerate() {
            inv_alphabet_mapping[b as usize] = Some(i as u8);
            num_mappings[b as usize] += 1;
        }
        for i in 0..alphabet_size {
            if num_mappings[i] != 1 {
                inv_alphabet_mapping[i] = None;
            }
        }

        debug!(
            "compressed: {}",
            exprset.expr_to_string_with_info(rx_list[0])
        );

        let alpha = AlphabetInfo {
            mapping: mapping.try_into().unwrap(),
            inv_mapping: inv_alphabet_mapping,
            size: alphabet_size,
        };
        (alpha, exprset, rx_list.to_vec())
    }

    pub fn map(&self, b: u8) -> usize {
        self.mapping[b as usize] as usize
    }

    pub fn inv_map(&self, v: usize) -> Option<u8> {
        self.inv_mapping[v]
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn has_error(&self) -> bool {
        self.size == 0
    }

    pub fn enter_error_state(&mut self) {
        self.size = 0;
    }
}

// private implementation
impl Regex {
    pub(crate) fn new_with_exprset(exprset: &ExprSet, top_rx: ExprRef) -> Self {
        let (alpha, exprset, rx_list) = AlphabetInfo::from_exprset(exprset, &[top_rx]);
        let top_rx = rx_list[0];
        let num_ast_nodes = exprset.len();

        let rx_sets = StateID::new_hash_cons();

        let mut r = Regex {
            deriv: DerivCache::new(),
            next_byte: NextByteCache::new(),
            exprs: exprset,
            alpha,
            rx_sets,
            state_table: vec![],
            state_descs: vec![],
            num_transitions: 0,
            num_ast_nodes,
            initial: StateID::MISSING,
            max_states: usize::MAX,
        };

        let desc = StateDesc {
            lookahead_len: Some(None),
            next_byte: Some(NextByte::Dead),
        };

        // DEAD
        r.append_state(desc.clone());
        // also append state for the "MISSING"
        r.append_state(desc);
        // in fact, transition from MISSING and DEAD should both lead to DEAD
        r.state_table.fill(StateID::DEAD);

        r.initial = r.insert_state(top_rx);

        assert!(r.alpha.len() > 0);
        r
    }

    fn append_state(&mut self, state_desc: StateDesc) {
        let mut new_states = vec![StateID::MISSING; self.alpha.len()];
        self.state_table.append(&mut new_states);
        self.state_descs.push(state_desc);
        if self.state_descs.len() >= self.max_states {
            self.alpha.enter_error_state();
        }
    }

    fn insert_state(&mut self, d: ExprRef) -> StateID {
        let id = StateID(self.rx_sets.insert(&[d.as_u32()]));
        if id.as_usize() >= self.state_descs.len() {
            self.append_state(StateDesc::default());
        }
        id
    }

    fn transition_inner(&mut self, state: StateID, b: u8) -> StateID {
        assert!(state.is_valid());

        let e = Self::resolve(&self.rx_sets, state);
        let d = self.deriv.derivative(&mut self.exprs, e, b);
        if d == ExprRef::NO_MATCH {
            StateID::DEAD
        } else {
            self.insert_state(d)
        }
    }
}

impl Debug for Regex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Regex({})", self.stats())
    }
}
