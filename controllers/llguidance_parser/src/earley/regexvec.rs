use toktrie::SimpleVob;
use derivre::raw::{DerivCache, ExprSet, NextByteCache, VecHashCons};
use std::fmt::Debug;

pub use derivre::{AlphabetInfo, ExprRef, NextByte, StateID};

#[derive(Clone)]
pub struct RegexVec {
    exprs: ExprSet,
    deriv: DerivCache,
    next_byte: NextByteCache,
    alpha: AlphabetInfo,
    lazy: SimpleVob,
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
    pub state: StateID,
    pub lowest_accepting: Option<usize>,
    pub accepting: SimpleVob,
    pub possible: SimpleVob,

    possible_lookahead_len: Option<usize>,
    lookahead_len: Option<Option<usize>>,
    next_byte: Option<NextByte>,
    lowest_match: Option<Option<(usize, usize)>>,
}

impl StateDesc {
    pub fn is_accepting(&self) -> bool {
        self.lowest_accepting.is_some()
    }

    pub fn is_dead(&self) -> bool {
        self.possible.is_zero()
    }
}

// public implementation
impl RegexVec {
    pub fn alpha(&self) -> &AlphabetInfo {
        &self.alpha
    }

    pub fn lazy_regexes(&self) -> &SimpleVob {
        &self.lazy
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

    pub fn possible_lookahead_len(&mut self, state: StateID) -> usize {
        let desc = &mut self.state_descs[state.as_usize()];
        if let Some(len) = desc.possible_lookahead_len {
            return len;
        }
        let mut max_len = 0;
        for (_, e) in iter_state(&self.rx_sets, state) {
            max_len = max_len.max(self.exprs.possible_lookahead_len(e));
        }
        desc.possible_lookahead_len = Some(max_len);
        max_len
    }

    pub fn lookahead_len_for_state(&mut self, state: StateID) -> Option<usize> {
        let desc = &mut self.state_descs[state.as_usize()];
        if desc.lowest_accepting.is_none() {
            return None;
        }
        let idx = desc.lowest_accepting.unwrap();
        if let Some(len) = desc.lookahead_len {
            return len;
        }
        let mut res = None;
        let exprs = &self.exprs;
        for (idx2, e) in iter_state(&self.rx_sets, state) {
            if res.is_none() && exprs.is_nullable(e) {
                assert!(idx == idx2);
                res = Some(exprs.lookahead_len(e).unwrap_or(0));
            }
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

    /// Estimate the size of the regex tables in bytes.
    pub fn num_bytes(&self) -> usize {
        self.exprs.num_bytes()
            + self.deriv.num_bytes()
            + self.next_byte.num_bytes()
            + self.state_descs.len() * 100
            + self.state_table.len() * std::mem::size_of::<StateID>()
            + self.rx_sets.num_bytes()
    }

    fn lowest_match_inner(&mut self, state: StateID) -> Option<(usize, usize)> {
        let mut all_eoi = true;
        let mut eoi_candidate = None;
        // fine the first lazy matching regex
        // failing that, if all regexes are matching and force EOI, pick the first one
        for (idx, e) in iter_state(&self.rx_sets, state) {
            if !self.exprs.is_nullable(e) {
                all_eoi = false;
                continue;
            }
            if self.lazy[idx] {
                let len = self.exprs.possible_lookahead_len(e);
                return Some((idx, len));
            }
            if all_eoi {
                if self.next_byte.next_byte(&self.exprs, e) == NextByte::ForcedEOI {
                    if eoi_candidate.is_none() {
                        eoi_candidate = Some((idx, self.exprs.possible_lookahead_len(e)));
                    }
                } else {
                    all_eoi = false;
                }
            }
        }

        if all_eoi {
            eoi_candidate
        } else {
            None
        }
    }

    /// Return index of lowest matching regex if any.
    /// Lazy regexes match as soon as they accept, while greedy only
    /// if they accept and force EOI.
    pub fn lowest_match(&mut self, state: StateID) -> Option<(usize, usize)> {
        if let Some(lowest_match) = self.state_descs[state.as_usize()].lowest_match {
            return lowest_match;
        }
        let res = self.lowest_match_inner(state);
        self.state_descs[state.as_usize()].lowest_match = Some(res);
        res
    }

    /// Check if the there is only one transition out of state.
    /// This is an approximation - see docs for NextByte.
    pub fn next_byte(&mut self, state: StateID) -> NextByte {
        let desc = &mut self.state_descs[state.as_usize()];
        if let Some(next_byte) = desc.next_byte {
            return next_byte;
        }

        let mut next_byte = NextByte::Dead;
        for (_, e) in iter_state(&self.rx_sets, state) {
            next_byte = next_byte | self.next_byte.next_byte(&self.exprs, e);
            if next_byte == NextByte::SomeBytes {
                break;
            }
        }
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

    pub fn limit_state_to(&mut self, state: StateID, allowed_lexemes: &SimpleVob) -> StateID {
        let mut vec_desc = vec![];
        for (idx, e) in iter_state(&self.rx_sets, state) {
            if allowed_lexemes.get(idx) {
                Self::push_rx(&mut vec_desc, idx, e);
            }
        }
        self.insert_state(vec_desc)
    }

    pub fn total_fuel_spent(&self) -> usize {
        self.exprs.cost()
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
        self.alpha.has_error()
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
            self.exprs.len() - self.num_ast_nodes,
            self.deriv.num_deriv,
            self.total_fuel_spent(),
            self.state_descs.len(),
            self.num_transitions,
            self.num_bytes(),
            self.alpha.len(),
            if self.has_error() {
                "ERROR"
            } else { "" }
        )
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

// private implementation
impl RegexVec {
    pub(crate) fn new_with_exprset(
        exprset: &ExprSet,
        rx_list: &[ExprRef],
        lazy: Option<SimpleVob>,
    ) -> Self {
        let (alpha, exprset, rx_list) = AlphabetInfo::from_exprset(exprset, rx_list);
        let num_ast_nodes = exprset.len();

        let rx_sets = StateID::new_hash_cons();

        let mut r = RegexVec {
            deriv: DerivCache::new(),
            next_byte: NextByteCache::new(),
            lazy: lazy.unwrap_or_else(|| SimpleVob::alloc(rx_list.len())),
            exprs: exprset,
            alpha,
            rx_list,
            rx_sets,
            state_table: vec![],
            state_descs: vec![],
            num_transitions: 0,
            num_ast_nodes,
            fuel: usize::MAX,
            max_states: usize::MAX,
        };

        r.insert_state(vec![]);
        // also append state for the "MISSING"
        r.append_state(r.state_descs[0].clone());
        // in fact, transition from MISSING and DEAD should both lead to DEAD
        r.state_table.fill(StateID::DEAD);
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

    fn insert_state(&mut self, lst: Vec<u32>) -> StateID {
        // does this help?
        // if lst.len() == 0 {
        //     return StateID::DEAD;
        // }
        assert!(lst.len() % 2 == 0);
        let id = StateID::new(self.rx_sets.insert(&lst));
        if id.as_usize() >= self.state_descs.len() {
            self.append_state(self.compute_state_desc(id));
        }
        id
    }

    fn compute_state_desc(&self, state: StateID) -> StateDesc {
        let mut res = StateDesc {
            state,
            lowest_accepting: None,
            accepting: SimpleVob::alloc(self.rx_list.len()),
            possible: SimpleVob::alloc(self.rx_list.len()),
            possible_lookahead_len: None,
            lookahead_len: None,
            next_byte: None,
            lowest_match: None,
        };
        for (idx, e) in iter_state(&self.rx_sets, state) {
            res.possible.set(idx, true);
            if self.exprs.is_nullable(e) {
                res.accepting.set(idx, true);
                if res.lowest_accepting.is_none() {
                    res.lowest_accepting = Some(idx);
                }
            }
        }
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

        let d0 = self.deriv.num_deriv;
        let c0 = self.exprs.cost();
        let t0 = std::time::Instant::now();

        for (idx, e) in iter_state(&self.rx_sets, state) {
            let d = self.deriv.derivative(&mut self.exprs, e, b);
            if d != ExprRef::NO_MATCH {
                Self::push_rx(&mut vec_desc, idx, d);
            }
        }

        let num_deriv = self.deriv.num_deriv - d0;
        let cost = self.exprs.cost() - c0;
        self.fuel = self.fuel.saturating_sub(cost);
        if self.fuel == 0 {
            self.alpha.enter_error_state();
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
}

impl Debug for RegexVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RegexVec({})", self.stats())
    }
}

fn iter_state<'a>(
    rx_sets: &'a VecHashCons,
    state: StateID,
) -> impl Iterator<Item = (usize, ExprRef)> + 'a {
    let lst = rx_sets.get(state.as_u32());
    (0..lst.len())
        .step_by(2)
        .map(move |idx| (lst[idx] as usize, ExprRef::new(lst[idx + 1])))
}

// #[test]
// fn test_fuel() {
//     let mut rx = RegexVec::new_single("a(bc+|b[eh])g|.h").unwrap();
//     println!("{:?}", rx);
//     rx.set_fuel(200);
//     match_(&mut rx, "abcg");
//     assert!(!rx.has_error());
//     let mut rx = RegexVec::new_single("a(bc+|b[eh])g|.h").unwrap();
//     println!("{:?}", rx);
//     rx.set_fuel(20);
//     no_match(&mut rx, "abcg");
//     assert!(rx.has_error());
// }
