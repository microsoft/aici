use crate::{
    ast::{ExprRef, MatchState},
    deriv::DerivCache,
    hashcons::VecHashMap,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateID(u32);

impl StateID {
    pub const INVALID: StateID = StateID(0);

    pub fn new(id: u32) -> Self {
        assert!(id != 0, "StateID(0) is reserved for invalid state");
        StateID(id)
    }

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }

    pub fn is_valid(&self) -> bool {
        self.0 != 0
    }
}

pub struct RegexSet {
    cache: DerivCache,
    rx_sets: VecHashMap,
    state_table: Vec<Vec<StateID>>,
    num_states: usize,
    num_transitions: usize,
}

impl RegexSet {
    pub fn new() -> Self {
        RegexSet {
            cache: DerivCache::new(),
            rx_sets: VecHashMap::new(),
            state_table: Vec::new(),
            num_states: 0,
            num_transitions: 0,
        }
    }

    pub fn initial_state(&mut self, exprs: &[ExprRef]) -> StateID {
        let exprs = bytemuck::cast_slice(exprs).to_vec();
        StateID(self.rx_sets.insert(exprs))
    }

    fn get_expr_refs(&self, state: StateID) -> &[ExprRef] {
        bytemuck::cast_slice(self.rx_sets.get(state.as_u32()).unwrap())
    }

    pub fn classify_state(&self, state: StateID) -> Vec<MatchState> {
        assert!(state.is_valid());
        self.get_expr_refs(state)
            .iter()
            .map(|&e| self.cache.get_expr(e).classify_state())
            .collect()
    }

    pub fn transition(&mut self, state: StateID, b: u8) -> StateID {
        let idx = state.as_usize();

        if idx >= self.state_table.len() {
            self.state_table
                .extend((self.state_table.len()..(idx + 20)).map(|_| vec![]));
        }
        let vec = &self.state_table[idx];
        if vec.len() > 0 && vec[b as usize].is_valid() {
            return vec[b as usize];
        }

        let d = self.transition_inner(state, b);

        if self.state_table[idx].len() == 0 {
            self.state_table[idx] = vec![StateID::INVALID; 256];
            self.num_states += 1;
        }
        self.state_table[idx][b as usize] = d;
        self.num_transitions += 1;

        d
    }

    fn transition_inner(&mut self, state: StateID, b: u8) -> StateID {
        assert!(state.is_valid());
        let mut exprs: Vec<u32> = self.rx_sets.get(state.as_u32()).unwrap().to_vec();
        for i in 0..exprs.len() {
            exprs[i] = self.cache.derivative(ExprRef(exprs[i]), b).0;
        }
        StateID(self.rx_sets.insert(exprs))
    }

    /// Estimate the size of the regex tables in bytes.
    pub fn bytes(&self) -> usize {
        self.cache.bytes()
            + self.num_states * 256 * std::mem::size_of::<StateID>()
            + self.state_table.len() * std::mem::size_of::<Vec<StateID>>()
            + self.rx_sets.bytes()
    }

    pub fn stats(&self) -> String {
        format!(
            "states: {}; transitions: {}; bytes: {}",
            self.num_states,
            self.num_transitions,
            self.bytes()
        )
    }
}
