const NO_TOKEN: TokenId = 0;

pub type TokenId = u16;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StateId {
    off: u32,
}

impl StateId {
    const NONE: StateId = StateId { off: 0 };
    const DEAD: StateId = StateId { off: 1 };
    pub const START: StateId = StateId { off: 4 };
}

#[derive(Clone)]
pub struct TokenCompiled {
    pub token_data: &'static [u16],
    pub state_data: &'static [u32],
}

impl TokenCompiled {
    fn token_in_token_set(&self, token: TokenId, set: u32) -> bool {
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

    fn state_bias(state: StateId) -> f32 {
        if state == StateId::DEAD {
            -100.0
        } else {
            0.0
        }
    }

    pub fn compute_logit_bias(&self, state_offset: StateId, bias: &mut [f32]) {
        let mut p = state_offset.off as usize;
        let default_state = StateId {
            off: self.state_data[p],
        };
        p += 1;

        let init_val = Self::state_bias(default_state);
        for idx in 0..bias.len() {
            bias[idx] = init_val;
        }

        loop {
            let state = StateId {
                off: self.state_data[p],
            };
            if state == StateId::NONE {
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

    pub fn advance(&self, state_offset: StateId, token: TokenId) -> StateId {
        let mut p = state_offset.off as usize;
        let default_state = StateId {
            off: self.state_data[p],
        };
        p += 1;
        loop {
            let state = StateId {
                off: self.state_data[p],
            };
            if state == StateId::NONE {
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
