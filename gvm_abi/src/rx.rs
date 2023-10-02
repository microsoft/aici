use std::{mem::size_of, slice::from_raw_parts};

pub type TokenId = u32;
pub type Transition = (StateOffset, TokenSetOffset);

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TokenSetOffset {
    pub off: u32,
}

pub struct StateDesc {
    default_transition: StateOffset,
    transitions: &'static [Transition],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StateOffset {
    pub off: u32,
}

impl StateOffset {
    pub const DEAD: StateOffset = StateOffset { off: 1 };
    pub const START: StateOffset = StateOffset { off: 3 };
}

#[repr(C)]
struct TokRxHeader {
    magic: u32,
    hd_size: u32,
    state_bytes: u32,
    token_bytes: u32,
    info: TokRxInfo,
    align: [u32; 0],
}

#[repr(C)]
#[derive(Clone)]
pub struct TokRxInfo {
    pub tok_eos: TokenId,
}

fn clone_vec_as_bytes<T>(input: &Vec<T>) -> Vec<u8> {
    unsafe {
        let byte_slice = from_raw_parts(input.as_ptr() as *const u8, input.len() * size_of::<T>());
        byte_slice.to_vec()
    }
}

fn clone_as_bytes<T>(input: &T) -> Vec<u8> {
    unsafe {
        let byte_slice = from_raw_parts(input as *const T as *const u8, size_of::<T>());
        byte_slice.to_vec()
    }
}

impl TokRxHeader {
    pub const MAGIC: u32 = 0x6623f10b;
    pub const SIZE: u32 = size_of::<TokRxHeader>() as u32;
}

#[derive(Clone)]
pub struct TokRx {
    pub info: &'static TokRxInfo,
    pub token_data: &'static [TokenId],
    pub state_data: &'static [u32],
}

impl TokRx {
    pub fn deserialize(bytes: &'static [u8]) -> TokRx {
        unsafe {
            assert!(bytes.len() > TokRxHeader::SIZE as usize);
            let hd = (bytes.as_ptr() as *const TokRxHeader).as_ref().unwrap();
            assert!(hd.magic == TokRxHeader::MAGIC);
            assert!(hd.hd_size == TokRxHeader::SIZE);
            let state_data = from_raw_parts(
                bytes.as_ptr().add(TokRxHeader::SIZE as usize) as *const u32,
                hd.state_bytes as usize / size_of::<u32>(),
            );
            let token_data = from_raw_parts(
                bytes
                    .as_ptr()
                    .add((TokRxHeader::SIZE + hd.state_bytes) as usize)
                    as *const TokenId,
                hd.token_bytes as usize / size_of::<TokenId>(),
            );
            TokRx {
                info: &hd.info,
                state_data,
                token_data,
            }
        }
    }

    pub fn serialize(
        info: &TokRxInfo,
        token_data: &Vec<TokenId>,
        state_data: &Vec<u32>,
    ) -> Vec<u8> {
        let mut token_bytes = clone_vec_as_bytes(&token_data);
        let mut state_bytes = clone_vec_as_bytes(&state_data);
        let hd = TokRxHeader {
            magic: TokRxHeader::MAGIC,
            hd_size: TokRxHeader::SIZE,
            info: info.clone(),
            state_bytes: state_bytes.len() as u32,
            token_bytes: token_bytes.len() as u32,
            align: [],
        };
        let mut bytes = clone_as_bytes(&hd);
        bytes.append(&mut state_bytes);
        bytes.append(&mut token_bytes);
        bytes
    }

    fn token_set(&self, set: TokenSetOffset) -> &'static [TokenId] {
        let idx = set.off as usize;
        let sz = self.token_data[idx] as usize;
        unsafe { from_raw_parts(self.token_data.as_ptr().add(idx + 1), sz) }
    }

    fn state_desc(&self, state: StateOffset) -> StateDesc {
        let idx = state.off as usize;
        let default_transition = StateOffset {
            off: self.state_data[idx],
        };
        let sz = self.state_data[idx + 1] as usize;
        StateDesc {
            default_transition,
            transitions: unsafe {
                from_raw_parts(
                    self.state_data.as_ptr().add(idx + 2) as *const Transition,
                    sz,
                )
            },
        }
    }

    fn state_bias(state: StateOffset) -> f32 {
        if state == StateOffset::DEAD {
            -100.0
        } else {
            0.0
        }
    }

    pub fn compute_logit_bias(&self, state_offset: StateOffset, bias: &mut [f32]) {
        let state = self.state_desc(state_offset);

        let init_val = Self::state_bias(state.default_transition);
        for idx in 0..bias.len() {
            bias[idx] = init_val;
        }

        for (st, ts) in state.transitions {
            let val = Self::state_bias(*st);
            let toks = self.token_set(*ts);
            for tok in toks {
                bias[*tok as usize] = val;
            }
        }
    }

    pub fn advance(&self, state_offset: StateOffset, token: TokenId) -> StateOffset {
        let state = self.state_desc(state_offset);

        for (st, ts) in state.transitions {
            if self.token_set(*ts).contains(&token) {
                return *st;
            }
        }

        state.default_transition
    }
}
