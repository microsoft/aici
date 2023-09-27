use std::{mem::size_of, slice::from_raw_parts};

const NO_TOKEN: TokenId = 0;

pub type TokenId = u16;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StateOffset {
    pub off: u32,
}

impl StateOffset {
    pub const NONE: StateOffset = StateOffset { off: 0 };
    pub const DEAD: StateOffset = StateOffset { off: 1 };
    pub const START: StateOffset = StateOffset { off: 4 };
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
    pub token_data: &'static [u16],
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
                    as *const u16,
                hd.token_bytes as usize / size_of::<u16>(),
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

    fn state_bias(state: StateOffset) -> f32 {
        if state == StateOffset::DEAD {
            -100.0
        } else {
            0.0
        }
    }

    pub fn compute_logit_bias(&self, state_offset: StateOffset, bias: &mut [f32]) {
        let mut p = state_offset.off as usize;
        let default_state = StateOffset {
            off: self.state_data[p],
        };
        p += 1;

        let init_val = Self::state_bias(default_state);
        for idx in 0..bias.len() {
            bias[idx] = init_val;
        }

        loop {
            let state = StateOffset {
                off: self.state_data[p],
            };
            if state == StateOffset::NONE {
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

    pub fn advance(&self, state_offset: StateOffset, token: TokenId) -> StateOffset {
        let mut p = state_offset.off as usize;
        let default_state = StateOffset {
            off: self.state_data[p],
        };
        p += 1;
        loop {
            let state = StateOffset {
                off: self.state_data[p],
            };
            if state == StateOffset::NONE {
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
