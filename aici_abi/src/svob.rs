use crate::TokenId;

#[derive(Clone)]
pub struct SimpleVob {
    data: Vec<u32>,
}

const BITS: usize = 32;

impl SimpleVob {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.data.len() * BITS
    }

    #[inline(always)]
    pub fn allow_token(&mut self, tok: TokenId) {
        let idx = tok as usize;
        let byte_idx = idx / BITS;
        let bit_idx = idx % BITS;
        self.data[byte_idx] |= 1 << bit_idx;
    }

    pub fn resize(&mut self, size: usize) {
        let new_size = size / BITS + 1;
        assert!(new_size >= self.data.len());
        self.data.resize(new_size, 0);
    }

    #[inline(always)]
    pub fn is_allowed(&self, tok: TokenId) -> bool {
        let idx = tok as usize;
        let byte_idx = idx / 32;
        let bit_idx = idx % 32;
        (self.data[byte_idx] & (1 << bit_idx)) != 0
    }

    pub fn set_all(&mut self, val: bool) {
        let val = if val { !0 } else { 0 };
        self.data.iter_mut().for_each(|x| *x = val);
    }

    pub fn apply_to(&self, logits: &mut [f32]) {
        for (idx, v) in self.data.iter().enumerate() {
            if *v == 0 {
                continue;
            }
            let idx = idx * BITS;
            for bit_idx in 0..BITS {
                if v & (1 << bit_idx) != 0 {
                    logits[idx + bit_idx] = 0.0;
                }
            }
        }
    }
}
