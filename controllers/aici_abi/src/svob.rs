use crate::TokenId;
use std::{fmt::Debug, ops::Index};

#[derive(Clone)]
pub struct SimpleVob {
    data: Vec<u32>,
    size: usize,
}

impl Debug for SimpleVob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleVob")
            .field("len", &self.len())
            .finish()
    }
}

impl Default for SimpleVob {
    fn default() -> Self {
        Self::new()
    }
}

impl Into<Vec<u32>> for SimpleVob {
    fn into(self) -> Vec<u32> {
        self.data
    }
}

const BITS: usize = 32;

impl SimpleVob {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            size: 0,
        }
    }

    pub fn alloc(size: usize) -> Self {
        let mut r = Self::new();
        r.resize(size);
        r
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn num_set(&self) -> usize {
        self.data.iter().map(|x| x.count_ones() as usize).sum()
    }

    fn clear_excessive_bits(&mut self) {
        for i in self.size..(self.data.len() * 32) {
            // disallow tokens that are out of range
            self.disallow_token(i as TokenId);
        }
    }

    pub fn negated(&self) -> Self {
        let mut r = Self::new();
        r.data = self.data.iter().map(|x| !x).collect();
        r.size = self.size;
        r.clear_excessive_bits();
        r
    }

    pub unsafe fn as_ptr(&self) -> *const u32 {
        self.data.as_ptr()
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.data
    }

    #[inline(always)]
    pub fn iter_set_entries(&self, mut f: impl FnMut(usize)) {
        let src = self.as_slice();
        let numelts = self.size;
        let max_len = numelts / 32;
        for idx in 0..max_len {
            let d = src[idx];
            // optimize for the two common cases
            if d == 0 {
                continue;
            } else if d == u32::MAX {
                for bit in 0..32 {
                    f(idx * 32 + bit);
                }
            } else {
                for bit in 0..32 {
                    if d & (1 << bit) != 0 {
                        f(idx * 32 + bit);
                    }
                }
            }
        }
        // final few elts
        for idx in (max_len * 32)..numelts {
            if self.is_allowed(idx as TokenId) {
                f(idx);
            }
        }
    }

    #[inline(always)]
    pub fn iter_unset_entries(&self, mut f: impl FnMut(usize)) {
        let src = self.as_slice();
        let numelts = self.size;
        let max_len = numelts / 32;
        for idx in 0..max_len {
            let d = src[idx];
            // optimize for the two common cases
            if d == 0 {
                for bit in 0..32 {
                    f(idx * 32 + bit);
                }
            } else if d == u32::MAX {
                continue;
            } else {
                for bit in 0..32 {
                    if d & (1 << bit) == 0 {
                        f(idx * 32 + bit);
                    }
                }
            }
        }
        // final few elts
        for idx in (max_len * 32)..numelts {
            if !self.is_allowed(idx as TokenId) {
                f(idx);
            }
        }
    }

    #[inline(always)]
    pub fn iter_entries(&self, mut f: impl FnMut(bool, usize)) {
        let src = self.as_slice();
        let numelts = self.size;
        let max_len = numelts / 32;
        for idx in 0..max_len {
            let d = src[idx];
            // optimize for the two common cases
            if d == 0 {
                for bit in 0..32 {
                    f(false, idx * 32 + bit);
                }
            } else if d == u32::MAX {
                for bit in 0..32 {
                    f(true, idx * 32 + bit);
                }
            } else {
                for bit in 0..32 {
                    f(d & (1 << bit) != 0, idx * 32 + bit);
                }
            }
        }
        // final few elts
        for idx in (max_len * 32)..numelts {
            f(self.is_allowed(idx as TokenId), idx);
        }
    }

    pub fn write_to(&self, buf: &mut [u8]) {
        assert!(buf.len() == self.data.len() * 4);
        bytemuck::cast_slice_mut(buf).copy_from_slice(&self.data);
    }

    #[inline(always)]
    pub fn allow_token(&mut self, tok: TokenId) {
        let idx = tok as usize;
        let byte_idx = idx / BITS;
        let bit_idx = idx % BITS;
        self.data[byte_idx] |= 1 << bit_idx;
    }

    #[inline(always)]
    pub fn disallow_token(&mut self, tok: TokenId) {
        let idx = tok as usize;
        let byte_idx = idx / BITS;
        let bit_idx = idx % BITS;
        self.data[byte_idx] &= !(1 << bit_idx);
    }

    pub fn set(&mut self, tok: TokenId, val: bool) {
        if val {
            self.allow_token(tok);
        } else {
            self.disallow_token(tok);
        }
    }

    pub fn resize(&mut self, size: usize) {
        let new_size = size / BITS + 1;
        assert!(new_size >= self.data.len());
        self.data.resize(new_size, 0);
        self.size = size;
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
        self.clear_excessive_bits();
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

    pub fn iter(&self) -> SimpleVobIter {
        SimpleVobIter { vob: self, idx: 0 }
    }
}

pub struct SimpleVobIter<'a> {
    vob: &'a SimpleVob,
    idx: usize,
}

impl<'a> Iterator for SimpleVobIter<'a> {
    type Item = u32;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let mut bitoff = self.idx % BITS;
        let mut dataoff = self.idx / BITS;
        let data = &self.vob.data;
        while dataoff < data.len() {
            let d = data[dataoff] >> bitoff;
            if d != 0 {
                let idx = dataoff * BITS + d.trailing_zeros() as usize + bitoff;
                self.idx = idx + 1;
                return Some(idx as u32);
            }
            bitoff = 0;
            dataoff += 1;
        }
        return None;
    }
}

impl Index<usize> for SimpleVob {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        if self.is_allowed(index as TokenId) {
            &true
        } else {
            &false
        }
    }
}
