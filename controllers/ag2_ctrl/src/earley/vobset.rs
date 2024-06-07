use aici_abi::svob::SimpleVob;
use rustc_hash::FxHashMap;
use std::{fmt::Debug, hash::Hash};

#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub struct VobIdx {
    v: u32,
}

impl VobIdx {
    pub fn new(v: usize) -> Self {
        VobIdx { v: v as u32 }
    }

    pub fn all_zero() -> Self {
        VobIdx { v: 0 }
    }

    pub fn as_usize(&self) -> usize {
        self.v as usize
    }

    pub fn is_zero(&self) -> bool {
        self.v == 0
    }
}

pub struct VobSet {
    vobs: Vec<SimpleVob>,
    by_vob: FxHashMap<SimpleVob, VobIdx>,
}

impl VobSet {
    pub fn new(single_vob_size: usize) -> Self {
        let mut r = VobSet {
            vobs: Vec::new(),
            by_vob: FxHashMap::default(),
        };
        let v = SimpleVob::alloc(single_vob_size);
        r.insert_or_get(&v);
        r.insert_or_get(&v.negated());
        r
    }

    pub fn insert_or_get(&mut self, vob: &SimpleVob) -> VobIdx {
        if let Some(idx) = self.by_vob.get(vob) {
            return *idx;
        }
        let len = self.vobs.len();
        if len == 0 && !vob.is_zero() {
            panic!("first vob must be empty");
        }
        let idx = VobIdx::new(len);
        self.vobs.push(vob.clone());
        self.by_vob.insert(vob.clone(), idx);
        idx
    }

    pub fn resolve(&self, idx: VobIdx) -> &SimpleVob {
        &self.vobs[idx.as_usize()]
    }

    pub fn and_is_zero(&self, a: VobIdx, b: VobIdx) -> bool {
        self.vobs[a.as_usize()].and_is_zero(&self.vobs[b.as_usize()])
    }
}
