use std::{collections::HashMap, rc::Rc};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VecHolder {
    data: Rc<Vec<u32>>,
}

pub struct VecHashMap {
    num_elts: usize,
    by_id: Vec<VecHolder>,
    by_data: HashMap<VecHolder, u32>,
}

impl Default for VecHashMap {
    fn default() -> Self {
        Self::new()
    }
}

impl VecHashMap {
    pub fn new() -> Self {
        let mut r = VecHashMap {
            by_id: Vec::new(),
            by_data: HashMap::new(),
            num_elts: 0,
        };
        r.insert(Vec::new());
        r
    }

    pub fn insert(&mut self, data: Vec<u32>) -> u32 {
        let holder = VecHolder {
            data: Rc::new(data),
        };
        if let Some(&id) = self.by_data.get(&holder) {
            return id;
        }
        let id = self.by_id.len() as u32;
        self.num_elts += holder.data.len();
        self.by_id.push(holder.clone());
        self.by_data.insert(holder, id);
        id
    }

    pub fn get(&self, id: u32) -> Option<&[u32]> {
        self.by_id.get(id as usize).map(|holder| &holder.data[..])
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn num_bytes(&self) -> usize {
        // estimate 8 void* per hashtable entry + Vec + Rc
        (self.num_elts * 4) + (self.len() * 8 * std::mem::size_of::<usize>())
    }
}
