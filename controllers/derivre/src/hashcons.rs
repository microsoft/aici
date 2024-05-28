// TODO use hashbrown::HashTable to avoid Rc<>

use ahash::RandomState;
use hashbrown::hash_table::Entry;

use hashbrown::HashTable;

#[derive(Debug, Clone)]
struct Element {
    backing_start: u32,
    backing_end: u32,
}

impl Element {
    fn as_range(&self) -> std::ops::Range<usize> {
        (self.backing_start as usize)..(self.backing_end as usize)
    }
}

/// A hashconsing data structure for vectors of u32.
/// Given a vector, it stores it only once and returns a unique id.
pub struct VecHashCons {
    hasher: RandomState,
    backing: Vec<u32>,
    elements: Vec<Element>,
    table: HashTable<u32>,
    curr_elt: Element,
}

impl Default for VecHashCons {
    fn default() -> Self {
        Self::new()
    }
}

impl VecHashCons {
    pub fn new() -> Self {
        let mut r = VecHashCons {
            hasher: RandomState::new(),
            backing: Vec::new(),
            elements: Vec::new(),
            table: HashTable::new(),
            curr_elt: Element {
                backing_start: 0,
                backing_end: 0,
            },
        };
        r.insert(&[]);
        r
    }

    /// Insert a given vector and return its unique id.
    pub fn insert(&mut self, data: &[u32]) -> u32 {
        self.start_insert();
        self.push_slice(data);
        self.finish_insert()
    }

    /// Get vector with given unique id.
    /// Panics if id is out of bounds.
    #[inline(always)]
    pub fn get(&self, id: u32) -> &[u32] {
        &self.backing[self.elements[id as usize].as_range()]
    }

    /// Return number of elements in the hashcons (also largest unique id + 1).
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Estimate number of bytes used by the hashcons.
    pub fn num_bytes(&self) -> usize {
        self.backing.len() * std::mem::size_of::<u32>()
            + self.elements.len() * (5 + std::mem::size_of::<Element>())
    }
    
    // Incremental, zero-copy insertion:

    /// Start insertion process for a vector.
    /// Panics if start_insert() is called twice without finish_insert().
    #[inline(always)]
    pub fn start_insert(&mut self) {
        assert!(self.curr_elt.backing_end == 0);
        self.curr_elt.backing_end = self.curr_elt.backing_start;
    }

    /// Add an element to the vector being inserted.
    /// Requires start_insert() to have been called.
    #[inline(always)]
    pub fn push_u32(&mut self, head: u32) {
        assert!(self.curr_elt.backing_end >= self.curr_elt.backing_start);
        self.curr_elt.backing_end += 1;
        if self.backing.len() < self.curr_elt.backing_end as usize {
            self.backing.push(head);
        } else {
            self.backing[self.curr_elt.backing_end as usize - 1] = head;
        }
    }

    /// Add a slice to the vector being inserted.
    /// Requires start_insert() to have been called.
    #[inline(always)]
    pub fn push_slice(&mut self, elts: &[u32]) {
        assert!(self.curr_elt.backing_end >= self.curr_elt.backing_start);
        let slice_start = self.curr_elt.backing_end;
        self.curr_elt.backing_end += elts.len() as u32;
        if self.backing.len() < self.curr_elt.backing_end as usize {
            self.backing
                .resize(self.curr_elt.backing_end as usize + 1000, 0);
        }
        self.backing[slice_start as usize..self.curr_elt.backing_end as usize]
            .copy_from_slice(elts);
    }

    /// Finish insertion process for a vector.
    /// Returns the unique id of the vector.
    /// Requires start_insert() to have been called.
    pub fn finish_insert(&mut self) -> u32 {
        let hasher = &self.hasher;
        let curr_backing = &self.backing[self.curr_elt.as_range()];
        let hash = hasher.hash_one(curr_backing);
        let get_slice =
            |x: &u32| -> &[u32] { &self.backing[self.elements[*x as usize].as_range()] };
        let hasher = |x: &u32| -> u64 { hasher.hash_one(get_slice(x)) };
        let eq = |x: &u32| -> bool { get_slice(x) == curr_backing };

        match self.table.entry(hash, eq, hasher) {
            Entry::Occupied(e) => {
                self.curr_elt.backing_end = 0;
                *e.get()
            }
            Entry::Vacant(e) => {
                let id = self.elements.len() as u32;
                self.elements.push(self.curr_elt.clone());
                e.insert(id);
                self.curr_elt.backing_start = self.curr_elt.backing_end;
                self.curr_elt.backing_end = 0;
                id
            }
        }
    }
}
