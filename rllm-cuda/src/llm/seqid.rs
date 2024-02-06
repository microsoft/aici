use std::sync::Mutex;

use crate::{SeqId, SequenceManager};

pub struct TchSeqMgr {
    next: Mutex<usize>,
}

impl TchSeqMgr {
    pub fn new() -> Self {
        Self {
            next: Mutex::new(1),
        }
    }
}

impl SequenceManager for TchSeqMgr {
    fn new_sequence(&self) -> SeqId {
        let mut l = self.next.lock().unwrap();
        let r = SeqId(*l);
        *l = *l + 1;
        r
    }

    fn copy(&self, src: SeqId, dst: SeqId, length: usize) {}

    fn trim(&self, seq: SeqId, length: usize) {}

    fn delete(&self, seq: SeqId) {}
}
