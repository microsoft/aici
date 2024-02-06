use std::sync::Mutex;

use crate::{HashMap, SeqId, SequenceManager};
use llama_cpp_low as cpp;

pub struct CppSequenceManager {
    model: cpp::Model,
    seqs: Mutex<HashMap<SeqId, cpp::Sequence>>,
}

impl CppSequenceManager {
    pub fn new(model: cpp::Model) -> Self {
        Self {
            model,
            seqs: Mutex::new(HashMap::default()),
        }
    }

    pub fn with_cpp(&self, seq: SeqId, cb: impl FnOnce(&cpp::Sequence)) {
        let seqs = self.seqs.lock().unwrap();
        let seq = seqs.get(&seq).unwrap();
        cb(seq);
    }
}

impl SequenceManager for CppSequenceManager {
    fn new_sequence(&self) -> SeqId {
        let r = self.model.new_sequence();
        let id = SeqId(r.id() as usize);
        self.seqs.lock().unwrap().insert(id, r);
        id
    }

    fn copy(&self, src: SeqId, dst: SeqId, length: usize) {
        let seqs = self.seqs.lock().unwrap();
        let src = seqs.get(&src).unwrap();
        let dst = seqs.get(&dst).unwrap();
        dst.cp_from(src, 0, length as i32);
    }

    fn trim(&self, seq: SeqId, length: usize) {
        let seqs = self.seqs.lock().unwrap();
        let seq = seqs.get(&seq).unwrap();
        seq.rm(length as i32, -1);
    }

    fn delete(&self, seq: SeqId) {
        self.seqs.lock().unwrap().remove(&seq);
    }
}
