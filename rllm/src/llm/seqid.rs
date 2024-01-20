use std::sync::Mutex;

pub struct SeqId {
    num: usize,
}

impl SeqId {
    pub fn to_num(&self) -> usize {
        self.num
    }
}

impl std::fmt::Display for SeqId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.num)
    }
}

pub struct SeqIdGen {
    next: Mutex<usize>,
}

impl SeqIdGen {
    pub fn new() -> Self {
        Self { next: Mutex::new(1) }
    }

    pub fn next(&self) -> SeqId {
        let mut l = self.next.lock().unwrap();
        let r = SeqId { num: *l };
        *l = *l + 1;
        r
    }
}

