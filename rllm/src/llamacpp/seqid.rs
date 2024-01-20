use llama_cpp_low as cpp;

#[repr(transparent)]
pub struct SeqId {
    pub(super) cpp: cpp::Sequence,
}

impl SeqId {
    pub fn to_num(&self) -> usize {
        self.cpp.id() as usize
    }

    pub fn clone_from(&self, other: &SeqId, length: usize) {
        self.cpp.cp(&other.cpp, 0, length as i32);
    }

    pub fn trim(&self, length: usize) {
        self.cpp.rm(length as i32, -1);
    }
}

impl std::fmt::Display for SeqId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_num())
    }
}

pub struct SeqIdGen {
    pub(super) model: cpp::Model,
}

impl SeqIdGen {
    pub fn next(&self) -> SeqId {
        SeqId {
            cpp: self.model.new_sequence(),
        }
    }
}
