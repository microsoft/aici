use crate::toktree::{append_bias, TokTrie};

pub trait Recognizer {
    fn append(&self, byte: u8) -> Self;
    fn allowed(&self, byte: u8) -> bool;
}

#[inline(never)]
pub fn compute_bias(trie: &TokTrie, rec: &impl Recognizer, logits: &mut [f32]) {
    logits.iter_mut().for_each(|x| *x = -100.0);
    append_bias(trie, rec, logits);
}

pub struct Uppercase {
    len: usize,
}

impl Uppercase {
    pub fn new() -> Self {
        Uppercase { len: 0 }
    }
}

impl Recognizer for Uppercase {
    #[inline(always)]
    fn append(&self, _byte: u8) -> Self {
        Uppercase { len: self.len + 1 }
    }

    #[inline(always)]
    fn allowed(&self, byte: u8) -> bool {
        byte != 0xff
        // let ch = _byte as char;
        // if self.len < 2 {
        //     'A' <= ch && ch <= 'Z'
        // } else {
        //     'a' <= ch && ch <= 'z'
        // }
    }
}

// pub struct PrefixEnum {
//     prefix_ch: u8,
//     depth: u32,
//     allowed: Vec<Vec<u8>>,
// }

// impl Recognizer for PrefixEnum {
//     fn append1(&self, byte: u8) -> Self {
//         let mut depth = self.depth;
//         for b in bytes {
//             if depth > 0 {
//                 depth += 1;
//             }
//             if depth == 0 && *b == self.prefix_ch {
//                 depth = 1
//             }
//         }
//         todo!()
//     }

//     fn allowed(&self) -> Vec<Vec<u8>> {
//         self.allowed.clone()
//     }
// }
