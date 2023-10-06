use crate::toktree::{append_bias, TokTrie, Recognizer};

#[inline(never)]
pub fn compute_bias<S: Copy>(trie: &TokTrie, rec: &mut impl Recognizer<S>, logits: &mut [f32]) {
    logits.iter_mut().for_each(|x| *x = -100.0);
    append_bias(trie, rec, logits);
}

pub struct LenExcluder {}

impl Recognizer<u32> for LenExcluder {
    fn initial(&mut self) -> u32 {
        0
    }

    #[inline(never)]
    fn append(&mut self, state: u32, _byte: u8) -> u32 {
        state + 1
    }

    #[inline(never)]
    fn allowed(&mut self, state: u32, byte: u8) -> bool {
        byte != (('z' as u32 + state) & 0xff) as u8
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
