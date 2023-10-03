use crate::toktree::{TokTrie, TrieNode};

pub trait Recognizer {
    fn append(&self, byte: u8) -> Self;
    fn allowed(&self, mask: &mut [u8]);
}

fn append_bias(
    trie: &TokTrie,
    rec: &impl Recognizer,
    logits: &mut [f32],
    mask: &mut [u8],
    n: TrieNode,
) {
    rec.allowed(mask);

    let sel = trie
        .children(n)
        .filter(|c| mask[trie.child_byte(*c) as usize] != 0)
        .collect::<Vec<_>>();

    for ch in sel {
        if let Some(tok) = trie.token_id(ch) {
            logits[tok as usize] = 0.0;
        }
        append_bias(trie, &rec.append(trie.child_byte(n)), logits, mask, ch)
    }
}

pub fn compute_bias(trie: &TokTrie, rec: &impl Recognizer, logits: &mut [f32]) {
    logits.iter_mut().for_each(|x| *x = -100.0);
    let mut mask = Vec::new();
    mask.resize(256, 0);
    append_bias(trie, rec, logits, &mut mask, trie.root());
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
    fn append(&self, _byte: u8) -> Self {
        Uppercase { len: self.len + 1 }
    }

    fn allowed(&self, mask: &mut [u8]) {
        for idx in 0..255 {
            mask[idx] = 0;
        }
        if self.len < 2 {
            for ch in 'A'..'Z' {
                mask[ch as usize] = 1;
            }
        } else {
            for ch in 'a'..'z' {
                mask[ch as usize] = 1;
            }
        }
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
