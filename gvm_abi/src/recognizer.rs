use crate::toktree::{TokTrie, TrieNode};

pub trait Recognizer {
    fn append(&self, byte: u8) -> Self;
    fn allowed<'a>(&self, mask: &'a mut [u8]) -> AllowedResult<'a>;
}

fn append_bias(
    trie: &TokTrie,
    rec: &impl Recognizer,
    logits: &mut [f32],
    maskbuf: &mut [u8],
    n: TrieNode,
) {
    let sel = trie.children(n);
    let sel: Vec<TrieNode> = match rec.allowed(maskbuf) {
        AllowedResult::All => sel.collect(),
        AllowedResult::None => return,
        AllowedResult::Mask(mask) => sel
            .filter(|c| mask[trie.child_byte(*c) as usize] != 0)
            .collect(),
        AllowedResult::List(lst) => sel.filter(|c| lst.contains(&trie.child_byte(*c))).collect(),
    };

    for ch in sel {
        if let Some(tok) = trie.token_id(ch) {
            logits[tok as usize] = 0.0;
        }
        append_bias(trie, &rec.append(trie.child_byte(ch)), logits, maskbuf, ch)
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

pub enum AllowedResult<'a> {
    All,
    None,
    Mask(&'a [u8]),
    List(&'a [u8]),
}

impl Recognizer for Uppercase {
    fn append(&self, _byte: u8) -> Self {
        Uppercase { len: self.len + 1 }
    }

    fn allowed<'a>(&self, _mask: &'a mut [u8]) -> AllowedResult<'a> {
        AllowedResult::All

        // if self.len < 2 {
        //     AllowedResult::List(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        // } else {
        //     AllowedResult::List(b"abcdefghijklmnopqrstuvwxyz")
        // }

        // let mut idx = 0;
        // if self.len < 2 {
        //     for ch in 'A'..'Z' {
        //         mask[idx] = ch as u8;
        //         idx += 1;
        //     }
        // } else {
        //     for ch in 'a'..'z' {
        //         mask[idx] = ch as u8;
        //         idx += 1;
        //     }
        // }
        // AllowedResult::List(&mask[0..idx])
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
