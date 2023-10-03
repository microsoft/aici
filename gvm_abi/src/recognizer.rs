use crate::toktree::{TokTrie, TrieNode};

pub trait Recognizer {
    fn append(&self, bytes: &[u8]) -> Self
    where
        Self: Sized,
    {
        let mut rec = self.append1(bytes[0]);
        for b in &bytes[1..] {
            rec = rec.append1(*b);
        }
        rec
    }
    fn append1(&self, byte: u8) -> Self;
    fn allowed(&self) -> Vec<Vec<u8>>;
}

fn append_bias(
    trie: &TokTrie,
    rec: &impl Recognizer,
    logits: &mut [f32],
    mut n: Option<TrieNode>,
    v: &Vec<u8>,
) {
    for b in v {
        match n {
            Some(n2) => {
                n = trie.child_at_byte(n2, *b);
                match n {
                    Some(n3) => {
                        if let Some(tok) = trie.token_id(n3) {
                            logits[tok as usize] = 0.0;
                        }
                    }
                    None => break,
                }
            }
            None => break,
        }
    }

    if n.is_some() {
        let rec = rec.append(v);
        for v in rec.allowed() {
            append_bias(trie, &rec, logits, n, &v);
        }
    }
}

pub fn compute_bias(trie: &TokTrie, rec: &impl Recognizer, logits: &mut [f32]) {
    logits.iter_mut().for_each(|x| *x = -100.0);
    for v in rec.allowed() {
        append_bias(trie, rec, logits, Some(trie.root()), &v);
    }
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
    fn append1(&self, _byte: u8) -> Self {
        Uppercase { len: self.len + 1 }
    }

    fn allowed(&self) -> Vec<Vec<u8>> {
        if self.len > 1 {
            ('a'..'z').map(|c| vec![c as u8]).collect()
        } else {
            ('A'..'Z').map(|c| vec![c as u8]).collect()
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
