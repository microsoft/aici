// use 8:24 encoding - num_ch:tok_id (ch_byte:ch_off)* - 8 bytes per tree node
// special case num_ch=0xff -> num_ch=0x100

use crate::rx::TokenId;

pub struct TokTrie {
    data: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct TrieNode {
    // ch_byte:ch_off
    bits: u32,
}

const NO_TOKEN: u32 = 0xffffff;

impl TokTrie {
    pub fn from(words: &Vec<Vec<u8>>) -> TokTrie {
        let mut trie = TrieHash::new(0xff);
        for (idx, word) in words.iter().enumerate() {
            if word.len() > 0 {
                trie.insert(word, idx as u32)
            }
        }
        let mut data = Vec::new();
        trie.serialize(&mut data);
        TokTrie { data }
    }

    pub fn root(&self) -> TrieNode {
        TrieNode { bits: 0 }
    }

    #[inline(always)]
    fn at(&self, n: TrieNode, off: usize) -> u32 {
        self.data[(n.bits >> 8) as usize + off]
    }

    pub fn child_byte(&self, n: TrieNode) -> u8 {
        (n.bits & 0xff) as u8
    }

    pub fn token_id(&self, n: TrieNode) -> Option<TokenId> {
        let r = self.at(n, 0) >> 8;
        if r == NO_TOKEN {
            None
        } else {
            Some(r)
        }
    }

    pub fn num_children(&self, n: TrieNode) -> usize {
        let num_ch = self.at(n, 0) & 0xff;
        if num_ch == 0xff {
            0x100
        } else {
            num_ch as usize
        }
    }

    pub fn child_at_idx(&self, n: TrieNode, idx: usize) -> TrieNode {
        assert!(idx < self.num_children(n));
        TrieNode {
            bits: self.at(n, 1 + idx),
        }
    }

    pub fn child_at_byte(&self, n: TrieNode, byte: u8) -> Option<TrieNode> {
        let num_ch = self.num_children(n);
        for idx in 0..num_ch {
            // let byte2 = self.child_byte(self.child_at_idx(n, idx));
            let byte2 = (self.at(n, 1 + idx) & 0xff) as u8;
            if byte2 == byte {
                return Some(self.child_at_idx(n, idx));
            }
        }
        None
    }

    pub fn child_at_bytes(&self, mut n: TrieNode, bytes: &[u8]) -> Option<TrieNode> {
        for &byte in bytes {
            n = match self.child_at_byte(n, byte) {
                Some(n) => n,
                None => return None,
            }
        }
        Some(n)
    }

    pub fn children(&self, n: TrieNode) -> TrieNodeChildrenIter {
        TrieNodeChildrenIter {
            parent: self,
            node: n,
            idx: 0,
            max_idx: self.num_children(n),
        }
    }
}

pub struct TrieNodeChildrenIter<'a> {
    parent: &'a TokTrie,
    node: TrieNode,
    idx: usize,
    max_idx: usize,
}

impl<'a> Iterator for TrieNodeChildrenIter<'a> {
    type Item = TrieNode;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.max_idx {
            let child = self.parent.child_at_idx(self.node, self.idx);
            self.idx += 1;
            Some(child)
        } else {
            None
        }
    }
}

#[repr(C)]
pub struct TokenizerBin {
    magic: u32,
    tokens_bytes: u32,
    tree_bytes: u32,
}

struct TrieHash {
    token_id: u32,
    byte: u8,
    children: Vec<TrieHash>,
}

impl TrieHash {
    fn new(byte: u8) -> TrieHash {
        TrieHash {
            token_id: NO_TOKEN,
            byte,
            children: Vec::new(),
        }
    }
    fn insert(&mut self, word: &[u8], token_id: u32) {
        if word.len() == 0 {
            assert!(self.token_id == NO_TOKEN);
            self.token_id = token_id;
        } else {
            for idx in 0..self.children.len() {
                if self.children[idx].byte == word[0] {
                    self.children[idx].insert(&word[1..], token_id);
                    return;
                }
            }
            let mut ch = TrieHash::new(word[0]);
            ch.insert(&word[1..], token_id);
            self.children.push(ch);
        }
    }
    fn serialize_val(&self, len: usize) -> u32 {
        (self.token_id << 8) | len as u32
    }

    fn serialize(&mut self, data: &mut Vec<u32>) {
        fn serialize_ch(off: usize, ch: u8) -> u32 {
            let ptr = off as u32;
            assert!(ptr as usize == off);
            assert!((ptr << 8) >> 8 == ptr);
            (ptr << 8) | (ch as u32)
        }
        let idx = data.len();
        let mut len = self.children.len();
        if len == 0x100 {
            len = 0xff;
        } else {
            assert!(len < 0xf0);
        }
        data.push(self.serialize_val(len));
        data.resize(idx + 1 + self.children.len(), 0);
        self.children.sort_by_key(|e| e.byte);
        let mut ch_idx = idx + 1;
        for entry in &mut self.children {
            data[ch_idx] = serialize_ch(data.len(), entry.byte);
            ch_idx += 1;
            entry.serialize(data);
        }
    }
}
