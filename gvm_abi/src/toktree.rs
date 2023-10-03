// use 8:24 encoding - num_ch:tok_id (ch_idx:ch_off)* - 8 bytes per token
// special case num_ch=0xff -> num_ch=0x100

use crate::rx::TokenId;

pub struct TokTrie {
    data: Vec<u32>,
}

pub struct TrieNode<'a> {
    trie: &'a TokTrie,
    pub byte: u8,
    off: usize,
    parent: Option<&'a TrieNode<'a>>
}

impl TokTrie {
    pub fn new() -> TokTrie {
        TokTrie { data: Vec::new() }
    }

    pub fn root<'a>(&'a self) -> TrieNode<'a> {
        TrieNode {
            trie: &self,
            byte: 0,
            off: 0,
            parent: None,
        }
    }
}

impl<'a> TrieNode<'a> {
    const NO_TOKEN: u32 = 0xffffff;

    pub fn token_id(&self) -> Option<TokenId> {
        let r = self.trie.data[self.off] >> 8;
        if r == Self::NO_TOKEN {
            None
        } else {
            Some(r)
        }
    }

    pub fn num_children(&self) -> usize {
        let num_ch = self.trie.data[self.off] & 0xff;
        if num_ch == 0xff {
            0x100
        } else {
            num_ch as usize
        }
    }

    pub fn child_at_idx(&'a self, idx: usize) -> TrieNode<'a> {
        assert!(idx < self.num_children());
        let off = self.off + 1 + idx;
        let ch_off = self.trie.data[off] >> 8;
        TrieNode {
            trie: self.trie,
            byte: (self.trie.data[off] & 0xff) as u8,
            off: ch_off as usize,
            parent: Some(self),
        }
    }

    pub fn child_at_byte(&'a self, byte: u8) -> Option<TrieNode<'a>> {
        let num_ch = self.num_children();
        for idx in 0..num_ch {
            let off = self.off + 1 + idx;
            if (self.trie.data[off] & 0xff) as u8 == byte {
                return Some(self.child_at_idx(idx));
            }
        }
        None
    }

    pub fn children(&self) -> TrieNodeChildrenIter {
        TrieNodeChildrenIter {
            parent: self,
            idx: 0,
            max_idx: self.num_children(),
        }
    }
}

pub struct TrieNodeChildrenIter<'a> {
    parent: &'a TrieNode<'a>,
    idx: usize,
    max_idx: usize,
}

impl<'a> Iterator for TrieNodeChildrenIter<'a> {
    type Item = TrieNode<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.max_idx {
            let child = self.parent.child_at_idx(self.idx);
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

pub fn iter(word: &[u8]) {
    let trie = TokTrie::new();
    let mut n = trie.root();
    for &ch in word {
        n = n.child_at_byte(ch).unwrap();
    }
}
