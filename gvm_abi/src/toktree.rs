// use 8:24 encoding - num_ch:tok_id (ch_byte:ch_off)* - 8 bytes per tree node
// special case num_ch=0xff -> num_ch=0x100

use crate::recognizer::Recognizer;

pub struct TokTrie {
    pub data: Vec<TrieNode>,
}

pub struct TrieNode {
    // byte:token
    bits: u32,
    subtree_size: u32,
}

const NO_TOKEN: u32 = 0xffffff;

impl TrieNode {
    fn new(byte: u8, token_id: u32) -> TrieNode {
        TrieNode {
            bits: (token_id << 8) | byte as u32,
            subtree_size: 0,
        }
    }

    #[inline(always)]
    unsafe fn next(&self) -> *const TrieNode {
        self.ptr().add(self.subtree_size as usize)
    }

    #[inline(always)]
    unsafe fn ptr(&self) -> *const TrieNode {
        self as *const TrieNode
    }

    #[inline(always)]
    unsafe fn child0(&self) -> *const TrieNode {
        self.ptr().add(1)
    }

    #[inline(always)]
    pub fn byte(&self) -> u8 {
        (self.bits & 0xff) as u8
    }

    #[inline(always)]
    pub fn token_id(&self) -> Option<u32> {
        let r = self.bits >> 8;
        if r == NO_TOKEN {
            None
        } else {
            Some(r)
        }
    }
}

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

    pub fn root(&self) -> &TrieNode {
        &self.data[0]
    }

    pub fn child_at_byte(&self, n: &TrieNode, byte: u8) -> Option<&TrieNode> {
        unsafe {
            let mut p = n.child0();
            let endp = n.next();
            while p < endp {
                if (*p).byte() == byte {
                    return Some(&*p);
                }
                p = (*p).next();
            }
        }
        None
    }

    pub fn child_at_bytes<'a>(&'a self, mut n: &'a TrieNode, bytes: &[u8]) -> Option<&'a TrieNode> {
        for &byte in bytes {
            n = match self.child_at_byte(n, byte) {
                Some(n) => n,
                None => return None,
            }
        }
        Some(n)
    }
}

pub fn append_bias(trie: &TokTrie, rec: &impl Recognizer, logits: &mut [f32]) {
    let n = trie.root();
    append_bias_core(rec, logits, n);
}

fn append_bias_core(rec: &impl Recognizer, logits: &mut [f32], n: &TrieNode) {
    unsafe {
        let endp = n.next();
        let mut p = n.child0();
        while p < endp {
            let n = &*p;
            p = n.next();
            let b = n.byte();
            if rec.allowed(b) {
                if let Some(tok) = n.token_id() {
                    logits[tok as usize] = 0.0;
                }
                if n.subtree_size > 1 {
                    append_bias_core(&rec.append(b), logits, n);
                }
            }
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
            if self.children.len() == 0x100 {
                // assert!(self.children[word[0] as usize].byte == word[0]);
                self.children[word[0] as usize].insert(&word[1..], token_id);
                return;
            }

            for ch in &mut self.children {
                if ch.byte == word[0] {
                    ch.insert(&word[1..], token_id);
                    return;
                }
            }

            let mut ch = TrieHash::new(word[0]);
            ch.insert(&word[1..], token_id);
            self.children.push(ch);

            // if it's getting dense, make it full
            // for cl100k threshold 60->15 nodes, 50->22, 40->45 30->94
            // for llama (32k) 50->5, 40->15
            // TODO remove this?
            if self.children.len() > 250 {
                let mut v2 = (0..=255).map(TrieHash::new).collect::<Vec<_>>();
                for ch in self.children.drain(..) {
                    let idx = ch.byte as usize;
                    v2[idx] = ch;
                }
                self.children = v2;
            }
        }
    }
    fn serialize(&mut self, data: &mut Vec<TrieNode>) {
        let idx = data.len();
        data.push(TrieNode::new(self.byte, self.token_id));
        self.children.sort_by_key(|e| e.byte);
        for entry in &mut self.children {
            entry.serialize(data);
        }
        data[idx].subtree_size = (data.len() - idx) as u32;
    }
}
