// use 8:24 encoding - num_ch:tok_id (ch_byte:ch_off)* - 8 bytes per tree node
// special case num_ch=0xff -> num_ch=0x100

use crate::{
    recognizer::Recognizer,
    rx::{box_from_bytes, clone_as_bytes, clone_vec_as_bytes, vec_from_bytes, TokRxInfo, TokenId},
};

pub struct TokTrie {
    info: TokRxInfo,
    token_offsets: Vec<u32>,
    token_data: Vec<u8>,
    nodes: Vec<TrieNode>,
}

#[repr(C)]
pub struct TokTrieHeader {
    magic: u32,
    hd_size: u32,
    trie_bytes: u32,
    token_offset_bytes: u32,
    token_data_bytes: u32,
    info: TokRxInfo,
    align: [u32; 0],
}

impl TokTrieHeader {
    const MAGIC: u32 = 0x558b6fd3;
}

#[repr(C)]
pub struct TrieNode {
    // byte:token
    bits: u32,
    bits2: u32,
}

const NO_TOKEN: u32 = 0xffffff;

impl TrieNode {
    fn new(byte: u8, token_id: u32, num_parents: u8) -> TrieNode {
        TrieNode {
            bits: (token_id << 8) | byte as u32,
            bits2: num_parents as u32,
        }
    }

    #[inline(always)]
    unsafe fn next(&self) -> *const TrieNode {
        self.ptr().add(self.subtree_size())
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
    pub fn subtree_size(&self) -> usize {
        (self.bits2 >> 8) as usize
    }

    #[inline(always)]
    pub fn num_parents(&self) -> usize {
        (self.bits2 & 0xff) as usize
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
    pub fn from(info: &TokRxInfo, words: &Vec<Vec<u8>>) -> Self {
        let mut trie = TrieHash::new(0xff);
        let mut token_offsets = Vec::new();
        let mut token_data = Vec::new();
        println!("info: {:?} wl={}", info, words.len());
        assert!(info.vocab_size == words.len() as u32);
        for (idx, word) in words.iter().enumerate() {
            if word.len() > 0 {
                trie.insert(word, idx as u32)
            }
            assert!(word.len() < 0xff);
            let desc = (word.len() as u32) | ((token_data.len() as u32) << 8);
            token_offsets.push(desc);
            token_data.extend_from_slice(word);
        }
        let mut nodes = Vec::new();
        trie.serialize(&mut nodes, 1);
        let r = TokTrie {
            info: info.clone(),
            token_offsets,
            token_data,
            nodes,
        };
        r.validate();
        r
    }

    pub fn info(&self) -> &TokRxInfo {
        &self.info
    }

    pub fn vocab_size(&self) -> usize {
        self.info.vocab_size as usize
    }

    pub fn token(&self, idx: u32) -> &[u8] {
        let off = self.token_offsets[idx as usize];
        let len = off & 0xff;
        let off = (off >> 8) as usize;
        &self.token_data[off..(off + len as usize)]
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let pref = std::mem::size_of::<TokTrieHeader>();
        let hd = *box_from_bytes::<TokTrieHeader>(&bytes[0..pref]);
        assert!(hd.magic == TokTrieHeader::MAGIC);
        assert!(hd.hd_size as usize == pref);

        let trie_end = pref + hd.trie_bytes as usize;
        let nodes = vec_from_bytes(&bytes[pref..trie_end]);
        let offsets_end = trie_end + hd.token_offset_bytes as usize;
        let token_offsets = vec_from_bytes(&bytes[trie_end..offsets_end]);
        let token_data = vec_from_bytes(&bytes[offsets_end..]);

        let r = TokTrie {
            info: hd.info,
            token_offsets,
            token_data,
            nodes,
        };
        r.validate();
        r
    }

    fn validate_node(&self, n: &TrieNode, ep: *const TrieNode, used: &mut [bool]) {
        if let Some(tok) = n.token_id() {
            assert!(tok < self.info.vocab_size);
            assert!(!used[tok as usize]);
            used[tok as usize] = true;
        }
        unsafe {
            let endp = n.next();
            assert!(endp <= ep);
            let mut p = n.child0();
            while p < endp {
                let n = &*p;
                p = n.next();
                self.validate_node(n, endp, used)
            }
        }
    }

    fn validate(&self) {
        self.validate_node(
            self.root(),
            self.nodes.as_ptr_range().end,
            &mut vec![false; self.info.vocab_size as usize],
        );
        for idx in 0..self.info.vocab_size {
            let _ = self.token(idx);
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut trie_data = clone_vec_as_bytes(&self.nodes);
        let mut token_offsets = clone_vec_as_bytes(&self.token_offsets);
        let mut token_data = clone_vec_as_bytes(&self.token_data);

        let hd = TokTrieHeader {
            magic: TokTrieHeader::MAGIC,
            hd_size: std::mem::size_of::<TokTrieHeader>() as u32,
            trie_bytes: trie_data.len() as u32,
            token_offset_bytes: token_offsets.len() as u32,
            token_data_bytes: trie_data.len() as u32,
            info: self.info.clone(),
            align: [],
        };

        let mut bytes = clone_as_bytes(&hd);
        bytes.append(&mut trie_data);
        bytes.append(&mut token_offsets);
        bytes.append(&mut token_data);
        bytes
    }

    pub fn root(&self) -> &TrieNode {
        &self.nodes[0]
    }

    pub fn check_against(&self, tokens: &Vec<Vec<u8>>) {
        let vocab_size = tokens.len();
        for idx in 0..vocab_size {
            let bytes = &tokens[idx];
            let tid = idx as TokenId;
            assert!(bytes == self.token(tid));
            let root = self.root();
            if bytes.len() > 0 {
                assert!(
                    self.child_at_bytes(root, &bytes)
                        .unwrap()
                        .token_id()
                        .unwrap()
                        == tid
                );
            }
        }
    }

    pub fn child_at_byte<'a>(&'a self, n: &'a TrieNode, byte: u8) -> Option<&'a TrieNode> {
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

pub fn append_bias<T: Recognizer + Copy>(trie: &TokTrie, rec0: T, logits: &mut [f32]) {
    let n = trie.root();
    let mut stack_buf = [rec0; 130];
    let mut stack_ptr = 1;
    let defl_tok = trie.vocab_size() as u32;
    unsafe {
        let mut p = n.child0();
        let endp = n.next();
        while p < endp {
            let n = &*p;
            let b = n.byte();
            let rec = &stack_buf[stack_ptr - 1];
            if rec.allowed(b) {
                logits[n.token_id().unwrap_or(defl_tok) as usize] = 0.0;
                stack_buf[stack_ptr] = rec.append(b);
                stack_ptr += 1;
                if n.subtree_size() == 1 {
                    stack_ptr -= n.num_parents();
                }
                p = n.child0();
            } else {
                p = n.next();
                stack_ptr -= n.num_parents() - 1;
            }
        }
        //panic!("st: {}", stack_ptr);
    }
}

fn walk_core(n: &TrieNode) -> u32 {
    let mut sum = 0;
    let mut stack_buf: [(*const TrieNode, *const TrieNode); 130] = [(0 as _, 0 as _); 130];
    let mut stack_ptr = 0;
    unsafe {
        stack_buf[stack_ptr] = (n.child0(), n.next());
        stack_ptr += 1;
        loop {
            stack_ptr -= 1;
            let (mut p, mut endp) = stack_buf[stack_ptr];
            while p < endp {
                let n = &*p;
                p = n.next();
                sum += n.subtree_size() as u32;
                if n.subtree_size() > 1 {
                    stack_buf[stack_ptr] = (p, endp);
                    stack_ptr += 1;
                    endp = p;
                    p = n.child0();
                }
            }
            if stack_ptr == 0 {
                break;
            }
        }
    }
    sum
}

pub fn walk(trie: &TokTrie) -> u32 {
    let n = trie.root();
    walk_core(n)
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
    fn serialize(&mut self, data: &mut Vec<TrieNode>, num_parents: u8) {
        let idx = data.len();
        let mut num_ch = self.children.len();
        data.push(TrieNode::new(self.byte, self.token_id, num_parents));
        self.children.sort_by_key(|e| e.byte);
        for entry in &mut self.children {
            num_ch -= 1;
            entry.serialize(data, if num_ch == 0 { num_parents + 1 } else { 1 });
        }
        data[idx].bits2 |= ((data.len() - idx) as u32) << 8;
    }
}
