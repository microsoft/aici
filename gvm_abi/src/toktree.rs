// use 8:24 encoding - num_ch:tok_id (ch_idx:ch_off)* - 8 bytes per token
// special case num_ch=0xff -> num_ch=0x100

use crate::rx::TokenId;

pub struct TokNode {
    pub byte: u8,
    off: usize,
    data: &'static [u32],
}

impl TokNode {
    const NO_TOKEN: u32 = 0xffffff;

    pub fn token_id(&self) -> Option<TokenId> {
        let r = self.data[self.off] >> 8;
        if r == Self::NO_TOKEN {
            None
        } else {
            Some(r)
        }
    }

    pub fn num_children(&self) -> usize {
        let num_ch = self.data[self.off] & 0xff;
        if num_ch == 0xff {
            0x100
        } else {
            num_ch as usize
        }
    }

    pub fn child_at_idx(&self, idx: usize) -> TokNode {
        assert!(idx < self.num_children());
        let off = self.off + 1 + idx;
        let ch_off = self.data[off] >> 8;
        TokNode {
            byte: (self.data[off] & 0xff) as u8,
            off: ch_off as usize,
            data: self.data,
        }
    }

    pub fn child_at_byte(&self, byte: u8) -> Option<TokNode> {
        let num_ch = self.num_children();
        for idx in 0..num_ch {
            let off = self.off + 1 + idx;
            if (self.data[off] & 0xff) as u8 == byte {
                return Some(self.child_at_idx(idx));
            }
        }
        None
    }

    pub fn children(&self) -> TokNodeChildrenIter {
        TokNodeChildrenIter {
            parent: self,
            idx: 0,
            max_idx: self.num_children(),
        }
    }
}

pub struct TokNodeChildrenIter<'a> {
    parent: &'a TokNode,
    idx: usize,
    max_idx: usize,
}

impl<'a> Iterator for TokNodeChildrenIter<'a> {
    type Item = TokNode;

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
