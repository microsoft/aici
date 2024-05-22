use std::collections::HashMap;

use bytemuck_derive::{Pod, Zeroable};

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(transparent)]
pub struct HashNode(u32);

pub struct HashConstructor {
    data: Vec<u32>,
    hash: HashMap<Vec<u32>, u32>,
}

pub struct HashNodeRef<'a> {
    constructor: &'a HashConstructor,
    node: HashNode,
}

impl<'a> HashNodeRef<'a> {
    pub fn head(&self) -> u8 {
        self.constructor.head(self.node)
    }

    pub fn children(&self) -> &[HashNode] {
        self.constructor.children(self.node)
    }

    pub fn iter(&'a self) -> impl Iterator<Item = HashNodeRef<'a>> {
        self.children().iter().map(move |&n| self.constructor.node_ref(n))
    }
}

pub type HeadType = u8;

impl HashConstructor {
    fn mk_head(&self, head: HeadType, arity: usize) -> u32 {
        assert!(arity < 1 << 20);
        (head as u32) | ((arity as u32) << 8)
    }

    pub fn node_ref(&self, node: HashNode) -> HashNodeRef {
        HashNodeRef {
            constructor: self,
            node,
        }
    }

    pub fn mk(&mut self, head: HeadType, children: &[HashNode]) -> HashNode {
        let mut data: Vec<u32> = Vec::with_capacity(1 + children.len());
        data.push(self.mk_head(head, children.len()));
        for child in children {
            data.push(child.0);
        }
        if let Some(r) = self.hash.get(&data) {
            HashNode(*r)
        } else {
            let r = self.data.len() as u32;
            self.data.extend_from_slice(&data);
            self.hash.insert(data, r);
            HashNode(r)
        }
    }

    pub fn get(&self, node: HashNode) -> (HeadType, &[HashNode]) {
        let idx = node.0 as usize;
        let head = self.data[idx];
        let arity = (head >> 8) as usize;
        let head = head as u8;
        let children = bytemuck::cast_slice(&self.data[idx + 1..idx + 1 + arity]);
        (head, children)
    }

    pub fn head(&self, node: HashNode) -> HeadType {
        let idx = node.0 as usize;
        (self.data[idx] & 0xff) as HeadType
    }

    pub fn children(&self, node: HashNode) -> &[HashNode] {
        let idx = node.0 as usize;
        let arity = (self.data[idx] >> 8) as usize;
        bytemuck::cast_slice(&self.data[idx + 1..idx + 1 + arity])
    }
}
