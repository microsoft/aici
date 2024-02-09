use std::fmt::Display;

use crate::{
    bytes::limit_str,
    recognizer::{FunctionalRecognizer, StackRecognizer},
    toktree::SpecialToken,
};
use serde_json::json;

enum Node {
    Inner { children: Vec<(u8, usize)> },
    Leaf { source_offset: usize },
}

pub struct SubStrMatcher {
    end_str: String,
    source: String,
    nodes: Vec<Node>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubStrState {
    Dead,
    Node(usize),
    SourceOffset(usize),
    EndStrOffset(usize),
}

pub type SubStrStackRecognizer = StackRecognizer<SubStrState, SubStrMatcher>;

fn add_node(nodes: &mut Vec<Node>, n: Node) -> usize {
    let idx = nodes.len();
    nodes.push(n);
    idx
}

impl Display for SubStrMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.pp(f, 0, 0)
    }
}

impl SubStrMatcher {
    #[allow(dead_code)]
    fn to_json(&self, node_idx: usize) -> serde_json::Value {
        match &self.nodes[node_idx] {
            Node::Inner { children } => {
                let mut children_json = serde_json::Map::new();
                for (c, idx) in children.iter() {
                    children_json.insert(format!("{}", *c as char), self.to_json(*idx));
                }
                serde_json::Value::Object(children_json)
            }
            Node::Leaf { source_offset } => {
                json!(limit_str(&self.source[*source_offset..], 20))
            }
        }
    }

    fn pp(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        indent: usize,
        node_idx: usize,
    ) -> std::fmt::Result {
        let node = &self.nodes[node_idx];
        match node {
            Node::Inner { children } => {
                for (c, idx) in children.iter() {
                    writeln!(f, "{:indent$}{:?} -> {}", "", *c as char, idx)?;
                    self.pp(f, indent + 1, *idx)?;
                }
            }
            Node::Leaf { source_offset } => {
                writeln!(
                    f,
                    "{:indent$}{}: {:?}",
                    "",
                    *source_offset,
                    limit_str(&self.source[*source_offset..], 20),
                )?;
            }
        }
        Ok(())
    }

    pub fn new(source: &str, end_str: &str) -> Self {
        let mut tmp = Self {
            source: source.to_string() + " ",
            end_str: end_str.to_string(),
            nodes: vec![Node::Inner { children: vec![] }],
        };
        tmp.add(0);
        for i in 0..tmp.source.len() {
            if tmp.source.as_bytes()[i] == b' ' {
                tmp.add(i + 1);
            }
        }
        // println!("{}", tmp);
        // println!("JSON: {}", serde_json::to_string(&tmp.to_json(0)).unwrap());
        tmp
    }

    fn find(&self, s: &str) -> (usize, usize) {
        let mut node_idx = 0;
        for (i, b) in s.bytes().enumerate() {
            let node = &self.nodes[node_idx];
            match node {
                Node::Inner { children } => {
                    let mut found = false;
                    for (c, idx) in children.iter() {
                        if *c == b {
                            node_idx = *idx;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        return (node_idx, i);
                    }
                }
                Node::Leaf { .. } => return (node_idx, i),
            }
        }
        (node_idx, s.len())
    }

    fn add(&mut self, source_offset1: usize) {
        let s1 = &self.source[source_offset1..];
        let (mut node_idx, offset) = self.find(s1);
        if offset >= s1.len() {
            return;
        }
        let source_offset1 = source_offset1 + offset;
        let s1 = &self.source[source_offset1..];

        let num_nodes = self.nodes.len();
        match &mut self.nodes[node_idx] {
            Node::Inner { children } => {
                children.push((s1.as_bytes()[0], num_nodes));
                let n = add_node(
                    &mut self.nodes,
                    Node::Leaf {
                        source_offset: source_offset1 + 1,
                    },
                );
                assert!(n == num_nodes);
            }
            Node::Leaf { source_offset } => {
                let source_offset2 = *source_offset;
                let s2 = &self.source[source_offset2..];
                if s2.starts_with(s1) {
                    return;
                }
                if s1.starts_with(s2) {
                    self.nodes[node_idx] = Node::Leaf {
                        source_offset: source_offset1,
                    };
                    return;
                }

                for i in 0..s1.len() {
                    let b1 = s1.as_bytes()[i];
                    let b2 = s2.as_bytes()[i];
                    if b1 != b2 {
                        let n1 = add_node(
                            &mut self.nodes,
                            Node::Leaf {
                                source_offset: source_offset1 + i + 1,
                            },
                        );
                        let n2 = add_node(
                            &mut self.nodes,
                            Node::Leaf {
                                source_offset: source_offset2 + i + 1,
                            },
                        );
                        self.nodes[node_idx] = Node::Inner {
                            children: vec![(b1, n1), (b2, n2)],
                        };
                        return;
                    } else {
                        let n1 = add_node(&mut self.nodes, Node::Inner { children: vec![] });
                        self.nodes[node_idx] = Node::Inner {
                            children: vec![(b1, n1)],
                        };
                        node_idx = n1;
                    }
                }
            }
        }
    }

    pub fn to_stack_recognizer(self) -> SubStrStackRecognizer {
        StackRecognizer::from(self)
    }

    fn append_to_src_off(&self, off: usize, byte: u8) -> SubStrState {
        if off < self.source.len() && self.source.as_bytes()[off] == byte {
            SubStrState::SourceOffset(off + 1)
        } else {
            SubStrState::Dead
        }
    }

    fn append_inner(&self, state: SubStrState, byte: u8) -> SubStrState {
        match state {
            SubStrState::Dead => SubStrState::Dead,
            SubStrState::EndStrOffset(off) => {
                if off < self.end_str.len() && self.end_str.as_bytes()[off] == byte {
                    SubStrState::EndStrOffset(off + 1)
                } else {
                    SubStrState::Dead
                }
            }
            SubStrState::Node(state) => {
                let node = &self.nodes[state];
                match node {
                    Node::Inner { children } => {
                        for (c, idx) in children.iter() {
                            if *c == byte {
                                return SubStrState::Node(*idx);
                            }
                        }
                        SubStrState::Dead
                    }
                    Node::Leaf { source_offset } => self.append_to_src_off(*source_offset, byte),
                }
            }
            SubStrState::SourceOffset(off) => self.append_to_src_off(off, byte),
        }
    }
}

impl FunctionalRecognizer<SubStrState> for SubStrMatcher {
    fn initial(&self) -> SubStrState {
        SubStrState::Node(0)
    }

    #[inline(always)]
    fn append(&self, state: SubStrState, byte: u8) -> SubStrState {
        let state = match state {
            SubStrState::Node(_) | SubStrState::SourceOffset(_)
                if self.end_str.as_bytes().first() == Some(&byte)
                    && self.append_inner(state, b' ') != SubStrState::Dead =>
            {
                SubStrState::EndStrOffset(0)
            }
            _ => state,
        };

        self.append_inner(state, byte)
    }

    #[inline(always)]
    fn byte_allowed(&self, state: SubStrState, byte: u8) -> bool {
        self.append(state, byte) != SubStrState::Dead
    }

    #[inline(always)]
    fn special_allowed(&self, state: SubStrState, tok: SpecialToken) -> bool {
        match tok {
            SpecialToken::EndOfSentence => {
                let l = self.end_str.len();
                if l == 0 {
                    self.append_inner(state, b' ') != SubStrState::Dead
                } else {
                    state == SubStrState::EndStrOffset(l)
                }
            }
            _ => false,
        }
    }
}
