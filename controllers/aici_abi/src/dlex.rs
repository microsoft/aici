use crate::{
    recognizer::{FunctionalRecognizer, StackRecognizer},
    toktrie::SpecialToken,
    SimpleVob,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u32);

impl NodeId {
    const NULL: NodeId = NodeId(0);
    const ROOT: NodeId = NodeId(1);
}

#[derive(Debug, Default, Clone)]
pub struct NodeData {
    pub is_terminal: bool,
}

enum TrieNode {
    Sparse {
        data: NodeData,
        children: Vec<(u8, NodeId)>,
    },
    Dense {
        data: NodeData,
        children: Vec<NodeId>,
    },
}

impl TrieNode {
    fn new_dense(data: NodeData, children: &Vec<(u8, NodeId)>) -> Self {
        let mut dense_children = vec![NodeId::NULL; 256];
        for (byte, node_id) in children {
            dense_children[*byte as usize] = *node_id;
        }
        TrieNode::Dense {
            data,
            children: dense_children,
        }
    }

    fn new_leaf() -> Self {
        TrieNode::Sparse {
            data: NodeData::default(),
            children: vec![],
        }
    }

    fn data(&self) -> &NodeData {
        match self {
            TrieNode::Sparse { data, .. } => data,
            TrieNode::Dense { data, .. } => data,
        }
    }

    fn data_mut(&mut self) -> &mut NodeData {
        match self {
            TrieNode::Sparse { data, .. } => data,
            TrieNode::Dense { data, .. } => data,
        }
    }
}

pub struct Trie {
    nodes: Vec<TrieNode>,
}

impl Trie {
    const MAX_SPARSE: usize = 8;

    pub fn new() -> Self {
        Trie {
            nodes: vec![
                TrieNode::new_leaf(),
                TrieNode::new_dense(NodeData::default(), &vec![]),
            ],
        }
    }

    fn node(&self, node_id: NodeId) -> &TrieNode {
        &self.nodes[node_id.0 as usize]
    }

    fn node_mut(&mut self, node_id: NodeId) -> &mut TrieNode {
        &mut self.nodes[node_id.0 as usize]
    }

    pub fn node_data(&self, node_id: NodeId) -> &NodeData {
        self.node(node_id).data()
    }

    pub fn root(&self) -> NodeId {
        NodeId::ROOT
    }

    pub fn child_at(&self, start: NodeId, b: u8) -> Option<NodeId> {
        match self.node(start) {
            TrieNode::Sparse { children, .. } => {
                children.iter().find_map(
                    |&(byte, node_id)| {
                        if byte == b {
                            Some(node_id)
                        } else {
                            None
                        }
                    },
                )
            }
            TrieNode::Dense { children, .. } => {
                let node_id = children[b as usize];
                if node_id == NodeId::NULL {
                    None
                } else {
                    Some(node_id)
                }
            }
        }
    }

    pub fn lookup(&self, start: NodeId, word: &[u8]) -> Option<NodeId> {
        let mut node_id = start;
        for &byte in word {
            match self.child_at(node_id, byte) {
                Some(child_id) => {
                    node_id = child_id;
                }
                None => {
                    return None;
                }
            }
        }
        Some(node_id)
    }

    pub fn add(&mut self, word: &[u8]) {
        let mut node_id = NodeId::ROOT;
        for &byte in word {
            let new_node_id = NodeId(self.nodes.len() as u32);
            let node = self.node_mut(node_id);
            match node {
                TrieNode::Sparse { data, children } => {
                    match children.iter().find(|&&(b, _)| b == byte) {
                        Some(&(_, child_id)) => {
                            node_id = child_id;
                        }
                        None => {
                            children.push((byte, new_node_id));
                            if children.len() > Trie::MAX_SPARSE {
                                self.nodes[node_id.0 as usize] =
                                    TrieNode::new_dense(data.clone(), children);
                            }
                            self.nodes.push(TrieNode::new_leaf());
                            node_id = new_node_id;
                        }
                    }
                }
                TrieNode::Dense { children, .. } => {
                    node_id = children[byte as usize];
                    if node_id == NodeId::NULL {
                        children[byte as usize] = new_node_id;
                        self.nodes.push(TrieNode::new_leaf());
                        node_id = new_node_id;
                    }
                }
            }
        }

        self.node_mut(node_id).data_mut().is_terminal = true;
    }
}

pub struct DynamicLexer {
    trie: Trie,
    id_start: SimpleVob,
    id_body: SimpleVob,
}

#[derive(Debug, Clone, Copy)]
pub struct DState {
    node_id: NodeId,
}

impl DState {
    const ROOT: DState = DState {
        node_id: NodeId::ROOT,
    };
}

pub type DynamicLexerRec = StackRecognizer<DState, DynamicLexer>;

impl DynamicLexer {
    pub fn new(additional_id_chars: &Vec<char>) -> Self {
        let mut id_start = SimpleVob::alloc(0x100);
        let mut id_body = SimpleVob::alloc(0x100);
        for i in 0..=255u8 {
            match i as char {
                'a'..='z' | 'A'..='Z' | '_' => {
                    id_start.allow_token(i as u32);
                    id_body.allow_token(i as u32);
                }
                '0'..='9' => {
                    id_body.allow_token(i as u32);
                }
                _ => {}
            }
        }
        for &c in additional_id_chars {
            id_start.allow_token(c as u32);
            id_body.allow_token(c as u32);
        }
        DynamicLexer {
            trie: Trie::new(),
            id_start,
            id_body,
        }
    }

    pub fn to_stack_recognizer(self) -> StackRecognizer<DState, DynamicLexer> {
        StackRecognizer::from(self)
    }

    pub fn add(&mut self, word: &[u8]) {
        self.trie.add(word);
    }
}

impl FunctionalRecognizer<DState> for DynamicLexer {
    fn initial(&self) -> DState {
        DState::ROOT
    }

    fn try_append(&self, state: DState, byte: u8) -> Option<DState> {
        if state.node_id == NodeId::ROOT {
            if self.id_start.is_allowed(byte as u32) {
                match self.trie.child_at(state.node_id, byte) {
                    Some(node_id) => Some(DState { node_id }),
                    None => None,
                }
            } else {
                Some(state)
            }
        } else {
            if self.id_body.is_allowed(byte as u32) {
                match self.trie.child_at(state.node_id, byte) {
                    Some(node_id) => Some(DState { node_id }),
                    None => None,
                }
            } else {
                if self.trie.node_data(state.node_id).is_terminal {
                    Some(DState::ROOT)
                } else {
                    None
                }
            }
        }
    }

    fn special_allowed(&self, state: DState, tok: SpecialToken) -> bool {
        if tok == SpecialToken::EndOfSentence {
            self.trie.node_data(state.node_id).is_terminal
        } else {
            false
        }
    }
}
