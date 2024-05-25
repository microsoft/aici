use std::{collections::HashMap, rc::Rc};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VecHolder {
    data: Rc<Vec<u32>>,
}

pub struct VecHashMap {
    by_id: Vec<VecHolder>,
    by_data: HashMap<VecHolder, u32>,
}

impl VecHashMap {
    pub fn new() -> Self {
        let mut r = VecHashMap {
            by_id: Vec::new(),
            by_data: HashMap::new(),
        };
        r.insert(Vec::new());
        r
    }

    pub fn insert(&mut self, data: Vec<u32>) -> u32 {
        let holder = VecHolder {
            data: Rc::new(data),
        };
        if let Some(&id) = self.by_data.get(&holder) {
            return id;
        }
        let id = self.by_id.len() as u32;
        self.by_id.push(holder.clone());
        self.by_data.insert(holder, id);
        id
    }

    pub fn get(&self, id: u32) -> Option<&[u32]> {
        self.by_id.get(id as usize).map(|holder| &holder.data[..])
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }
}

pub trait VecNode {
    type Ref;
    fn serialize(&self) -> Vec<u32>;
    fn wrap_ref(v: u32) -> Self::Ref;
    fn unwrap_ref(r: Self::Ref) -> u32;
}

pub struct HashCons<Node: VecNode> {
    map: VecHashMap,
    _phantom1: std::marker::PhantomData<Node>,
}

impl<Node: VecNode> HashCons<Node> {
    pub fn new() -> Self {
        HashCons {
            map: VecHashMap::new(),
            _phantom1: std::marker::PhantomData,
        }
    }

    pub fn serialize(&self, node: &Node) -> Vec<u32> {
        node.serialize()
    }

    pub fn insert(&mut self, d: Vec<u32>) -> Node::Ref {
        let id = self.map.insert(d);
        Node::wrap_ref(id)
    }

    // pub fn insert<'s, 'a>(&'s mut self, node: &'a Node) -> Node::Ref {
    //     let data = node.serialize();
    //     let id = self.map.insert(data);
    //     Node::wrap_ref(id)
    // }

    pub fn get<'a>(&'a self, ref_: Node::Ref) -> Option<&'a [u32]> {
        self.map.get(Node::unwrap_ref(ref_))
    }
}
