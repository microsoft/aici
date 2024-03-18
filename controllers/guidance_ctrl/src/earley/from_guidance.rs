use anyhow::Result;
use quick_protobuf::MessageRead;
use rustc_hash::FxHashSet;

use super::{ByteSet, Grammar};
use crate::{
    earley::grammar::SymbolProps,
    serialization::guidance::{self, mod_GrammarFunction::OneOffunction_type},
};

#[derive(Debug)]
pub struct NodeProps {
    pub nullable: bool,
    pub name: String,
    pub hidden: bool,
    pub commit_point: bool,
    pub capture_name: String,
    pub max_tokens: i32,
}

impl NodeProps {
    pub fn from_grammar_function(function_type: &OneOffunction_type) -> Self {
        let mut r = match function_type {
            OneOffunction_type::join(n) => NodeProps {
                nullable: n.nullable,
                name: n.name.to_string(),
                hidden: n.hidden,
                commit_point: n.commit_point,
                capture_name: n.capture_name.to_string(),
                max_tokens: n.max_tokens,
            },
            OneOffunction_type::select(n) => NodeProps {
                nullable: n.nullable,
                name: n.name.to_string(),
                hidden: n.hidden,
                commit_point: n.commit_point,
                capture_name: n.capture_name.to_string(),
                max_tokens: n.max_tokens,
            },
            OneOffunction_type::byte(n) => NodeProps {
                nullable: n.nullable,
                name: "".to_string(),
                hidden: n.hidden,
                commit_point: n.commit_point,
                capture_name: n.capture_name.to_string(),
                max_tokens: i32::MAX,
            },
            OneOffunction_type::byte_range(n) => NodeProps {
                nullable: false, // n.nullable,
                name: "".to_string(),
                hidden: n.hidden,
                commit_point: n.commit_point,
                capture_name: n.capture_name.to_string(),
                max_tokens: i32::MAX,
            },
            OneOffunction_type::model_variable(n) => NodeProps {
                nullable: n.nullable,
                name: n.name.to_string(),
                hidden: n.hidden,
                commit_point: n.commit_point,
                capture_name: n.capture_name.to_string(),
                max_tokens: i32::MAX,
            },
            OneOffunction_type::None => {
                panic!("None function type in guidance::Grammar")
            }
        };
        if r.max_tokens >= 1_000_000 {
            // guidance is very liberal with unspecified max_tokens, sometimes it's 10m, sometimes it's 100m
            r.max_tokens = i32::MAX;
        }
        r
    }

    pub fn to_symbol_props(&self) -> SymbolProps {
        SymbolProps {
            commit_point: self.commit_point,
            hidden: self.hidden && self.commit_point,
            max_tokens: if self.max_tokens == i32::MAX {
                usize::MAX
            } else {
                self.max_tokens.try_into().unwrap()
            },
            model_variable: None,
        }
    }
}

pub fn earley_grm_from_guidance(bytes: &[u8]) -> Result<Grammar> {
    let mut reader = quick_protobuf::BytesReader::from_bytes(bytes);
    let gg = guidance::Grammar::from_reader(&mut reader, bytes).unwrap();
    let mut grm = Grammar::new();

    let symbols = gg
        .nodes
        .iter()
        .map(|n| {
            let term = match &n.function_type {
                OneOffunction_type::byte(n) => {
                    assert!(n.byte.len() == 1);
                    Some(grm.terminal(&ByteSet::from_range(n.byte[0], n.byte[0])))
                }
                OneOffunction_type::byte_range(n) => {
                    assert!(n.byte_range.len() == 2);
                    Some(grm.terminal(&ByteSet::from_range(n.byte_range[0], n.byte_range[1])))
                }
                OneOffunction_type::model_variable(n) => Some(grm.model_variable(&n.name)),
                _ => None,
            };
            let props = NodeProps::from_grammar_function(&n.function_type);
            // println!("props: {:?}", props);
            let sym = if let Some(term) = term {
                assert!(props.max_tokens == i32::MAX, "max_tokens on terminal");
                if props.commit_point {
                    let wrap = grm.fresh_symbol("t_wrap");
                    grm.add_rule(term, vec![term]);
                    wrap
                } else {
                    term
                }
            } else {
                assert!(props.name.len() > 0, "empty name");
                grm.fresh_symbol(&props.name)
            };
            grm.apply_props(sym, props.to_symbol_props());
            sym
        })
        .collect::<Vec<_>>();

    let set = FxHashSet::from_iter(symbols.iter());
    assert!(set.len() == symbols.len(), "duplicate symbols");

    for (n, sym) in gg.nodes.iter().zip(symbols.iter()) {
        let lhs = *sym;
        match &n.function_type {
            OneOffunction_type::join(n) => {
                let rhs = n.values.iter().map(|idx| symbols[*idx as usize]).collect();
                grm.add_rule(lhs, rhs);
            }
            OneOffunction_type::select(n) => {
                if n.nullable {
                    grm.add_rule(lhs, vec![]);
                }
                for v in &n.values {
                    grm.add_rule(lhs, vec![symbols[*v as usize]]);
                }
            }
            OneOffunction_type::byte(_)
            | OneOffunction_type::byte_range(_)
            | OneOffunction_type::model_variable(_) => {}
            OneOffunction_type::None => panic!("???"),
        }
    }

    grm.add_rule(grm.start(), vec![symbols[0]]);

    Ok(grm)
}
