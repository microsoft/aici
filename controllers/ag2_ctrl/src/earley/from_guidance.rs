use super::Grammar;
use crate::{earley::grammar::SymbolProps, grammar::TopLevelGrammar};
use anyhow::Result;

#[derive(Debug)]
pub struct NodeProps {
    pub nullable: bool,
    pub name: String,
    pub hidden: bool,
    pub commit_point: bool,
    pub capture_name: String,
    pub max_tokens: i32,
    pub temperature: f32,
}

impl NodeProps {
    #[allow(dead_code)]
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
            capture_name: if self.capture_name.is_empty() {
                None
            } else {
                Some(self.capture_name.clone())
            },
            temperature: self.temperature,
        }
    }
}

pub fn earley_grm_from_guidance(_bytes: TopLevelGrammar) -> Result<Grammar> {
    panic!();
    // let mut reader = quick_protobuf::BytesReader::from_bytes(bytes);
    // let gg = guidance::Grammar::from_reader(&mut reader, bytes).unwrap();
    // let mut grm = Grammar::new();

    // let symbols = gg
    //     .nodes
    //     .iter()
    //     .map(|n| {
    //         let term = match &n.function_type {
    //             OneOffunction_type::byte(n) => {
    //                 assert!(n.byte.len() == 1);
    //                 Some(grm.terminal(&ByteSet::from_range(n.byte[0], n.byte[0])))
    //             }
    //             OneOffunction_type::byte_range(n) => {
    //                 assert!(n.byte_range.len() == 2);
    //                 Some(grm.terminal(&ByteSet::from_range(n.byte_range[0], n.byte_range[1])))
    //             }
    //             OneOffunction_type::model_variable(n) => Some(grm.model_variable(&n.name)),
    //             _ => None,
    //         };
    //         let props = NodeProps::from_grammar_function(&n.function_type);
    //         let sym_props = props.to_symbol_props();
    //         let name = sym_props.capture_name.as_ref().unwrap_or(&props.name);
    //         // println!("props: {:?}", props);
    //         let sym = if let Some(term) = term {
    //             assert!(props.max_tokens == i32::MAX, "max_tokens on terminal");
    //             if sym_props.is_special() {
    //                 let wrap = grm.fresh_symbol(if name.is_empty() { "t_wrap" } else { name });
    //                 grm.add_rule(term, vec![term]);
    //                 wrap
    //             } else {
    //                 term
    //             }
    //         } else {
    //             assert!(name.len() > 0, "empty name");
    //             grm.fresh_symbol(name)
    //         };
    //         grm.apply_props(sym, sym_props);
    //         sym
    //     })
    //     .collect::<Vec<_>>();

    // for (n, sym) in gg.nodes.iter().zip(symbols.iter()) {
    //     let lhs = *sym;
    //     match &n.function_type {
    //         OneOffunction_type::join(n) => {
    //             let rhs = n.values.iter().map(|idx| symbols[*idx as usize]).collect();
    //             grm.add_rule(lhs, rhs);
    //         }
    //         OneOffunction_type::select(n) => {
    //             if n.nullable {
    //                 grm.add_rule(lhs, vec![]);
    //             }
    //             for v in &n.values {
    //                 grm.add_rule(lhs, vec![symbols[*v as usize]]);
    //             }
    //         }
    //         OneOffunction_type::byte(_)
    //         | OneOffunction_type::byte_range(_)
    //         | OneOffunction_type::model_variable(_) => {}
    //         OneOffunction_type::None => panic!("???"),
    //     }
    // }

    // grm.add_rule(grm.start(), vec![symbols[0]]);

    // Ok(grm)
}
