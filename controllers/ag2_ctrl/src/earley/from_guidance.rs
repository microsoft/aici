use super::{
    lexer::{quote_regex, LexerSpec},
    Grammar,
};
use crate::{
    api::{GrammarWithLexer, Node, TopLevelGrammar},
    earley::{grammar::SymbolProps, lexer::LexemeSpec},
};
use anyhow::{bail, ensure, Result};

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

pub fn grammar_from_json(input: GrammarWithLexer) -> Result<Grammar> {
    let is_greedy = input.greedy_lexer;
    let is_lazy = !is_greedy;

    let mut grm = Grammar::new(LexerSpec {
        greedy: is_greedy,
        lexemes: vec![],
    });
    let node_map = input
        .nodes
        .iter()
        .enumerate()
        .map(|(idx, n)| {
            let props = n.node_props();
            let name = match props.name.as_ref() {
                Some(n) => n.clone(),
                None if props.capture_name.is_some() => {
                    props.capture_name.as_ref().unwrap().clone()
                }
                None => format!("n{}", idx),
            };
            let symprops = SymbolProps {
                commit_point: false,
                hidden: false,
                max_tokens: props.max_tokens.unwrap_or(usize::MAX),
                model_variable: None,
                capture_name: props.capture_name.clone(),
                temperature: 0.0,
            };
            grm.fresh_symbol_ext(&name, symprops)
        })
        .collect::<Vec<_>>();

    for (n, sym) in input.nodes.iter().zip(node_map.iter()) {
        let lhs = *sym;
        match &n {
            Node::Select { among, .. } => {
                ensure!(among.len() > 0, "empty select");
                for v in among {
                    grm.add_rule(lhs, vec![node_map[v.0]]);
                }
            }
            Node::Join { sequence, .. } => {
                let rhs = sequence.iter().map(|idx| node_map[idx.0]).collect();
                grm.add_rule(lhs, rhs);
            }
            Node::Gen { data, .. } => {
                ensure!(is_lazy, "gen() only allowed in lazy grammars");
                let body_rx = if data.body_rx.is_empty() {
                    ".*"
                } else {
                    &data.body_rx
                };
                let info = LexemeSpec {
                    rx: format!("({})({})", body_rx, data.stop_rx),
                    ends_at_eos_only: data.stop_rx.is_empty(),
                    allow_others: false,
                };
                grm.make_terminal(lhs, info)?;
                let symprops = grm.sym_props_mut(lhs);
                if let Some(t) = data.temperature {
                    symprops.temperature = t;
                }
            }
            Node::Lexeme {
                rx, allow_others, ..
            } => {
                ensure!(is_greedy, "lexeme() only allowed in greedy grammars");
                let info = LexemeSpec {
                    rx: rx.clone(),
                    ends_at_eos_only: false,
                    allow_others: *allow_others,
                };
                grm.make_terminal(lhs, info)?;
            }
            Node::String { literal, .. } => {
                let info = LexemeSpec {
                    rx: quote_regex(&literal),
                    ends_at_eos_only: false,
                    allow_others: false,
                };
                grm.make_terminal(lhs, info)?;
            }
            Node::GrammarRef { grammar_id, .. } => {
                let _ = grammar_id;
                bail!("TODO: grammar ref")
            }
        }
    }
    Ok(grm)
}

pub fn grammars_from_json(input: TopLevelGrammar) -> Result<Vec<Grammar>> {
    input.grammars.into_iter().map(grammar_from_json).collect()
}
