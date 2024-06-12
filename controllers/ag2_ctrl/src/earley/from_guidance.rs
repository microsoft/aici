use std::{rc::Rc, vec};

use super::{grammar::SymbolProps, lexerspec::LexerSpec, CGrammar, Grammar};
use crate::api::{GrammarWithLexer, Node, TopLevelGrammar, DEFAULT_CONTEXTUAL};
use anyhow::{ensure, Result};
use derivre::RegexAst;

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

fn grammar_from_json(input: GrammarWithLexer) -> Result<(LexerSpec, Grammar)> {
    let is_greedy = input.greedy_lexer;
    let is_lazy = !is_greedy;
    let skip = match input.greedy_skip_rx {
        Some(rx) if is_greedy => RegexAst::Regex(rx),
        _ => RegexAst::NoMatch,
    };
    let mut lexer_spec = LexerSpec::new(is_greedy, skip)?;
    let mut grm = Grammar::new();
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
                // TODO add some optimization to throw these away?
                // ensure!(among.len() > 0, "empty select");
                for v in among {
                    grm.add_rule(lhs, vec![node_map[v.0]])?;
                }
            }
            Node::Join { sequence, .. } => {
                let rhs = sequence.iter().map(|idx| node_map[idx.0]).collect();
                grm.add_rule(lhs, rhs)?;
            }
            Node::Gen { data, .. } => {
                // parser backtracking relies on only lazy lexers having hidden lexemes
                ensure!(is_lazy, "gen() only allowed in lazy grammars");
                let body_rx = if data.body_rx.is_empty() {
                    ".*"
                } else {
                    &data.body_rx
                };
                let idx = lexer_spec.add_rx_and_stop(
                    format!("gen_{}", grm.sym_name(lhs)),
                    body_rx,
                    &data.stop_rx,
                )?;
                grm.make_terminal(lhs, idx)?;
                let symprops = grm.sym_props_mut(lhs);
                if let Some(t) = data.temperature {
                    symprops.temperature = t;
                }
            }
            Node::Lexeme { rx, contextual, .. } => {
                ensure!(is_greedy, "lexeme() only allowed in greedy grammars");
                let idx = lexer_spec.add_greedy_lexeme(
                    format!("lex_{}", grm.sym_name(lhs)),
                    rx,
                    contextual.unwrap_or(input.contextual.unwrap_or(DEFAULT_CONTEXTUAL)),
                )?;
                grm.make_terminal(lhs, idx)?;
            }
            Node::String { literal, .. } => {
                let idx = lexer_spec.add_simple_literal(
                    format!("str_{}", grm.sym_name(lhs)),
                    &literal,
                    input.contextual.unwrap_or(DEFAULT_CONTEXTUAL),
                )?;
                grm.make_terminal(lhs, idx)?;
            }
            Node::GenGrammar { data, props } => {
                let mut data = data.clone();
                data.max_tokens_grm = props.max_tokens.unwrap_or(usize::MAX);
                grm.make_gen_grammar(lhs, data)?;
            }
        }
    }
    Ok((lexer_spec, grm))
}

pub fn grammars_from_json(input: TopLevelGrammar, print_out: bool) -> Result<Vec<Rc<CGrammar>>> {
    let grammars = input
        .grammars
        .into_iter()
        .map(grammar_from_json)
        .collect::<Result<Vec<_>>>()?;

    for (_, g) in &grammars {
        g.validate_grammar_refs(&grammars)?;
    }

    Ok(grammars
        .into_iter()
        .enumerate()
        .map(|(idx, (lex, mut grm))| {
            if print_out {
                println!("\nGrammar #{}:\n{:?}\n{:?}", idx, lex, grm);
            }

            grm = grm.optimize();

            if print_out {
                println!("  == Optimize ==>\n{:?}", grm);
            }

            Rc::new(grm.compile(lex))
        })
        .collect::<Vec<_>>())
}
