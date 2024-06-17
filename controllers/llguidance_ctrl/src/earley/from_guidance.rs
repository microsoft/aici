use std::{sync::Arc, vec};

use super::{grammar::SymbolProps, lexerspec::LexerSpec, CGrammar, Grammar};
use crate::api::{
    GrammarWithLexer, Node, RegexId, RegexNode, RegexSpec, TopLevelGrammar, DEFAULT_CONTEXTUAL,
};
use anyhow::{bail, ensure, Result};
use derivre::{ExprRef, RegexAst, RegexBuilder};

fn resolve_rx(rx_refs: &[ExprRef], node: &RegexSpec) -> Result<RegexAst> {
    match node {
        RegexSpec::Regex(rx) => Ok(RegexAst::Regex(rx.clone())),
        RegexSpec::RegexId(id) => {
            if id.0 >= rx_refs.len() {
                bail!("invalid regex id: {}", id.0)
            } else {
                Ok(RegexAst::ExprRef(rx_refs[id.0]))
            }
        }
    }
}

fn map_rx_ref(rx_refs: &[ExprRef], id: RegexId) -> Result<RegexAst> {
    if id.0 >= rx_refs.len() {
        bail!("invalid regex id when building nodes: {}", id.0)
    } else {
        Ok(RegexAst::ExprRef(rx_refs[id.0]))
    }
}

fn map_rx_refs(rx_refs: &[ExprRef], ids: Vec<RegexId>) -> Result<Vec<RegexAst>> {
    ids.into_iter().map(|id| map_rx_ref(rx_refs, id)).collect()
}

fn map_rx_nodes(rx_nodes: Vec<RegexNode>) -> Result<(RegexBuilder, Vec<ExprRef>)> {
    let mut builder = RegexBuilder::new();
    let mut rx_refs = vec![];
    for node in rx_nodes {
        rx_refs.push(builder.mk(&map_node(&rx_refs, node)?)?);
    }
    return Ok((builder, rx_refs));

    fn map_node(rx_refs: &[ExprRef], node: RegexNode) -> Result<RegexAst> {
        match node {
            RegexNode::Not(id) => Ok(RegexAst::Not(Box::new(map_rx_ref(rx_refs, id)?))),
            RegexNode::Repeat(id, min, max) => Ok(RegexAst::Repeat(
                Box::new(map_rx_ref(rx_refs, id)?),
                min,
                max.unwrap_or(u32::MAX),
            )),
            RegexNode::EmptyString => Ok(RegexAst::EmptyString),
            RegexNode::NoMatch => Ok(RegexAst::NoMatch),
            RegexNode::Regex(rx) => Ok(RegexAst::Regex(rx)),
            RegexNode::Literal(lit) => Ok(RegexAst::Literal(lit)),
            RegexNode::Byte(b) => Ok(RegexAst::Byte(b)),
            RegexNode::ByteSet(bs) => Ok(RegexAst::ByteSet(bs)),
            RegexNode::ByteLiteral(bs) => Ok(RegexAst::ByteLiteral(bs)),
            RegexNode::And(lst) => Ok(RegexAst::And(map_rx_refs(rx_refs, lst)?)),
            RegexNode::Concat(lst) => Ok(RegexAst::Concat(map_rx_refs(rx_refs, lst)?)),
            RegexNode::Or(lst) => Ok(RegexAst::Or(map_rx_refs(rx_refs, lst)?)),
            RegexNode::LookAhead(id) => Ok(RegexAst::LookAhead(Box::new(map_rx_ref(rx_refs, id)?))),
        }
    }
}

fn grammar_from_json(input: GrammarWithLexer) -> Result<(LexerSpec, Grammar)> {
    let is_greedy = input.greedy_lexer;
    let is_lazy = !is_greedy;

    let (builder, rx_nodes) = map_rx_nodes(input.rx_nodes)?;

    let skip = match input.greedy_skip_rx {
        Some(rx) if is_greedy => resolve_rx(&rx_nodes, &rx)?,
        _ => RegexAst::NoMatch,
    };
    let mut lexer_spec = LexerSpec::new(is_greedy, builder, skip)?;
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
                stop_capture_name: None,
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
                let body_rx = if data.body_rx.is_missing() {
                    RegexAst::Regex(".*".to_string())
                } else {
                    resolve_rx(&rx_nodes, &data.body_rx)?
                };
                let stop_rx = if data.stop_rx.is_missing() {
                    RegexAst::NoMatch
                } else {
                    resolve_rx(&rx_nodes, &data.stop_rx)?
                };
                let idx = lexer_spec.add_rx_and_stop(
                    format!("gen_{}", grm.sym_name(lhs)),
                    body_rx,
                    stop_rx,
                )?;

                let symprops = grm.sym_props_mut(lhs);
                if let Some(t) = data.temperature {
                    symprops.temperature = t;
                }
                if data.stop_capture_name.is_some() {
                    symprops.stop_capture_name = data.stop_capture_name.clone();
                    let wrap_props = symprops.for_wrapper();
                    let wrap_name = format!("stop_wrap_{}", grm.sym_name(lhs));
                    let wrap_sym = grm.fresh_symbol_ext(&wrap_name, wrap_props);
                    grm.make_terminal(wrap_sym, idx)?;
                    grm.add_rule(lhs, vec![wrap_sym])?;
                } else {
                    grm.make_terminal(lhs, idx)?;
                }
            }
            Node::Lexeme { rx, contextual, .. } => {
                ensure!(is_greedy, "lexeme() only allowed in greedy grammars");
                let idx = lexer_spec.add_greedy_lexeme(
                    format!("lex_{}", grm.sym_name(lhs)),
                    resolve_rx(&rx_nodes, rx)?,
                    contextual.unwrap_or(input.contextual.unwrap_or(DEFAULT_CONTEXTUAL)),
                )?;
                grm.make_terminal(lhs, idx)?;
            }
            Node::String { literal, .. } => {
                if literal.is_empty() {
                    grm.add_rule(lhs, vec![])?;
                } else {
                    let idx = lexer_spec.add_simple_literal(
                        format!("str_{}", grm.sym_name(lhs)),
                        &literal,
                        input.contextual.unwrap_or(DEFAULT_CONTEXTUAL),
                    )?;
                    grm.make_terminal(lhs, idx)?;
                }
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

pub fn grammars_from_json(input: TopLevelGrammar, print_out: bool) -> Result<Vec<Arc<CGrammar>>> {
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

            Arc::new(grm.compile(lex))
        })
        .collect::<Vec<_>>())
}
