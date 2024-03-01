use anyhow::Result;
use quick_protobuf::MessageRead;
use rustc_hash::FxHashSet;

use crate::{
    earley::{
        guidance,
        parser::{ByteSet, Parser},
    },
    toktree::TokTrie,
};

use super::parser::Grammar;

pub fn earley_grm_from_guidance(bytes: &[u8]) -> Result<Grammar> {
    let mut reader = quick_protobuf::BytesReader::from_bytes(bytes);
    let gg = guidance::Grammar::from_reader(&mut reader, bytes).unwrap();
    let mut grm = Grammar::new();

    let symbols = gg
        .nodes
        .iter()
        .map(|n| match &n.function_type {
            guidance::mod_GrammarFunction::OneOffunction_type::join(n) => grm.fresh_symbol(&n.name),
            guidance::mod_GrammarFunction::OneOffunction_type::select(n) => {
                grm.fresh_symbol(&n.name)
            }
            guidance::mod_GrammarFunction::OneOffunction_type::byte(n) => {
                assert!(n.byte.len() == 1);
                grm.terminal(ByteSet::from_range(n.byte[0], n.byte[0]))
            }
            guidance::mod_GrammarFunction::OneOffunction_type::byte_range(n) => {
                assert!(n.byte_range.len() == 2);
                grm.terminal(ByteSet::from_range(n.byte_range[0], n.byte_range[1]))
            }
            guidance::mod_GrammarFunction::OneOffunction_type::model_variable(n) => {
                grm.fresh_symbol(&n.name)
            }
            guidance::mod_GrammarFunction::OneOffunction_type::None => {
                panic!("None function type in guidance::Grammar")
            }
        })
        .collect::<Vec<_>>();

    let set = FxHashSet::from_iter(symbols.iter());
    assert!(set.len() == symbols.len(), "duplicate symbols");

    for (n, sym) in gg.nodes.iter().zip(symbols.iter()) {
        let lhs = *sym;
        match &n.function_type {
            guidance::mod_GrammarFunction::OneOffunction_type::join(n) => {
                if n.nullable {
                    //println!("nullable join: {:?}", n.name);
                }
                let rhs = n.values.iter().map(|idx| symbols[*idx as usize]).collect();
                grm.add_rule(lhs, rhs);
            }
            guidance::mod_GrammarFunction::OneOffunction_type::select(n) => {
                if n.nullable {
                    // println!("nullable sel: {:?} {:?}", n.name, n.values);
                    grm.add_rule(lhs, vec![]);
                }
                for v in &n.values {
                    grm.add_rule(lhs, vec![symbols[*v as usize]]);
                }
            }
            guidance::mod_GrammarFunction::OneOffunction_type::byte(_) => {}
            guidance::mod_GrammarFunction::OneOffunction_type::byte_range(_) => {}
            guidance::mod_GrammarFunction::OneOffunction_type::model_variable(n) => {
                // eos_token, bos_token etc
                panic!("model_variable not implemented yet ({:?})", n.name);
            }
            guidance::mod_GrammarFunction::OneOffunction_type::None => panic!("???"),
        }
    }

    grm.add_rule(grm.start(), vec![symbols[0]]);

    Ok(grm)
}

#[allow(dead_code)]
pub fn earley_test(trie: TokTrie) {
    let g_bytes = include_bytes!("../../grammars/json0.guidance");
    let cfg = earley_grm_from_guidance(g_bytes).unwrap();
    println!("cfg0: {:?}", cfg);
    let cfg = cfg.optimize();
    println!("cfg: {:?}", cfg);

    let input = r#"{"name":"Joe","info":{"foo":10,"bar":"20"}}"#.as_bytes();

    let toks = trie.greedy_tokenize(input);
    println!("toks: {:?}", toks.len());

    let mut parser = Parser::new(cfg);
    for b in input {
        let row = parser.scan(*b);
        if row.is_empty() {
            println!("reject");
            break;
        }
        println!("row: {}", parser.row_to_string(&row));
        parser.push_row(row);
    }

    // #[cfg(not(target_arch = "wasm32"))]
    // let t0 = std::time::Instant::now();

    // let mut line = 1;
    // let mut vob = trie.alloc_token_set();

    // for tok in &toks[0..1000] {
    //     let tok = *tok;
    //     trie.compute_bias(&mut cfg, &mut vob);
    //     if !vob.is_allowed(tok) {
    //         println!("reject, line={}, tok={:?}", line, trie.token_str(tok));
    //         panic!();
    //     }
    //     for b in trie.token(tok) {
    //         if *b == b'\n' {
    //             line += 1;
    //         }
    //     }
    //     if false {
    //         println!(
    //             "tok: {:?} {}; {}",
    //             trie.token_str(tok),
    //             vob.is_allowed(tok),
    //             cfg.get_stats()
    //         );
    //         cfg.viable_now();
    //     }
    //     trie.append_token(&mut cfg, tok);
    // }

    // #[cfg(not(target_arch = "wasm32"))]
    // println!("time: {:?} ", t0.elapsed());

    // println!("stats:  {}", cfg.get_stats());
}
