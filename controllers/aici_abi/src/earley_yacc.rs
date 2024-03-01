use anyhow::Result;
use cfgrammar::Symbol;

use crate::{
    cfg::{parse_rx_token, parse_yacc},
    earley::Grammar,
    toktree::TokTrie,
};

pub fn earley_grm_from_yacc(yacc: &str) -> Result<Grammar> {
    let grm = parse_yacc(yacc)?;

    let mut res = Grammar::new();

    for pidx in grm.iter_pidxs() {
        let ridx = grm.prod_to_rule(pidx);

        let lhs = res.symbol(grm.rule_name_str(ridx));
        let rhs = grm
            .prod(pidx)
            .iter()
            .map(|sym| match sym {
                Symbol::Token(tidx) => {
                    let name = grm.token_name(*tidx).unwrap();
                    let t = res.symbol(name);
                    res.make_terminal(t, &parse_rx_token(name));
                    t
                }
                Symbol::Rule(ridx) => res.symbol(grm.rule_name_str(*ridx)),
            })
            .collect();

        res.add_rule(lhs, rhs);
    }

    let start_sym = grm.rule_name_str(grm.start_rule_idx());
    println!("start_sym: {:?}", start_sym);
    let ss = res.symbol(start_sym);
    res.add_rule(res.start(), vec![ss]);

    Ok(res)
}

#[allow(dead_code)]
pub fn earley_test(trie: TokTrie) {
    let yacc_bytes = include_bytes!("../grammars/c.y");
    let cfg = earley_grm_from_yacc(&String::from_utf8_lossy(yacc_bytes)).unwrap();

    println!("cfg: {:?}", cfg);

    let sample = include_bytes!("../grammars/sample.c");
    let toks = trie.greedy_tokenize(sample);

    println!("toks: {:?}", toks.len());

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
