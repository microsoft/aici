use toktrie::{self, Recognizer};
use serde::{Deserialize, Serialize};

use super::Parser;
use crate::{
    api::{GenGrammarOptions, TopLevelGrammar},
    earley::from_guidance::grammars_from_json,
};

#[derive(Serialize, Deserialize)]
struct RunnerArg {
    grammar: TopLevelGrammar,
}

pub fn earley_test(trie: toktrie::TokTrie) {
    let g_bytes = include_bytes!("../../grammars/json.json");
    let data: RunnerArg = serde_json::from_slice(g_bytes).unwrap();
    let cfg = grammars_from_json(data.grammar, true).unwrap();

    let input = "{\n    \"name\": \"John Doe\",\n    \"age\": 30,\n    \"armor\": \"leather\",\n    \"weapon\": \"sword\",\n    \"class\": \"warrior\",\n    \"mantra\": \"I am the master of my fate, I am the captain of my soul.\",\n    \"strength\": 18,\n    \"items\": [\"health potion\", \"mana potion\", \"bandages\"]".as_bytes();

    let toks = trie.greedy_tokenize(input);
    println!("tokens: {:?}", toks.len());

    let grm = cfg[0].clone();

    let mut parser = Parser::new(grm.clone(), GenGrammarOptions::default()).unwrap();

    if false {
        parser.trie_started();
        let a = parser.try_push_byte(b'{');
        let b = parser.try_push_byte(b'{');
        println!("a: {}, b: {}", a, b);
        parser.pop_bytes(2);
        parser.trie_finished();
    }

    if true {
        let mut ts = trie.alloc_token_set();
        trie.compute_bias(&mut parser, &mut ts);
        println!("bias: {}", trie.token_set_dbg(&ts));
    }

    for b in input {
        if !parser.try_push_byte_definitive(Some(*b)) {
            panic!("reject");
        }
    }

    let forced = parser.force_bytes();
    println!("forced: {:?}", String::from_utf8_lossy(&forced));

    println!("final: {:?}", String::from_utf8_lossy(&parser.get_bytes()));

    if !parser.is_accepting() {
        panic!("final non-accept");
    }

    if true {
        return;
    }

    const COLLECT_TIMES: bool = false;
    const NUM_REP: usize = if COLLECT_TIMES { 5 } else { 500 };
    let mut durations = vec![];
    let mut durations_us = vec![];
    println!("start!");

    let num_tok = 4;

    for _ in 0..NUM_REP {
        let mut line = 1;
        let mut vob = trie.alloc_token_set();

        parser = Parser::new(grm.clone(), GenGrammarOptions::default()).unwrap();
        let mut times = vec![];

        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();

        for (idx, tok) in toks.iter().take(num_tok).enumerate() {
            let tok = *tok;
            let tt = std::time::Instant::now();
            trie.compute_bias(&mut parser, &mut vob);
            if idx == num_tok - 1 {
                durations_us.push(tt.elapsed().as_micros() as u64);
            }
            // parser.print_stats();
            if !vob.is_allowed(tok) {
                println!("reject, line={}, tok={:?}", line, trie.token_str(tok));
                panic!();
            }
            for b in trie.token(tok) {
                if *b == b'\n' {
                    line += 1;
                }
            }
            // println!(
            //     "TOK: {} ===> {}",
            //     trie.token_dbg(tok),
            //     trie.token_set_dbg(&vob)
            // );
            trie.append_token(&mut parser, tok).unwrap();
            if COLLECT_TIMES {
                times.push(tt.elapsed().as_micros() as u32);
            }
        }

        durations.push(t0.elapsed().as_micros() as u64);

        if COLLECT_TIMES {
            println!("times: {:?}", times);
        }
    }

    durations.sort();
    durations_us.sort();

    let min_us = *durations_us.iter().min().unwrap();
    // println!("min_time_us: {:?}", min_us);
    // for ~5ms 0.1ms is the precision we expect
    println!("min_time_ms: {:.1}", min_us as f64 / 1000.0);
}
