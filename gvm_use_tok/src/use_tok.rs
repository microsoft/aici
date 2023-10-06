use gvm_abi::{
    printing::init_panic,
    recognizer::compute_bias,
    toktree::{Recognizer, TokTrie},
    wprintln,
};
use regex_automata::{
    dfa::{dense, Automaton},
    util::{primitives::StateID, syntax},
};

struct RecRx {
    dfa: dense::DFA<Vec<u32>>,
}

impl Recognizer<StateID> for RecRx {
    fn initial(&self) -> StateID {
        self.dfa
            .universal_start_state(regex_automata::Anchored::Yes)
            .unwrap()
    }

    fn append(&self, state: StateID, byte: u8) -> StateID {
        self.dfa.next_state(state, byte)
    }

    fn allowed(&self, state: StateID, byte: u8) -> bool {
        !self.dfa.is_dead_state(self.dfa.next_state(state, byte))
    }
}

fn main() {
    init_panic();

    let trie = TokTrie::from_env();
    for idx in 1000..1001 {
        let bytes = trie.token(idx);
        wprintln!("{}: {:?}", idx, String::from_utf8_lossy(bytes));
    }

    let _rx = r#"\{\n?"name": "(\\(["\\\/bfnrt]|u[a-fA-F0-9]{4})|[^"\\\x00-\x1F\x7F]+)*",\n"valid": (true|false),\n"type": "(foo|bar|baz|something|else)",\n"address": \{\n?"street": "(\\(["\\\/bfnrt]|u[a-fA-F0-9]{4})|[^"\\\x00-\x1F\x7F]+)*",\n"city": "(\\(["\\\/bfnrt]|u[a-fA-F0-9]{4})|[^"\\\x00-\x1F\x7F]+)*",\n"state": "([A-Z][A-Z])"\n?\},\n"age": \d+\n?\}"#;

    let rx = r#"[^X]*"#;

    let dfa = dense::Builder::new()
        .configure(dense::Config::new().start_kind(regex_automata::dfa::StartKind::Anchored))
        .syntax(syntax::Config::new().unicode(false).utf8(false))
        .build(&rx)
        .unwrap();
    wprintln!("dfa: {} bytes", dfa.memory_usage());

    let rec = RecRx { dfa };
    let mut logits = vec![0.0; trie.vocab_size() + 1];
    // let mut rec = LenExcluder {};
    for _ in 0..1000 {
        let s = rec.initial();
        compute_bias(&trie, &rec, s, &mut logits);
    }

    let count = logits.iter().filter(|x| **x > -50.0).count();
    wprintln!("resx: {}", count);
    if count < 100 {
        for (idx, logit) in logits.iter().enumerate() {
            if *logit > -50.0 {
                let bytes = trie.token(idx as u32);
                wprintln!("{}: {:?}", idx, String::from_utf8_lossy(bytes));
            }
        }
    }
}
