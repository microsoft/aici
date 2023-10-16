use std::rc::Rc;

use aici_abi::{
    aici_expose_all, aici_harness,
    recognizer::{FunctionalRecognizer, AiciRecognizer, StackRecognizer},
    toktree::{SpecialToken, TokTrie},
    wprintln, AiciVm,
};
use regex_automata::{
    dfa::{dense, Automaton},
    util::{primitives::StateID, syntax},
};

#[derive(Clone)]
pub struct RecRx {
    dfa: dense::DFA<Vec<u32>>,
}

pub type RxRecognizer = AiciRecognizer<StackRecognizer<StateID, RecRx>>;

impl RecRx {
    pub fn from_rx(rx: &str) -> Self {
        let dfa = dense::Builder::new()
            .configure(dense::Config::new().start_kind(regex_automata::dfa::StartKind::Anchored))
            .syntax(syntax::Config::new().unicode(false).utf8(false))
            .build(&rx)
            .unwrap();
        wprintln!("dfa: {} bytes", dfa.memory_usage());
        RecRx { dfa }
    }

    pub fn to_recognizer(self) -> RxRecognizer {
        let trie = Rc::new(Box::new(TokTrie::from_host()));
        AiciRecognizer::<StackRecognizer<StateID, RecRx>>::from_recognizer(
            trie,
            StackRecognizer::from(self),
        )
    }
}

impl FunctionalRecognizer<StateID> for RecRx {
    fn initial(&self) -> StateID {
        self.dfa
            .universal_start_state(regex_automata::Anchored::Yes)
            .unwrap()
    }

    #[inline(always)]
    fn append(&self, state: StateID, byte: u8) -> StateID {
        self.dfa.next_state(state, byte)
    }

    #[inline(always)]
    fn byte_allowed(&self, state: StateID, byte: u8) -> bool {
        !self.dfa.is_dead_state(self.dfa.next_state(state, byte))
    }

    #[inline(always)]
    fn special_allowed(&self, state: StateID, tok: SpecialToken) -> bool {
        match tok {
            // if in dead state, stop
            SpecialToken::EndOfSentence => {
                self.dfa.is_match_state(state) || self.dfa.is_dead_state(state)
            }
            _ => false,
        }
    }
}

fn main() {
    let trie = TokTrie::from_host();
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

    let mut t = tokrx();
    aici_harness(&mut t, trie.vocab_size(), &vec![1, 2]);

    if false {
        let mut rec = StackRecognizer::from(RecRx { dfa });
        let mut logits = vec![0.0; trie.vocab_size() + 1];

        for _ in 0..1000 {
            rec.reset();
            trie.compute_bias(&mut rec, &mut logits);
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
}

fn tokrx() -> RxRecognizer {
    let rx = r#"\{\n?"name": "(\\(["\\\/bfnrt]|u[a-fA-F0-9]{4})|[^"\\\x00-\x1F\x7F]+)*",\n"valid": (true|false),\n"type": "(foo|bar|baz|something|else)",\n"address": \{\n?"street": "(\\(["\\\/bfnrt]|u[a-fA-F0-9]{4})|[^"\\\x00-\x1F\x7F]+)*",\n"city": "(\\(["\\\/bfnrt]|u[a-fA-F0-9]{4})|[^"\\\x00-\x1F\x7F]+)*",\n"state": "([A-Z][A-Z])"\n?\},\n"age": \d+\n?\}"#;
    // let rx = r#"[^X]*"#;
    RecRx::from_rx(rx).to_recognizer()
}

aici_expose_all!(RxRecognizer, tokrx());
