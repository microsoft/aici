use aici_abi::{recognizer::FunctionalRecognizer, toktree::SpecialToken, wprintln};
use regex_automata::{
    dfa::{dense, Automaton},
    util::{primitives::StateID, syntax},
};

pub type RecRxState = StateID;

#[derive(Clone)]
pub struct RecRx {
    dfa: dense::DFA<Vec<u32>>,
}

impl RecRx {
    pub fn from_rx(rx: &str) -> Self {
        let dfa = dense::Builder::new()
            .configure(dense::Config::new().start_kind(regex_automata::dfa::StartKind::Anchored))
            .syntax(syntax::Config::new().unicode(false).utf8(false))
            .build(&rx)
            .unwrap();
        wprintln!("dfa: {} bytes", dfa.memory_usage());
        Self { dfa }
    }
}

impl FunctionalRecognizer<RecRxState> for RecRx {
    fn initial(&self) -> RecRxState {
        self.dfa
            .universal_start_state(regex_automata::Anchored::Yes)
            .unwrap()
    }

    #[inline(always)]
    fn append(&self, state: RecRxState, byte: u8) -> RecRxState {
        self.dfa.next_state(state, byte)
    }

    #[inline(always)]
    fn byte_allowed(&self, state: RecRxState, byte: u8) -> bool {
        !self.dfa.is_dead_state(self.dfa.next_state(state, byte))
    }

    #[inline(always)]
    fn special_allowed(&self, state: RecRxState, tok: SpecialToken) -> bool {
        match tok {
            // if in dead state, stop
            SpecialToken::EndOfSentence => {
                self.dfa.is_match_state(state) || self.dfa.is_dead_state(state)
            }
            _ => false,
        }
    }
}
