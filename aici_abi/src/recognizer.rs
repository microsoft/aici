use std::fmt::Debug;

use crate::{
    toktree::{Recognizer, SpecialToken, TokTrie},
    AiciVm, MidProcessArg, MidProcessResult, PostProcessArg, PostProcessResult,
};

pub struct AiciRecognizer<R: Recognizer> {
    pub trie: TokTrie,
    pub rec: R,
}

impl<R: Recognizer> AiciRecognizer<R> {
    pub fn from_recognizer(rec: R) -> Self {
        AiciRecognizer {
            trie: TokTrie::from_host(),
            rec,
        }
    }
}

impl<R: Recognizer + Clone> AiciVm for AiciRecognizer<R> {
    fn mid_process(&mut self, _arg: MidProcessArg) -> MidProcessResult {
        let mut set = self.trie.alloc_token_set();
        self.trie.compute_bias(&mut self.rec, &mut set);
        MidProcessResult::SampleWithBias {
            allowed_tokens: set,
        }
    }

    fn post_process(&mut self, arg: PostProcessArg) -> PostProcessResult {
        for token in arg.tokens {
            let bytes = self.trie.token(token);
            // wprintln!("process {} {:?}", token, bytes);
            for b in bytes {
                self.rec.push_byte(*b)
            }
            self.rec.collapse();
        }
        PostProcessResult {}
    }
}

pub trait FunctionalRecognizer<S: Copy> {
    /// Initial state
    fn initial(&self) -> S;
    /// Extend the recognizer with given byte.
    fn append(&self, state: S, byte: u8) -> S;
    /// Check if given byte is allowed in given state.
    fn byte_allowed(&self, state: S, byte: u8) -> bool;
    /// Check if given special token is allowed in given state.
    fn special_allowed(&self, state: S, tok: SpecialToken) -> bool;
}

#[derive(Clone)]
pub struct StackRecognizer<S: Copy, R: FunctionalRecognizer<S>> {
    rec: R,
    stack: Vec<S>,
    stack_ptr: usize,
}

impl<S: Copy, R: FunctionalRecognizer<S>> StackRecognizer<S, R> {
    pub fn from(rec: R) -> Self {
        let stack = vec![rec.initial(); 130];
        StackRecognizer {
            rec,
            stack,
            stack_ptr: 0,
        }
    }

    pub fn reset(&mut self) {
        self.stack_ptr = 0;
        self.stack[0] = self.rec.initial();
    }
}

impl<S: Copy + Debug, R: FunctionalRecognizer<S>> Recognizer for StackRecognizer<S, R> {
    #[inline(always)]
    fn push_byte(&mut self, byte: u8) {
        let state = self.stack[self.stack_ptr];
        let state = self.rec.append(state, byte);
        self.stack_ptr += 1;
        self.stack[self.stack_ptr] = state;
    }

    #[inline(always)]
    fn pop_bytes(&mut self, num: usize) {
        self.stack_ptr -= num;
    }

    #[inline(always)]
    fn byte_allowed(&mut self, byte: u8) -> bool {
        self.rec.byte_allowed(self.stack[self.stack_ptr], byte)
    }

    fn trie_finished(&mut self) {
        // wprintln!("{:?}", &self.stack[0..=self.stack_ptr]);
        assert!(self.stack_ptr == 0);
    }

    fn collapse(&mut self) {
        self.stack[0] = self.stack[self.stack_ptr];
        self.stack_ptr = 0;
    }

    fn special_allowed(&mut self, tok: SpecialToken) -> bool {
        self.rec.special_allowed(self.stack[self.stack_ptr], tok)
    }

    #[inline(always)]
    fn try_push_byte(&mut self, byte: u8) -> bool {
        if self.rec.byte_allowed(self.stack[self.stack_ptr], byte) {
            self.push_byte(byte);
            true
        } else {
            false
        }
    }
}

#[derive(Clone)]
pub struct AnythingGoes {}

impl FunctionalRecognizer<()> for AnythingGoes {
    fn initial(&self) -> () {
        ()
    }

    fn append(&self, state: (), _byte: u8) -> () {
        state
    }

    fn byte_allowed(&self, _state: (), _byte: u8) -> bool {
        true
    }

    fn special_allowed(&self, _state: (), _tok: SpecialToken) -> bool {
        true
    }
}
