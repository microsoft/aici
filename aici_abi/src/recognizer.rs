use std::{fmt::Debug, rc::Rc};

use crate::{
    toktree::{Recognizer, SpecialToken, TokTrie},
    wprintln, AiciVm, AiciVmHelper,
};

pub struct LenExcluder {}

impl FunctionalRecognizer<u32> for LenExcluder {
    fn initial(&self) -> u32 {
        0
    }

    #[inline(never)]
    fn append(&self, state: u32, _byte: u8) -> u32 {
        state + 1
    }

    #[inline(never)]
    fn byte_allowed(&self, state: u32, byte: u8) -> bool {
        byte != (('z' as u32 + state) & 0xff) as u8
    }

    #[inline(never)]
    fn special_allowed(&self, state: u32, tok: SpecialToken) -> bool {
        match tok {
            SpecialToken::EndOfSentence => state < 10,
            _ => false,
        }
    }
}

pub struct AiciRecognizer<R: Recognizer> {
    pub helper: AiciVmHelper,
    pub rec: R,
    pub trie: Rc<Box<TokTrie>>,
}

impl<R: Recognizer> AiciRecognizer<R> {
    pub fn from_recognizer(trie: Rc<Box<TokTrie>>, rec: R) -> Self {
        AiciRecognizer {
            helper: AiciVmHelper::new(),
            rec,
            trie,
        }
    }

    fn compute(&mut self) {
        // wprintln!("compute");
        self.trie
            .compute_bias(&mut self.rec, &mut self.helper.logit_biases);
    }
}

impl<R: Recognizer + Clone> AiciVm for AiciRecognizer<R> {
    fn aici_process_prompt(&mut self) {
        wprintln!("prompt, {} tokens", self.helper.prompt_length);
        // the regex doesn't care about the prompt
        self.compute();
    }

    fn aici_append_token(&mut self, token: u32) {
        let bytes = self.trie.token(token);
        // wprintln!("xapp {} {:?}", token, bytes);
        for b in bytes {
            self.rec.push_byte(*b)
        }
        self.rec.collapse();

        // save the token, just in case
        let toks = &mut self.helper.tokens;
        toks.push(token);

        self.compute();
    }

    fn get_helper(&mut self) -> &mut AiciVmHelper {
        &mut self.helper
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
