use std::rc::Rc;

use crate::{
    toktree::{Recognizer, TokTrie},
    wprintln, GuidanceVm, GuidanceVmHelper,
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
    fn allowed(&self, state: u32, byte: u8) -> bool {
        byte != (('z' as u32 + state) & 0xff) as u8
    }
}

pub struct GvmRecognizer<R: Recognizer> {
    pub helper: GuidanceVmHelper,
    pub rec: R,
    pub trie: Rc<Box<TokTrie>>,
}

impl<R: Recognizer> GvmRecognizer<R> {
    pub fn from_recognizer(trie: Rc<Box<TokTrie>>, rec: R) -> Self {
        GvmRecognizer {
            helper: GuidanceVmHelper::new(),
            rec,
            trie,
        }
    }

    fn compute(&mut self) {
        self.trie
            .compute_bias(&mut self.rec, &mut self.helper.logit_biases);
    }
}

impl<R: Recognizer + Clone> GuidanceVm for GvmRecognizer<R> {
    fn gvm_clone(&mut self) -> Self {
        GvmRecognizer {
            helper: self.helper.clone(),
            rec: self.rec.clone(),
            trie: self.trie.clone(),
        }
    }

    fn gvm_process_prompt(&mut self) {
        wprintln!("prompt, {} tokens", self.helper.prompt_length);
        // the regex doesn't care about the prompt
        self.compute();
    }

    fn gvm_append_token(&mut self, token: u32) {
        let bytes = self.trie.token(token);
        wprintln!("xapp {} {:?}", token, bytes);
        for b in bytes {
            self.rec.push_byte(*b)
        }

        // save the token, just in case
        let toks = &mut self.helper.tokens;
        toks.push(token);

        self.compute();
    }

    fn get_helper(&mut self) -> &mut GuidanceVmHelper {
        &mut self.helper
    }
}

pub trait FunctionalRecognizer<S: Copy> {
    fn initial(&self) -> S;
    fn append(&self, state: S, byte: u8) -> S;
    fn allowed(&self, state: S, byte: u8) -> bool;
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

impl<S: Copy, R: FunctionalRecognizer<S>> Recognizer for StackRecognizer<S, R> {
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
        self.rec.allowed(self.stack[self.stack_ptr], byte)
    }
}
