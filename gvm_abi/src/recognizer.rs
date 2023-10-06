use std::rc::Rc;

use crate::{
    toktree::{append_bias, Recognizer, TokTrie},
    wprintln, GuidanceVm, GuidanceVmHelper,
};

#[inline(never)]
pub fn compute_bias<S: Copy>(
    trie: &TokTrie,
    rec: &impl Recognizer<S>,
    state: S,
    logits: &mut [f32],
) {
    logits.iter_mut().for_each(|x| *x = -100.0);
    append_bias(trie, rec, state, logits);
}

pub struct LenExcluder {}

impl Recognizer<u32> for LenExcluder {
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

pub struct GvmRecognizer<S: Copy, R: Recognizer<S> + Clone> {
    pub helper: GuidanceVmHelper,
    pub rec: Rc<Box<R>>,
    pub trie: Rc<Box<TokTrie>>,
    pub state: S,
}

impl<S: Copy, R: Recognizer<S> + Clone> GvmRecognizer<S, R> {
    pub fn from_recognizer(trie: Rc<Box<TokTrie>>, rec: Rc<Box<R>>) -> Self {
        let state = rec.as_ref().initial();
        GvmRecognizer {
            helper: GuidanceVmHelper::new(),
            rec,
            state,
            trie,
        }
    }

    fn compute(&mut self) {
        let trie = (*self.trie).as_ref();
        let rec = (*self.rec).as_ref();
        compute_bias(trie, rec, self.state, &mut self.helper.logit_biases);
    }
}

impl<S: Copy, R: Recognizer<S> + Clone> GuidanceVm for GvmRecognizer<S, R> {
    fn gvm_clone(&mut self) -> Self {
        GvmRecognizer {
            helper: self.helper.clone(),
            rec: self.rec.clone(),
            state: self.state,
            trie: self.trie.clone(),
        }
    }

    fn gvm_process_prompt(&mut self) {
        wprintln!("prompt, {} tokens", self.helper.prompt_length);
        // the regex doesn't care about the prompt
        self.state = self.rec.initial();
        self.compute();
    }

    fn gvm_append_token(&mut self, token: u32) {
        // wprintln!("xapp {:?} {} {}", self as *const _, token, self.state.off);
        let bytes = self.trie.token(token);
        for b in bytes {
            self.state = self.rec.append(self.state, *b);
        }

        // save the token, just in case
        let toks = &mut self.helper.tokens;
        toks.push(token);

        self.compute();
    }
}
