use crate::{
    toktree::{append_bias, Recognizer, TokTrie},
    wprintln, GuidanceVm, GuidanceVmHelper,
};

#[inline(never)]
pub fn compute_bias<S: Copy>(
    trie: &TokTrie,
    rec: &mut impl Recognizer<S>,
    state: S,
    logits: &mut [f32],
) {
    logits.iter_mut().for_each(|x| *x = -100.0);
    append_bias(trie, rec, state, logits);
}

pub struct LenExcluder {}

impl Recognizer<u32> for LenExcluder {
    fn initial(&mut self) -> u32 {
        0
    }

    #[inline(never)]
    fn append(&mut self, state: u32, _byte: u8) -> u32 {
        state + 1
    }

    #[inline(never)]
    fn allowed(&mut self, state: u32, byte: u8) -> bool {
        byte != (('z' as u32 + state) & 0xff) as u8
    }
}

pub struct GvmRecognizer<'a, S: Copy, R: Recognizer<S> + Clone> {
    pub helper: GuidanceVmHelper,
    pub rec: &'a mut R,
    pub trie: &'a TokTrie,
    pub state: S,
}

impl<'a, S: Copy, R: Recognizer<S> + Clone> GvmRecognizer<'a, S, R> {
    pub fn from_recognizer(trie: &'a TokTrie, rec: &'a mut R) -> Self {
        let state = rec.initial();
        GvmRecognizer {
            helper: GuidanceVmHelper::new(),
            rec,
            state,
            trie,
        }
    }

    fn compute(&mut self) {
        compute_bias(
            self.trie,
            self.rec,
            self.state,
            &mut self.helper.logit_biases,
        );
    }
}

impl<'a, S: Copy, R: Recognizer<S> + Clone> GuidanceVm for GvmRecognizer<'a, S, R> {
    fn gvm_clone(&mut self) -> Self {
        GvmRecognizer {
            helper: self.helper.clone(),
            rec: self.rec,
            state: self.state,
            trie: self.trie,
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
