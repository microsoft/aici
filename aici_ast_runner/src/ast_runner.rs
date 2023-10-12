mod rx;

use serde::{Deserialize, Serialize};
use std::rc::Rc;

use crate::rx::RecRx;

use aici_abi::{
    aici_expose_all,
    recognizer::{AnythingGoes, FunctionalRecognizer, StackRecognizer},
    toktree::{Recognizer, SpecialToken, TokTrie},
    wprintln, AiciVm, AiciVmHelper,
};

#[derive(Serialize, Deserialize, Clone)]
pub enum Step {
    Fixed {
        text: String,
    },
    Gen {
        max_tokens: usize,
        rx: Option<String>,
    },
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Program {
    pub steps: Vec<Step>,
}

pub struct Runner {
    helper: AiciVmHelper,
    program: Program,
    curr_step: usize,
    trie: Rc<Box<TokTrie>>,
    rec: Box<dyn Recognizer>,
    tok_limit: usize,
}

#[derive(Clone)]
pub struct FixedArgs {
    pub text: String,
}

impl FunctionalRecognizer<u32> for FixedArgs {
    fn initial(&self) -> u32 {
        0
    }

    fn append(&self, state: u32, _byte: u8) -> u32 {
        state + 1
    }

    fn byte_allowed(&self, state: u32, byte: u8) -> bool {
        let idx = state as usize;
        let bytes = self.text.as_bytes();
        idx < bytes.len() && bytes[idx] == byte
    }

    fn special_allowed(&self, state: u32, tok: SpecialToken) -> bool {
        match tok {
            SpecialToken::EndOfSentence => state >= self.text.len() as u32,
            _ => false,
        }
    }
}

impl Runner {
    pub fn new(program: Program) -> Self {
        Self {
            helper: AiciVmHelper::new(),
            program,
            curr_step: 0,
            trie: Rc::new(Box::new(TokTrie::from_env())),
            rec: Box::new(StackRecognizer::from(AnythingGoes {})),
            tok_limit: 0,
        }
    }

    fn stop(&mut self) {
        let rec = StackRecognizer::from(FixedArgs {
            text: "".to_string(),
        });
        self.rec = Box::new(rec);
    }

    fn next_step(&mut self) {
        self.curr_step += 1;
        if self.curr_step < self.program.steps.len() {
            self.new_step();
        } else {
            self.stop();
        }
    }

    fn new_step(&mut self) {
        match &self.program.steps[self.curr_step] {
            Step::Fixed { text } => {
                let rec = StackRecognizer::from(FixedArgs { text: text.clone() });
                self.rec = Box::new(rec);
                self.tok_limit = 0; // no limit
            }
            Step::Gen { max_tokens, rx } => {
                if let Some(rx) = rx {
                    let rec = StackRecognizer::from(RecRx::from_rx(&rx));
                    self.rec = Box::new(rec);
                } else {
                    self.rec = Box::new(StackRecognizer::from(AnythingGoes {}));
                }
                self.tok_limit = *max_tokens;
            }
        }
    }

    fn compute(&mut self) {
        // wprintln!("compute");
        self.trie
            .compute_bias(&mut *self.rec, &mut self.helper.logit_biases);
    }
}

impl AiciVm for Runner {
    fn aici_clone(&mut self) -> Self {
        todo!()
    }

    fn aici_process_prompt(&mut self) {
        wprintln!("prompt, {} tokens", self.helper.prompt_length);
        // ignore the prompt (for now)
        self.new_step();
        self.compute();
    }

    fn aici_append_token(&mut self, token: u32) {
        let bytes = self.trie.token(token);
        wprintln!("xapp {} {:?}", token, bytes);

        // save the token, just in case
        let toks = &mut self.helper.tokens;
        toks.push(token);

        let rec = &mut *self.rec;
        for b in bytes {
            rec.push_byte(*b)
        }
        rec.collapse();

        if self.tok_limit == 1 {
            self.next_step();
        } else if (0..=255).all(|byte| !rec.byte_allowed(byte)) {
            if rec.special_allowed(SpecialToken::EndOfSentence) {
                self.next_step();
            } else {
                wprintln!("no more bytes allowed, but no end of sentence");
                self.stop();
            }
        } else {
            if self.tok_limit > 0 {
                self.tok_limit -= 1;
            }
        }

        self.compute();
    }

    fn get_helper(&mut self) -> &mut AiciVmHelper {
        &mut self.helper
    }
}

fn main() {
    let _run = sample_prog();
}

fn sample_prog() -> Runner {
    Runner::new(Program {
        steps: vec![
            Step::Fixed {
                text: "I am ".to_string(),
            },
            Step::Gen {
                max_tokens: 5,
                rx: Some(r#"\d\d"#.to_string()),
            },
            Step::Fixed {
                text: " years and ".to_string(),
            },
            Step::Gen {
                max_tokens: 5,
                rx: Some(r#"\d+"#.to_string()),
            },
        ],
    })
}

aici_expose_all!(Runner, sample_prog());
