/*
- byte-level constraint that forces a specific string can cause unusual tokenization, for example:
  "I am about" will likely force the model to output ["I", " am", " a", "b", "out"] instead of ["I", " am", " about"]
  This is because after "I am" we allow both " a" and " about" but the model finds " a" much more likely
- this is a problem, since the model can later get confused with [" a", "b", "out"] tokens being used instead of [" about"]
- this could be fixed by giving boost to longer tokens, but that doesn't work for regexps: simplest example
  ".*" would always prefer the longest possible token (eg. 128 space token in gpt4 case)
- thus, the "Fixed" step shouldn't be implemented at byte level, and instead tokenize the string and then force
  these specific tokens; the regexps in "Gen" should avoid long forced strings

- the tokenization algorithm is not simply the greedy longest prefix - it breaks string into "words", splits words
  into single-byte tokens and then merges adjecnt pairs of tokens in order of token number, see
  https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py
- in all tokenizers (gpt4, llama, phi, ...), all tokens fit one of these 3 categories:
  - only whitespace (not only ' ', but also '\n', '\t' etc)
  - start with a ' '
  - have no ' '

- we could have a warning when the token encoding is not optimal

- Gen("\d+"); Fixed(" years") should be equivalent to Gen("\d+ years")? (modulo token problems above)
*/
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

    fn stop(&mut self, info: &str) {
        let rec = StackRecognizer::from(FixedArgs {
            text: "".to_string(),
        });
        self.rec = Box::new(rec);
        wprintln!("stop: {}", info)
    }

    fn next_step(&mut self, info: &str) {
        self.curr_step += 1;
        if self.curr_step < self.program.steps.len() {
            wprintln!("step -> {}; {}", self.curr_step, info);
            self.new_step();
        } else {
            self.stop(info);
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

    fn all_disallowed(&mut self) {
        self.helper
            .logit_biases
            .iter_mut()
            .for_each(|x| *x = -100.0);
    }

    fn compute(&mut self) {
        // wprintln!("compute");
        self.all_disallowed();
        self.trie
            .add_bias(&mut *self.rec, &mut self.helper.logit_biases);
    }

    #[allow(dead_code)]
    fn log_prob(&self, tok: &str) {
        if let Some(id) = self.trie.token_id(tok.as_bytes()) {
            wprintln!(
                "prob '{}' {} = {}",
                tok,
                id,
                self.helper.logit_biases[id as usize]
            );
        } else {
            wprintln!("prob '{}' -> no token", tok)
        }
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
        wprintln!(
            "xapp {} '{}' st={}",
            token,
            String::from_utf8_lossy(bytes),
            self.curr_step
        );

        // save the token, just in case
        let toks = &mut self.helper.tokens;
        toks.push(token);

        let rec = &mut *self.rec;
        for b in bytes {
            rec.push_byte(*b)
        }
        rec.collapse();

        if self.tok_limit == 1 {
            self.next_step("token limit reached");
        } else if (0..=255).all(|byte| !rec.byte_allowed(byte)) {
            if rec.special_allowed(SpecialToken::EndOfSentence) {
                self.next_step("no bytes, but only EOS allowed");
            } else {
                self.stop("no bytes, no EOS");
            }
        } else {
            if self.tok_limit > 0 {
                self.tok_limit -= 1;
            }
        }

        self.compute();
        // self.log_prob(" a");
        // self.log_prob(" about");
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
                text: "I am about ".to_string(),
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
