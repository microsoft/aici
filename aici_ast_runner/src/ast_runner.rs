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

- Gen("\d+"); Fixed(" years") should be equivalent to Gen("\d+ years"), that is the model decides when to stop
    generating digits and start generating " years" (modulo token problems above)
*/
mod rx;

use rx::RxStackRecognizer;
use serde::{Deserialize, Serialize};

use crate::rx::RecRx;

use aici_abi::{
    aici_expose_all,
    host::{self, tokenize},
    toktree::{Recognizer, SpecialToken},
    wprintln, AiciVm, AiciVmHelper, TokenId,
};

// The JSON AST
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

enum StepState {
    Fixed { tokens: Vec<TokenId> },
    Gen { rx: RxStackRecognizer },
    Stop,
}
pub struct Runner {
    helper: AiciVmHelper,
    program: Program,
    state: StepState,
    step_idx: usize,
    token_idx_in_step: usize,
    max_tokens_in_step: usize,
}

impl Runner {
    pub fn new(program: Program) -> Self {
        Self {
            helper: AiciVmHelper::new(),
            program,
            step_idx: 0,
            state: StepState::Stop,
            token_idx_in_step: 0,
            max_tokens_in_step: usize::MAX,
        }
    }

    fn set_state(&mut self, state: StepState) {
        self.state = state;
        self.token_idx_in_step = 0;
        self.max_tokens_in_step = usize::MAX;
    }

    fn stop(&mut self, info: &str) {
        self.set_state(StepState::Stop);
        wprintln!("stop: {}", info)
    }

    fn next_step(&mut self, info: &str) {
        self.step_idx += 1;
        if self.step_idx < self.program.steps.len() {
            wprintln!("step -> {}; {}", self.step_idx, info);
            self.new_step();
        } else {
            self.step_idx = self.program.steps.len();
            self.stop(info);
        }
    }

    fn step_state(&mut self) -> StepState {
        match &self.program.steps[self.step_idx] {
            Step::Fixed { text } => StepState::Fixed {
                tokens: tokenize(&text),
            },
            Step::Gen { rx, .. } => {
                let rx = match rx {
                    Some(rx) => rx,
                    None => ".*",
                };
                StepState::Gen {
                    rx: RecRx::from_rx(&rx).to_stack_recognizer(),
                }
            }
        }
    }

    fn new_step(&mut self) {
        let state = self.step_state();
        self.set_state(state);
        match &self.program.steps[self.step_idx] {
            Step::Fixed { .. } => {}
            Step::Gen { max_tokens, .. } => {
                self.max_tokens_in_step = *max_tokens;
            }
        }
    }

    fn check_step_finished_inner(&mut self) {
        if self.token_idx_in_step >= self.max_tokens_in_step {
            self.next_step("max tokens reached");
            return;
        }

        match &mut self.state {
            StepState::Stop => {}
            StepState::Fixed { tokens } => {
                if self.token_idx_in_step >= tokens.len() {
                    self.next_step("fixed finished")
                }
            }
            StepState::Gen { rx } => {
                if (0..=255).all(|byte| !rx.byte_allowed(byte)) {
                    if !rx.special_allowed(SpecialToken::EndOfSentence) {
                        self.stop("no bytes, no EOS");
                    } else {
                        self.next_step("rx finished")
                    }
                }
            }
        }
    }

    fn check_step_finished(&mut self) {
        loop {
            let s0 = self.step_idx;
            self.check_step_finished_inner();
            if s0 == self.step_idx {
                break;
            }
        }
    }

    fn advance(&mut self, token: TokenId) {
        let bytes = self.helper.trie.token(token);
        wprintln!(
            "append {} '{}' st: {}.{}/{}",
            token,
            String::from_utf8_lossy(bytes),
            self.step_idx,
            self.token_idx_in_step,
            std::cmp::min(self.max_tokens_in_step, 10000)
        );

        self.token_idx_in_step += 1;
        match &mut self.state {
            StepState::Stop => {}
            StepState::Fixed { .. } => {}
            StepState::Gen { rx } => {
                for b in bytes {
                    rx.push_byte(*b)
                }
                rx.collapse();
            }
        }

        self.check_step_finished();
    }

    fn compute(&mut self) {
        match &mut self.state {
            StepState::Stop => {
                return self.helper.allow_eos();
            }
            StepState::Fixed { tokens } => {
                let t = tokens[self.token_idx_in_step];
                return self.helper.allow_one(t);
            }
            StepState::Gen { rx } => {
                self.helper.all_disallowed();
                self.helper.trie.add_bias(rx, &mut self.helper.logit_biases);
                return;
            }
        }
    }

    #[allow(dead_code)]
    fn print_prob(&self, tok: &str) {
        if let Some(id) = self.helper.trie.token_id(tok.as_bytes()) {
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
    fn aici_process_prompt(&mut self) {
        wprintln!("prompt, {} tokens", self.helper.prompt_length);
        // ignore the prompt (for now)
        self.new_step();
        self.check_step_finished();
        self.compute();
    }

    fn aici_append_token(&mut self, token: u32) {
        // save the token, just in case
        let toks = &mut self.helper.tokens;
        toks.push(token);

        self.advance(token);
        self.compute();
        // self.print_prob(" a");
        // self.print_prob(" about");
    }

    fn get_helper(&mut self) -> &mut AiciVmHelper {
        &mut self.helper
    }
}

fn main() {
    //    let _run = sample_prog();
}

fn sample_prog() -> Runner {
    let a = host::arg_bytes();
    let p: Program = serde_json::from_slice(&a).unwrap();
    Runner::new(p)
    // Runner::new(Program {
    //     steps: vec![
    //         Step::Fixed {
    //             text: "I am about ".to_string(),
    //         },
    //         Step::Gen {
    //             max_tokens: 5,
    //             rx: Some(r#"\d\d"#.to_string()),
    //         },
    //         Step::Fixed {
    //             text: " years and ".to_string(),
    //         },
    //         Step::Gen {
    //             max_tokens: 5,
    //             rx: Some(r#"\d+"#.to_string()),
    //         },
    //     ],
    // })
}

aici_expose_all!(Runner, sample_prog());
