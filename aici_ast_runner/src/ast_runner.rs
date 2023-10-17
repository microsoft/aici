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

use std::fmt::Debug;

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
#[derive(Serialize, Deserialize, Clone, Debug)]
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

enum StepSpecific {
    Fixed { tokens: Vec<TokenId> },
    Gen { rx: RxStackRecognizer },
    Stop,
}
struct StepState {
    ast: Step,
    specific: StepSpecific,
    token_idx: usize,
    max_tokens: usize,
}
pub struct Runner {
    helper: AiciVmHelper,
    state_idx: usize,
    states: Vec<StepState>,
}

impl Debug for StepState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/", self.token_idx)?;
        if self.max_tokens > 10000 {
            write!(f, "inf")?;
        } else {
            write!(f, "{}", self.max_tokens)?;
        }
        Ok(())
    }
}

impl StepState {
    #[allow(dead_code)]
    fn pp(&self) -> String {
        format!("{:?}", self.ast)
    }

    fn from_specific(ast: &Step, specific: StepSpecific, max_tokens: usize) -> StepState {
        StepState {
            ast: ast.clone(),
            specific,
            token_idx: 0,
            max_tokens,
        }
    }

    fn from_ast(s: &Step) -> StepState {
        match s {
            Step::Fixed { text } => Self::from_specific(
                s,
                StepSpecific::Fixed {
                    tokens: tokenize(&text),
                },
                usize::MAX,
            ),

            Step::Gen { rx, max_tokens } => {
                let rx = match rx {
                    Some(rx) => &rx,
                    None => ".*",
                };
                Self::from_specific(
                    s,
                    StepSpecific::Gen {
                        rx: RecRx::from_rx(&rx).to_stack_recognizer(),
                    },
                    *max_tokens,
                )
            }
        }
    }

    fn check_eos(&self, optional: bool) -> bool {
        self.token_idx >= self.max_tokens
            || match &self.specific {
                StepSpecific::Stop => true,
                StepSpecific::Fixed { tokens } => self.token_idx >= tokens.len(),
                StepSpecific::Gen { rx } => {
                    rx.special_allowed(SpecialToken::EndOfSentence)
                        && (optional || (0..=255).all(|byte| !rx.byte_allowed(byte)))
                }
            }
    }

    fn allows_eos(&self) -> bool {
        self.check_eos(true)
    }

    fn _forces_eos(&self) -> bool {
        self.check_eos(false)
    }

    fn advance(&mut self, helper: &AiciVmHelper, token: TokenId) {
        self.token_idx += 1;
        match &mut self.specific {
            StepSpecific::Stop => {}
            StepSpecific::Fixed { .. } => {}
            StepSpecific::Gen { rx } => helper.trie.append_token(rx, token),
        }
    }

    // the mut on self is bogus - the state of the 'rx' doesn't change
    fn allows_token(&mut self, helper: &AiciVmHelper, token: TokenId) -> bool {
        if token == helper.trie.special_token(SpecialToken::EndOfSentence) {
            return self.allows_eos();
        }
        match &mut self.specific {
            StepSpecific::Stop => false,
            StepSpecific::Fixed { tokens } => {
                self.token_idx < tokens.len() && tokens[self.token_idx] == token
            }
            StepSpecific::Gen { rx } => helper.trie.token_allowed(rx, token),
        }
    }

    fn apply_to(&mut self, helper: &mut AiciVmHelper) {
        match &mut self.specific {
            StepSpecific::Stop => {
                helper.allow_eos();
            }
            StepSpecific::Fixed { tokens } => {
                if self.token_idx < tokens.len() {
                    helper.allow_one(tokens[self.token_idx]);
                }
            }
            StepSpecific::Gen { rx } => {
                helper.trie.add_bias(rx, &mut helper.logit_biases);
            }
        }
    }
}

impl Runner {
    pub fn new(program: Program) -> Self {
        let mut states = program
            .steps
            .iter()
            .map(StepState::from_ast)
            .collect::<Vec<_>>();
        let stop_ast = Step::Fixed {
            text: "[STOP]".to_string(),
        };
        states.push(StepState::from_specific(
            &stop_ast,
            StepSpecific::Stop,
            usize::MAX,
        ));

        for (idx, state) in states.iter().enumerate() {
            wprintln!("[{}] {} {:?}", idx, state.pp(), state);
        }

        Self {
            helper: AiciVmHelper::new(),
            state_idx: 0,
            states,
        }
    }

    fn stop(&mut self, info: &str) {
        self.state_idx = self.states.len() - 1;
        wprintln!("stop: {}", info)
    }

    fn advance(&mut self, token: TokenId) {
        let bytes = self.helper.trie.token(token);
        wprintln!(
            "append {} '{}' [{}] {:?}",
            token,
            String::from_utf8_lossy(bytes),
            self.state_idx,
            self.states[self.state_idx]
        );

        // skip as many states as we can (that allow EOS), and find the last one that allows the token
        let mut last_idx = usize::MAX;
        for idx in self.state_idx..self.states.len() {
            if self.states[idx].allows_token(&self.helper, token) {
                last_idx = idx;
            }
            if !self.states[idx].allows_eos() {
                break;
            }
        }

        if last_idx == usize::MAX {
            self.stop("no state allows token");
            return;
        }

        if self.state_idx != last_idx {
            self.state_idx = last_idx;
        }

        self.states[last_idx].advance(&mut self.helper, token);
        wprintln!(" => [{}] {:?}", self.state_idx, self.states[self.state_idx]);
    }

    fn compute(&mut self) {
        self.helper.all_disallowed();
        for state in &mut self.states[self.state_idx..] {
            state.apply_to(&mut self.helper);
            if !state.allows_eos() {
                break;
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
