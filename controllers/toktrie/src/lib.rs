use serde::{Deserialize, Serialize};

pub mod bytes;
pub mod recognizer;
pub mod rng;
mod svob;
mod toktree;

pub use svob::{SimpleVob, SimpleVobIter};
pub use toktree::{Recognizer, SpecialToken, TokRxInfo, TokTrie, TokenId};

#[derive(Serialize, Deserialize, Debug)]
pub struct StepArg {
    /// Sampling result for the previous iteration.
    /// For simple sampled token 't', backtrack==0 and tokens==[t].
    /// For first request, backtrack==0 and tokens==[] (prompt is passed separately, before).
    /// Can be more complex when splices are used.
    pub backtrack: u32,
    pub tokens: Vec<TokenId>,
}

impl StepArg {
    pub fn save_tokens(&self, acc_tokens: &mut Vec<TokenId>) {
        let bt = self.backtrack as usize;
        assert!(
            bt <= acc_tokens.len(),
            "attempting to backtrack past beginning"
        );
        acc_tokens.truncate(acc_tokens.len() - bt);
        acc_tokens.extend_from_slice(&self.tokens);
    }
}

/*
For example, if we're generating JSON, according to the following schema:
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"}
  }
}

Let's say we have generated: {"name": "something
We would use a single splice:
    when_sampled: ['"', '",', '", '],
    backtrack: 1,
    ff_tokens: tokenize('", "age": ')
Which means: when any token starting with '"' is sampled, we remove it (backtrack: 1)
and then append the next full fragment of JSON '", "age": '

If the tokenizers has tokens like 'a"', 'b"' etc, then we would need many splices
(there may be limits how many we want to pass over the IPC boundary).
*/

/// Describes what to do after sampling.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Splice {
    /// If one of the tokens in when_sampled is sampled, this sequence is appended.
    /// When empty, this sequence is appended unconditionally, regardless of sampling.
    pub when_sampled: Vec<TokenId>,
    /// Backtrack this much before appending this sequence (this includes sampled token if any).
    pub backtrack: u32,
    /// Append these tokens after backtracking.
    pub ff_tokens: Vec<TokenId>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Branch<S> {
    /// If None, no sampling is performed.
    /// If Some(set), only tokens from the set are allowed.
    pub sample_mask: Option<S>,
    /// Override temperature for sampling. It may or may not be sticky.
    pub temperature: Option<f32>,
    /// Describes what to do after sampling.
    /// If no sampling, there should be exactly one splice, with empty `when_sampled`.
    pub splices: Vec<Splice>,
}

impl<S: Clone> Clone for Branch<S> {
    fn clone(&self) -> Self {
        Branch {
            sample_mask: self.sample_mask.clone(),
            temperature: self.temperature,
            splices: self.splices.clone(),
        }
    }
}

impl<S> Branch<S> {
    pub fn map_mask<F, T>(&self, f: F) -> Branch<T>
    where
        F: FnOnce(&S) -> T,
    {
        Branch {
            sample_mask: self.sample_mask.as_ref().map(f),
            temperature: self.temperature,
            splices: self.splices.clone(),
        }
    }

    pub fn stop() -> Self {
        Branch {
            sample_mask: None,
            temperature: None,
            splices: vec![],
        }
    }

    pub fn is_stop(&self) -> bool {
        self.sample_mask.is_none() && self.splices.is_empty()
    }

    pub fn splice(backtrack: u32, ff_tokens: Vec<TokenId>) -> Self {
        Branch {
            sample_mask: None,
            temperature: None,
            splices: vec![Splice {
                when_sampled: vec![],
                backtrack,
                ff_tokens,
            }],
        }
    }

    pub fn noop() -> Self {
        Self::splice(0, vec![])
    }

    pub fn sample(set: S, temperature: Option<f32>) -> Self {
        Branch {
            sample_mask: Some(set),
            temperature,
            splices: vec![],
        }
    }
}
