wit_bindgen::generate!({
    world: "aici",
    path: "../../wit",
    additional_derives: [serde::Serialize, serde::Deserialize],
    pub_export_macro: true,
});

pub use self::{aici::abi::*, exports::aici::abi::*};

use controller::*;

impl From<InitPromptArg> for InitPromptResult {
    fn from(value: InitPromptArg) -> Self {
        InitPromptResult {
            prompt: value.prompt,
        }

    }
}

impl InitPromptResult {
    pub fn from_arg(arg: InitPromptArg) -> Self {
        arg.into()
    }
}

impl From<toktrie::SimpleVob> for Vocabulary {
    fn from(value: toktrie::SimpleVob) -> Self {
        let size = value.len() as u64;
        Vocabulary {
            data: value.into(),
            size,
        }
    }
}

impl From<toktrie::Splice> for Splice {
    fn from(value: toktrie::Splice) -> Self {
        Splice {
            backtrack: value.backtrack,
            ff_tokens: value.ff_tokens,
            when_sampled: value.when_sampled,
        }
    }
}

impl From<toktrie::Branch<toktrie::SimpleVob>> for Branch {
    fn from(value: toktrie::Branch<toktrie::SimpleVob>) -> Self {
        Branch {
            sample_mask: value.sample_mask.map(|x| x.into()),
            splices: value.splices.into_iter().map(|x| x.into()).collect(),
            temperature: value.temperature,
        }
    }
}

impl MidProcessArg {
    pub fn has_eos(&self) -> bool {
        let eos = tokenizer::eos_token();
        self.tokens.iter().any(|t| *t == eos)
    }

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

impl MidProcessResult {
    pub fn from_branch(branch: toktrie::Branch<toktrie::SimpleVob>) -> Self {
        if branch.is_stop() {
            Self::stop()
        } else {
            MidProcessResult {
                branches: vec![branch.into()],
            }
        }
    }
    pub fn from_branches(branches: Vec<toktrie::Branch<toktrie::SimpleVob>>) -> Self {
        MidProcessResult {
            branches: branches.into_iter().map(|x| x.into()).collect(),
        }
    }

    pub fn stop() -> Self {
        MidProcessResult { branches: vec![] }
    }

    pub fn sample(set: toktrie::SimpleVob) -> Self {
        Self::sample_with_temp(set, None)
    }

    pub fn sample_with_temp(set: toktrie::SimpleVob, temperature: Option<f32>) -> Self {
        Self::from_branch(toktrie::Branch::sample(set, temperature))
    }

    pub fn splice(backtrack: u32, ff_tokens: Vec<TokenId>) -> Self {
        Self::from_branch(toktrie::Branch::splice(backtrack, ff_tokens))
    }

    pub fn noop() -> Self {
        Self::splice(0, vec![])
    }

    pub fn is_stop(&self) -> bool {
        self.branches.is_empty()
    }
}
