use wasmtime::component::bindgen;

bindgen!({
    world: "aici",
    path: "../../wit",
    additional_derives: [
        serde::Deserialize,
        serde::Serialize,
    ],
    trappable_imports: true, // host implementations return wasmtime::Result<T>
});

impl Branch {
    pub fn map_mask<F, T>(&self, f: F) -> toktrie::Branch<T>
    where
        F: FnOnce(&Vocabulary) -> T,
    {
        toktrie::Branch {
            sample_mask: self.sample_mask.as_ref().map(f),
            temperature: self.temperature,
            splices: self
                .splices
                .iter()
                .map(|s| toktrie::Splice {
                    backtrack: s.backtrack,
                    when_sampled: s.when_sampled.clone(),
                    ff_tokens: s.ff_tokens.clone(),
                })
                .collect(),
        }
    }

    pub fn has_backtrack(&self) -> bool {
        let max_bt = if self.sample_mask.is_none() { 0 } else { 1 };
        self.splices.iter().any(|s| s.backtrack > max_bt)
    }

    pub fn has_ff_tokens(&self) -> bool {
        self.splices.len() > 0
    }
}

pub use self::{
    aici::abi::{runtime::SeqId, tokenizer::TokenId},
    exports::aici::abi::controller::*,
};
