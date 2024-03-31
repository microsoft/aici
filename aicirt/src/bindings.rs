use serde::{Deserialize, Deserializer, Serialize, Serializer};
use wasmtime::component::bindgen;

bindgen!("aici" in "../wit");

pub use self::{
    aici::abi::{runtime::SeqId, tokenizer::TokenId},
    exports::aici::abi::controller::*,
};

pub trait Json {
    fn to_json(&self) -> String;
}

// Workaround for WIT not supporting empty record types, define some here, and then we will pass ()
// to the guest.
#[derive(Debug, Serialize, Deserialize)]
pub struct InitPromptResult();

impl From<()> for InitPromptResult {
    fn from(_: ()) -> Self {
        InitPromptResult()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PreProcessArg();

impl From<()> for PreProcessArg {
    fn from(_: ()) -> Self {
        PreProcessArg()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Vocabulary")]
struct VocabularyDef {
    data: Vec<u32>,
}
#[derive(Serialize, Deserialize)]
#[serde(remote = "InitPromptArg")]
struct InitPromptArgDef {
    prompt: Vec<TokenId>,
}
// #[derive(Serialize, Deserialize)]
// #[serde(remote = "InitPromptResult")]
// struct InitPromptResultDef {}
// #[derive(Serialize, Deserialize)]
// #[serde(remote = "PreProcessArg")]
// struct PreProcessArgDef {}
#[derive(Serialize, Deserialize)]
#[serde(remote = "PreProcessResult")]
struct PreProcessResultDef {
    num_forks: u32,
    suspend: bool,
    ff_tokens: Vec<TokenId>,
}
#[derive(Serialize, Deserialize)]
#[serde(remote = "MidProcessArg")]
struct MidProcessArgDef {
    fork_group: Vec<SeqId>,
}
#[derive(Serialize, Deserialize)]
#[serde(remote = "MidProcessResult")]
pub enum MidProcessResultDef {
    Stop,
    SampleWithBias(SampleWithBias),
    Splice(Splice),
}
#[derive(Serialize, Deserialize)]
#[serde(remote = "SampleWithBias")]
struct SampleWithBiasDef {
    allowed_tokens: Vocabulary,
}
#[derive(Serialize, Deserialize)]
#[serde(remote = "Splice")]
struct SpliceDef {
    backtrack: u32,
    ff_tokens: Vec<TokenId>,
}
#[derive(Serialize, Deserialize)]
#[serde(remote = "PostProcessArg")]
struct PostProcessArgDef {
    tokens: Vec<TokenId>,
    backtrack: u32,
}
#[derive(Serialize, Deserialize)]
#[serde(remote = "PostProcessResult")]
struct PostProcessResultDef {
    stop: bool,
}

macro_rules! serde_wrapper {
    ($name:ident, $wrapper:ident) => {
        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                $wrapper::serialize(self, serializer)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<$name, D::Error>
            where
                D: Deserializer<'de>,
            {
                $wrapper::deserialize(deserializer)
            }
        }
    };
}

serde_wrapper!(Vocabulary, VocabularyDef);
serde_wrapper!(InitPromptArg, InitPromptArgDef);
// serde_wrapper!(InitPromptResult, InitPromptResultDef);
// serde_wrapper!(PreProcessArg, PreProcessArgDef);
serde_wrapper!(PreProcessResult, PreProcessResultDef);
serde_wrapper!(MidProcessArg, MidProcessArgDef);
serde_wrapper!(MidProcessResult, MidProcessResultDef);
serde_wrapper!(SampleWithBias, SampleWithBiasDef);
serde_wrapper!(Splice, SpliceDef);
serde_wrapper!(PostProcessArg, PostProcessArgDef);
serde_wrapper!(PostProcessResult, PostProcessResultDef);
