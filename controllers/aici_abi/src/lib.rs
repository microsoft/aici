use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[doc(hidden)]
pub mod bindings {
    wit_bindgen::generate!({
        world: "aici",
        path: "../../wit",
        additional_derives: [serde::Serialize, serde::Deserialize],
        pub_export_macro: true
    });

    pub use self::{aici::abi::*, exports::aici::abi::*};
}

pub use bindings::{controller::*, runtime, runtime_storage, tokenizer};
pub mod svob;
pub use svob::SimpleVob;

#[macro_export]
macro_rules! export {
    ($ty:ident) => {
        #[doc(hidden)]
        #[cfg(target_arch = "wasm32")]
        $crate::bindings::export!($ty with_types_in $crate::bindings);
    };
}

pub mod bytes;
mod host;
pub mod recognizer;
pub mod rng;
pub mod toktree;

#[cfg(feature = "cfg")]
pub mod cfg;
#[cfg(feature = "cfg")]
mod lex;

#[cfg(feature = "rx")]
pub mod rx;

pub mod substring;

pub use host::{aici_stop, StorageCmd, StorageOp, StorageResp};

// Workaround for WIT not supporting empty record types, define some here, and then we will pass ()
// to the guest.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct InitPromptResult();

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PreProcessArg();

impl PreProcessResult {
    pub fn new(num_forks: u32) -> Self {
        PreProcessResult {
            num_forks,
            suspend: false,
            ff_tokens: vec![],
        }
    }
    pub fn continue_() -> Self {
        PreProcessResult::new(1)
    }
    pub fn suspend() -> Self {
        PreProcessResult {
            num_forks: 1,
            suspend: true,
            ff_tokens: vec![],
        }
    }
    pub fn stop() -> Self {
        PreProcessResult::new(0)
    }
    pub fn ff_tokens(toks: Vec<TokenId>) -> Self {
        PreProcessResult {
            num_forks: 1,
            suspend: false,
            ff_tokens: toks,
        }
    }
}

impl PostProcessResult {
    pub fn stop() -> Self {
        PostProcessResult { stop: true }
    }

    pub fn continue_() -> Self {
        PostProcessResult { stop: false }
    }

    pub fn from_arg(arg: &PostProcessArg) -> Self {
        let stop = arg.tokens.contains(&tokenizer::eos_token());
        PostProcessResult { stop }
    }
}

pub trait AiciCtrl {
    fn init_prompt(&mut self, _arg: InitPromptArg) -> InitPromptResult {
        InitPromptResult()
    }
    fn pre_process(&mut self, PreProcessArg(): PreProcessArg) -> PreProcessResult {
        PreProcessResult::continue_()
    }
    fn mid_process(&mut self, arg: MidProcessArg) -> MidProcessResult;
    fn post_process(&mut self, arg: PostProcessArg) -> PostProcessResult {
        PostProcessResult::from_arg(&arg)
    }
}

pub trait Program: AiciCtrl {
    fn new(_: String) -> Self;
}

pub struct ExportedProgram<C: AiciCtrl> {
    mutable_controller: Arc<Mutex<C>>,
}

impl<C: AiciCtrl> ExportedProgram<C> {
    pub fn new(controller: C) -> Self {
        ExportedProgram {
            mutable_controller: Arc::new(Mutex::new(controller)),
        }
    }
}

impl<C: Program + 'static> GuestRunner for ExportedProgram<C> {
    fn new(arg: String) -> Self {
        ExportedProgram::new(C::new(arg))
    }

    fn init_prompt(&self, arg: InitPromptArg) {
        self.mutable_controller.lock().unwrap().init_prompt(arg);
    }

    fn pre_process(&self) -> PreProcessResult {
        self.mutable_controller
            .lock()
            .unwrap()
            .pre_process(PreProcessArg())
    }

    fn mid_process(&self, arg: MidProcessArg) -> MidProcessResult {
        self.mutable_controller.lock().unwrap().mid_process(arg)
    }

    fn post_process(&self, arg: PostProcessArg) -> PostProcessResult {
        self.mutable_controller.lock().unwrap().post_process(arg)
    }
}
