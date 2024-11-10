pub use toktrie;
pub use toktrie::{bytes, recognizer, rng};
pub use toktrie::{SimpleVob, TokenizerEnv};

use std::sync::{Arc, Mutex};

pub mod bindings;

pub use bindings::{controller::*, runtime, runtime_storage, tokenizer};

#[macro_export]
macro_rules! export {
    ($ty:ident) => {
        #[doc(hidden)]
        #[cfg(target_os = "wasi")]
        $crate::bindings::export!($ty with_types_in $crate::bindings);
    };
}

mod host;

#[cfg(feature = "cfg")]
pub mod cfg;
#[cfg(feature = "cfg")]
mod lex;

#[cfg(feature = "rx")]
pub mod rx;

pub mod dlex;

pub mod substring;

pub use host::{aici_stop, StorageCmd, StorageOp, StorageResp};

pub trait AiciCtrl {
    fn init_prompt(&mut self, arg: InitPromptArg) -> InitPromptResult {
        InitPromptResult::from_arg(arg)
    }
    fn mid_process(&mut self, arg: MidProcessArg) -> MidProcessResult;
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

    fn init_prompt(&self, arg: InitPromptArg) -> InitPromptResult {
        self.mutable_controller.lock().unwrap().init_prompt(arg)
    }

    fn mid_process(&self, arg: MidProcessArg) -> MidProcessResult {
        self.mutable_controller.lock().unwrap().mid_process(arg)
    }
}
