use serde::{Deserialize, Serialize};
use svob::SimpleVob;

pub mod bytes;
mod host;
pub mod recognizer;
pub mod rng;
pub mod svob;
pub mod toktree;

#[cfg(feature = "cfg")]
pub mod cfg;
#[cfg(feature = "cfg")]
mod lex;

#[cfg(feature = "rx")]
pub mod rx;

pub mod substring;

pub type TokenId = bytes::TokenId;

pub use host::{
    aici_stop, arg_bytes, arg_string, self_seq_id, tokenize, tokenize_bytes, StorageCmd, StorageOp,
    StorageResp, VariableStorage,
};

#[derive(Serialize, Deserialize, Debug)]
pub struct InitPromptArg {
    pub prompt: Vec<TokenId>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct InitPromptResult {}

#[repr(transparent)]
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct SeqId(pub u32);

#[derive(Serialize, Deserialize, Debug)]
pub struct MidProcessArg {
    /// Sampling result for the previous iteration.
    /// For simple sampled token 't', backtrack==0 and tokens==[t].
    /// For first request, backtrack==0 and tokens==[] (prompt is passed separetely, before).
    /// Can be more complex when splices are used.
    pub backtrack: u32,
    pub tokens: Vec<TokenId>,
    ///
    pub fork_group: Vec<SeqId>,
}

impl MidProcessArg {
    pub fn has_eos(&self) -> bool {
        let eos = host::eos_token();
        self.tokens.iter().any(|t| *t == eos)
    }

    pub fn save_tokens(&self, acc_tokens: &mut Vec<TokenId>) {
        acc_tokens.truncate(acc_tokens.len() - self.backtrack as usize);
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
(there may be limits how many we want to pass over the IPC boundry).
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
    /// Describes what to do after sampling.
    /// If no sampling, there should be exactly one splice, with empty `when_sampled`.
    pub splices: Vec<Splice>,
}

impl<S: Clone> Clone for Branch<S> {
    fn clone(&self) -> Self {
        Branch {
            sample_mask: self.sample_mask.clone(),
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
            splices: self.splices.clone(),
        }
    }
}

#[derive(Debug)]
pub struct MidProcessResult {
    /// Fork the request into multiple branches.
    /// Typically, exactly one branch is returned.
    /// If multiple branches are returned, they are executed in parallel.
    /// If no branches are returned, the request is terminated.
    pub branches: Vec<Branch<SimpleVob>>,
}

impl MidProcessResult {
    pub fn stop() -> Self {
        MidProcessResult { branches: vec![] }
    }

    pub fn sample(set: SimpleVob) -> Self {
        MidProcessResult {
            branches: vec![Branch {
                sample_mask: Some(set),
                splices: vec![],
            }],
        }
    }

    pub fn splice(backtrack: u32, ff_tokens: Vec<TokenId>) -> Self {
        MidProcessResult {
            branches: vec![Branch {
                sample_mask: None,
                splices: vec![Splice {
                    when_sampled: vec![],
                    backtrack,
                    ff_tokens,
                }],
            }],
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct ProcessResultOffset {
    pub branches: Vec<Branch<usize>>,
}

pub trait AiciCtrl {
    /// Called with the initial prompt. ~1000ms time limit.
    /// By default ignore prompt.
    /// This is typically just the start token if any (REST API forces empty prompt).
    fn init_prompt(&mut self, _arg: InitPromptArg) -> InitPromptResult {
        InitPromptResult::default()
    }

    /// This is the main entry point for the module. ~20ms time limit.
    fn mid_process(&mut self, arg: MidProcessArg) -> MidProcessResult;

    // Internals
    fn aici_init_prompt(&mut self) {
        let arg: InitPromptArg = serde_json::from_slice(&host::process_arg_bytes()).unwrap();
        let res = self.init_prompt(arg);
        let res_bytes = serde_json::to_vec(&res).unwrap();
        host::return_process_result(&res_bytes);
    }

    fn aici_mid_process(&mut self) {
        let arg: MidProcessArg = serde_json::from_slice(&host::process_arg_bytes())
            .expect("aici_mid_process: failed to deserialize MidProcessArg");
        let res = self.mid_process(arg);
        if res.branches.len() > 1 {
            panic!("aici_mid_process: multiple branches not yet supported");
        }
        let res = ProcessResultOffset {
            branches: res
                .branches
                .into_iter()
                .map(|b| {
                    b.map_mask(|vob| {
                        host::return_logit_bias(&vob);
                        0
                    })
                })
                .collect(),
        };
        let res_bytes = serde_json::to_vec(&res).expect("aici_mid_process: failed to serialize");
        host::return_process_result(&res_bytes);
    }
}

/// Expose method as extern "C", usage:
///     expose!(Foo::set_count(n: i32) -> i32);
/// Generates "C" function:
///     set_count(Foo *, i32) -> i32
#[macro_export]
macro_rules! expose {
    ($struct_name:ident :: $method_name:ident ( $($arg:ident : $typ:ty),* ) -> $ret:ty) => {
        #[no_mangle]
        pub extern "C" fn $method_name(self_: *mut $struct_name, $($arg : $typ),*) -> $ret {
            unsafe {
                (&mut *self_).$method_name($($arg),*)
            }
        }
    };
    ($struct_name:ident :: $field:ident :: $method_name:ident ( $($arg:ident : $typ:ty),* ) -> $ret:ty) => {
        #[no_mangle]
        pub extern "C" fn $method_name(self_: *mut $struct_name, $($arg : $typ),*) -> $ret {
            unsafe {
                (&mut *self_).$field.$method_name($($arg),*)
            }
        }
    };
}

#[macro_export]
macro_rules! aici_expose_all {
    ($struct_name:ident, $new:expr) => {
        $crate::expose!($struct_name::aici_mid_process() -> ());
        $crate::expose!($struct_name::aici_init_prompt() -> ());

        #[no_mangle]
        pub extern "C" fn aici_create() -> *mut $struct_name {
            let b = Box::new($new);
            Box::into_raw(b)
        }

        #[no_mangle]
        pub extern "C" fn aici_panic() {
            panic!("aici_panic()")
        }
    }
}

#[macro_export]
macro_rules! include_bytes_aligned {
    ($align_ty:ty, $path:literal) => {{
        #[repr(C)] // guarantee 'bytes' comes after '_align'
        pub struct AlignedAs<Align, Bytes: ?Sized> {
            pub _align: [Align; 0],
            pub bytes: Bytes,
        }

        // this assignment is made possible by CoerceUnsized
        static ALIGNED: &AlignedAs<$align_ty, [u8]> = &AlignedAs {
            _align: [],
            bytes: *include_bytes!($path),
        };

        &ALIGNED.bytes
    }};
}
