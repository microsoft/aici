use std::rc::Rc;

use svob::SimpleVob;
use toktree::{SpecialToken, TokTrie};

pub mod bytes;
pub mod host;
pub mod recognizer;
pub mod rng;
pub mod svob;
pub mod toktree;

pub type TokenId = bytes::TokenId;

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

#[derive(Clone)]
pub struct AiciVmHelper {
    pub logit_biases: Vec<f32>,
    pub allowed_tokens: SimpleVob,
    pub trie: Rc<Box<TokTrie>>,
}

// aici_* are exposed to C in both AiciVm and AiciVmHelper
impl AiciVmHelper {
    pub fn new() -> Self {
        AiciVmHelper {
            logit_biases: Vec::new(),
            allowed_tokens: SimpleVob::new(),
            trie: Rc::new(Box::new(TokTrie::from_host())),
        }
    }
    pub fn aici_get_logit_bias_buffer(&mut self, size: u32) -> *mut f32 {
        // we keep one more logit at the end as a placeholder to avoid branching in
        // the inner loop of append_bias
        self.logit_biases.resize((size + 1) as usize, 0.0);
        self.allowed_tokens.resize(self.logit_biases.len());
        self.logit_biases.as_mut_ptr()
    }

    pub fn all_disallowed(&mut self) {
        self.allowed_tokens.set_all(false);
    }

    pub fn allow_one(&mut self, tok: TokenId) {
        self.allowed_tokens.allow_token(tok);
    }

    pub fn allow_eos(&mut self) {
        self.allow_one(self.trie.special_token(SpecialToken::EndOfSentence));
    }

    pub fn compute_biases(&mut self) {
        self.logit_biases.iter_mut().for_each(|x| *x = -100.0);
        self.allowed_tokens.apply_to(&mut self.logit_biases);
    }
}

pub trait AiciVm {
    /// The prompt, single generated token, or all ff tokens, arg in host::tokens_arg().
    /// On return, self.helper.logit_biases are supposed to be updated.
    fn aici_process(&mut self);
    // Used in testing.
    fn get_helper(&mut self) -> &mut AiciVmHelper;
}

#[macro_export]
macro_rules! aici_expose_all {
    ($struct_name:ident, $new:expr) => {
        $crate::expose!($struct_name::aici_process() -> ());
        $crate::expose!($struct_name::helper::aici_get_logit_bias_buffer(size: u32) -> *mut f32);

        #[no_mangle]
        pub extern "C" fn aici_create() -> *mut $struct_name {
            let b = Box::new($new);
            Box::into_raw(b)
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

#[macro_export]
macro_rules! wprintln {
    () => {
        $crate::host::_print("\n")
    };
    ($($arg:tt)*) => {{
        $crate::host::_print(&format!($($arg)*));
        $crate::host::_print("\n");
    }};
}

#[macro_export]
macro_rules! wprint {
    ($($arg:tt)*) => {{
        $crate::host::_print(&format!($($arg)*));
    }};
}

