pub mod printing;
pub mod rx;
pub mod rxvm;
pub mod toktree;
pub mod recognizer;

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
pub struct GuidanceVmHelper {
    pub tokens: Vec<u32>,
    pub prompt_length: usize,
    pub logit_biases: Vec<f32>,
}

// gvm_* are exposed to C in both GuidanceVm and GuidanceVmHelper
impl GuidanceVmHelper {
    pub fn new() -> Self {
        GuidanceVmHelper {
            tokens: Vec::new(),
            prompt_length: 0,
            logit_biases: Vec::new(),
        }
    }
    pub fn gvm_get_logit_bias_buffer(&mut self, size: u32) -> *mut f32 {
        self.logit_biases.resize(size as usize, 0.0);
        self.logit_biases.as_mut_ptr()
    }
    pub fn gvm_get_prompt_buffer(&mut self, size: u32) -> *mut u32 {
        self.prompt_length = size as usize;
        self.tokens.resize(self.prompt_length, 0);
        self.tokens.as_mut_ptr()
    }
}

pub trait GuidanceVm {
    /// Create a new instance of VM, based on existing instance, for example when doing beam-search.
    fn gvm_clone(&mut self) -> Self;
    /// The prompt is in self.helper.tokens.
    /// On return, self.helper.logit_biases are supposed to be updated.
    fn gvm_process_prompt(&mut self);
    /// On return, self.helper.logit_biases are supposed to be updated.
    fn gvm_append_token(&mut self, token: u32);
}

#[macro_export]
macro_rules! gvm_expose_all {
    ($struct_name:ident, $new:expr) => {
        $crate::expose!($struct_name::gvm_process_prompt() -> ());
        $crate::expose!($struct_name::gvm_append_token(token: u32) -> ());
        $crate::expose!($struct_name::helper::gvm_get_logit_bias_buffer(size: u32) -> *mut f32);
        $crate::expose!($struct_name::helper::gvm_get_prompt_buffer(size: u32) -> *mut u32);

        #[no_mangle]
        pub extern "C" fn gvm_create() -> *mut $struct_name {
            let b = Box::new($new);
            Box::into_raw(b)
        }

        #[no_mangle]
        pub extern "C" fn gvm_clone(self_: *mut $struct_name) -> *mut $struct_name {
            let b = unsafe { (&mut *self_).gvm_clone() };
            Box::into_raw(Box::new(b))
        }

        #[no_mangle]
        pub extern "C" fn gvm_free(self_: *mut $struct_name) {
            let _drop = unsafe { Box::from_raw(self_) };
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
macro_rules! println {
    () => {
        $crate::printing::_print("\n")
    };
    ($($arg:tt)*) => {{
        $crate::printing::_print(&format!($($arg)*));
        $crate::printing::_print("\n");
    }};
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {{
        $crate::printing::_print(&format!($($arg)*));
    }};
}
