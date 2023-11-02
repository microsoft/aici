use std::io;

use crate::{
    bytes::{vec_from_bytes, TokenId},
    wprintln,
};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlobId(u32);

#[allow(dead_code)]
extern "C" {
    // Log a string.
    fn aici_host_print(ptr: *const u8, len: u32);

    // Read binary blob.
    // Always returns the size of the blob, will write up to `size` bytes to `dst`.
    fn aici_host_read_blob(blob: BlobId, dst: *mut u8, size: u32) -> u32;

    // Return the ID of TokTrie binary representation.
    fn aici_host_token_trie() -> BlobId;

    // Return the ID of argument passed by the user.
    fn aici_host_module_arg() -> BlobId;

    // Return the ID of argument passed by the user.
    fn aici_host_tokens() -> BlobId;

    // Tokenize given UTF8 string. The result is only valid until next call to this function.
    fn aici_host_tokenize(src: *const u8, src_size: u32) -> BlobId;

    // Append fast-forward (FF) token.
    // First FF token has to be returned by setting logit bias appropriately.
    // Next tokens are added using this interface.
    // All FF tokens are then generated in one go.
    fn aici_host_ff_token(token: u32);
}

// TODO: add <T>
fn read_blob(blob: BlobId, prefetch_size: usize) -> Vec<u8> {
    let mut buffer = vec![0u8; prefetch_size];
    let prefetch_size = prefetch_size as u32;
    let size = unsafe { aici_host_read_blob(blob, buffer.as_mut_ptr(), prefetch_size) };
    buffer.resize(size as usize, 0);
    if size > prefetch_size {
        // didn't read everything; retry
        unsafe { aici_host_read_blob(blob, buffer.as_mut_ptr(), size) };
    }
    buffer
}

#[cfg(not(target_arch = "wasm32"))]
pub type Printer = std::io::Stdout;

#[cfg(target_arch = "wasm32")]
pub struct Printer {}

#[cfg(target_arch = "wasm32")]
impl io::Write for Printer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        unsafe { aici_host_print(buf.as_ptr(), buf.len() as u32) };
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub fn init_panic() {
    #[cfg(target_arch = "wasm32")]
    std::panic::set_hook(Box::new(|info| {
        let file = info.location().unwrap().file();
        let line = info.location().unwrap().line();
        let col = info.location().unwrap().column();

        let msg = match info.payload().downcast_ref::<&'static str>() {
            Some(s) => *s,
            None => match info.payload().downcast_ref::<String>() {
                Some(s) => &s[..],
                None => "Box<Any>",
            },
        };

        let err_info = format!("Panicked at '{}', {}:{}:{}\n", msg, file, line, col);
        _print(&err_info);
    }))
}

pub fn stdout() -> Printer {
    #[cfg(target_arch = "wasm32")]
    {
        Printer {}
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        io::stdout()
    }
}

pub fn _print(msg: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        let vec: Vec<u8> = msg.into();
        unsafe { aici_host_print(vec.as_ptr(), vec.len() as u32) };
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::io::Write;
        std::io::stdout().write_all(msg.as_bytes()).unwrap();
    }
}

#[no_mangle]
pub extern "C" fn aici_init() {
    init_panic();
}

pub fn arg_bytes() -> Vec<u8> {
    #[cfg(target_arch = "wasm32")]
    return read_blob(unsafe { aici_host_module_arg() }, 1024);

    #[cfg(not(target_arch = "wasm32"))]
    return std::fs::read("arg.json").unwrap();
}

pub fn trie_bytes() -> Vec<u8> {
    #[cfg(target_arch = "wasm32")]
    return read_blob(unsafe { aici_host_token_trie() }, 0);

    #[cfg(not(target_arch = "wasm32"))]
    return std::fs::read("tokenizer.bin").unwrap();
}

pub fn tokens_arg() -> Vec<TokenId> {
    let r = read_blob(unsafe { aici_host_tokens() }, 256);
    vec_from_bytes(&r)
}

pub fn tokenize(s: &str) -> Vec<TokenId> {
    let id = unsafe { aici_host_tokenize(s.as_ptr(), s.len() as u32) };
    let r = read_blob(id, 4 * (s.len() / 3 + 10));
    let res = vec_from_bytes(&r);
    wprintln!("tokenize: {:?} -> {:?}", s, res);
    res
}

pub fn ff_token(token: TokenId) {
    unsafe {
        aici_host_ff_token(token);
    }
}
