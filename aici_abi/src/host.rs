use serde::{Deserialize, Serialize};
use std::io;

use crate::{
    bytes::{vec_from_bytes, TokenId},
    svob::SimpleVob,
    wprintln, SeqId,
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

    // Return the ID of argument passed to the process() function.
    // It's a JSON serialization of ProcessArg.
    fn aici_host_process_arg() -> BlobId;

    // Tokenize given UTF8 string. The result is only valid until next call to this function.
    fn aici_host_tokenize(src: *const u8, src_size: u32) -> BlobId;

    // Set logit bias based on bitmask in src.
    fn aici_host_return_logit_bias(src: *const u32);

    // Append fast-forward (FF) token.
    // First FF token has to be returned by setting logit bias appropriately.
    // Next tokens are added using this interface.
    // All FF tokens are then generated in one go.
    fn aici_host_ff_token(token: u32);

    fn aici_host_self_seq_id() -> u32;

    fn aici_host_return_process_result(res: *const u8, res_size: u32);

    fn aici_host_storage_cmd(cmd: *const u8, cmd_size: u32) -> BlobId;
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

pub fn ff_token(token: TokenId) {
    unsafe {
        aici_host_ff_token(token);
    }
}

pub fn return_logit_bias(vob: &SimpleVob) {
    assert!(vob.len() > 0);
    unsafe {
        aici_host_return_logit_bias(vob.as_ptr());
    }
}

pub fn process_arg_bytes() -> Vec<u8> {
    return read_blob(unsafe { aici_host_process_arg() }, 1024);
}

pub fn return_process_result(res: &[u8]) {
    unsafe {
        aici_host_return_process_result(res.as_ptr(), res.len() as u32);
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum StorageOp {
    Set,
    Append,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum StorageCmd {
    /// Read variable. Returns StorageResp::ReadVar or StorageResp::VariableMissing.
    ReadVar { name: String },

    /// Write variable.
    /// If `when_version_is == None`, always writes the variable and returns StorageResp::WriteVar.
    /// Otherwise, if the variable has the specified version, it writes the variable
    /// and returns StorageResp::WriteVar.
    /// Otherwise (version conflict), returns either StorageResp::ReadVar or StorageResp::VariableMissing
    /// just like ReadVar would.
    WriteVar {
        name: String,
        value: Vec<u8>,
        op: StorageOp,
        when_version_is: Option<u64>,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub enum StorageResp {
    /// Upon handling the request the variable had the specified value and version number.
    ReadVar { version: u64, value: Vec<u8> },
    /// Upon handling the request the variable was unset.
    VariableMissing {},
    /// The variable has been written, and the new version is returned.
    WriteVar { version: u64 },
}

pub fn storage_cmd(cmd: StorageCmd) -> StorageResp {
    let cmd_bytes = serde_json::to_vec(&cmd).unwrap();
    let res_id = unsafe { aici_host_storage_cmd(cmd_bytes.as_ptr(), cmd_bytes.len() as u32) };
    let resp_bytes = read_blob(res_id, 1024);
    serde_json::from_slice(&resp_bytes).unwrap()
}

// Public APIs

pub struct VariableStorage {
    // no fields yet
}

impl VariableStorage {
    /// Create a new instance of VariableStorage. It currently has no fields.
    pub fn new() -> Self {
        VariableStorage {}
    }

    /// Read variable. Returns None if the variable is unset.
    pub fn get(&self, name: &str) -> Option<Vec<u8>> {
        self.get_with_version(name).map(|x| x.1)
    }

    /// Write specified value to variable.
    pub fn set(&self, name: &str, value: Vec<u8>) {
        let _ver = self.write_var(name, value, StorageOp::Set);
    }

    /// Append specified value to variable.
    pub fn append(&self, name: &str, value: Vec<u8>) {
        let _ver = self.write_var(name, value, StorageOp::Append);
    }

    fn write_var(&self, name: &str, value: Vec<u8>, op: StorageOp) -> u64 {
        match storage_cmd(StorageCmd::WriteVar {
            name: name.to_string(),
            value,
            op,
            when_version_is: None,
        }) {
            StorageResp::WriteVar { version } => version,
            _ => panic!("unexpected response to writevar"),
        }
    }

    fn get_with_version(&self, name: &str) -> Option<(u64, Vec<u8>)> {
        match storage_cmd(StorageCmd::ReadVar {
            name: name.to_string(),
        }) {
            StorageResp::ReadVar { version, value } => Some((version, value)),
            StorageResp::VariableMissing {} => None,
            StorageResp::WriteVar { .. } => panic!("unexpected response to readvar"),
        }
    }
}

/// Tokenize given UTF8 string.
pub fn tokenize(s: &str) -> Vec<TokenId> {
    let id = unsafe { aici_host_tokenize(s.as_ptr(), s.len() as u32) };
    let r = read_blob(id, 4 * (s.len() / 3 + 10));
    let res = vec_from_bytes(&r);
    wprintln!("tokenize: {:?} -> {:?}", s, res);
    res
}

/// Return the ID of the current process.
pub fn self_seq_id() -> SeqId {
    unsafe { SeqId(aici_host_self_seq_id()) }
}
