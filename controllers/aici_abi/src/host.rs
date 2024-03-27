use crate::{
    bytes::{vec_from_bytes, TokenId},
    svob::SimpleVob,
    SeqId,
};
use serde::{Deserialize, Serialize};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlobId(u32);

#[allow(dead_code)]
extern "C" {
    // Read binary blob.
    // Always returns the size of the blob, will write up to `size` bytes to `dst`.
    fn aici_host_read_blob(blob: BlobId, dst: *mut u8, size: u32) -> u32;

    // Return the ID of TokTrie binary representation.
    fn aici_host_token_trie() -> BlobId;

    // Return the ID of argument passed by the user.
    fn aici_host_module_arg() -> BlobId;

    // Return the ID of argument passed to the process() function.
    // It's a JSON serialization of Pre/Mid/PostProcessArg.
    fn aici_host_process_arg() -> BlobId;

    // Tokenize given UTF8 string. The result is only valid until next call to this function.
    fn aici_host_tokenize(src: *const u8, src_size: u32) -> BlobId;

    // Set logit bias based on bit-mask in src.
    fn aici_host_return_logit_bias(src: *const u32);

    fn aici_host_self_seq_id() -> u32;

    fn aici_host_return_process_result(res: *const u8, res_size: u32);

    fn aici_host_storage_cmd(cmd: *const u8, cmd_size: u32) -> BlobId;

    // This can be also obtained from the TokTrie.
    fn aici_host_eos_token() -> TokenId;

    // Stop the program - any error info is assumed to have been printed already.
    // Backtraces will be limited.
    fn aici_host_stop();
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

fn init_panic() {
    #[cfg(target_arch = "wasm32")]
    std::panic::set_hook(Box::new(|info| {
        // skip 'run with `RUST_BACKTRACE=1`' message (not relevant for remote running)
        println!("{}", info);
    }))
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

pub fn arg_string() -> String {
    String::from_utf8_lossy(&arg_bytes()).to_string()
}

pub fn trie_bytes() -> Vec<u8> {
    #[cfg(target_arch = "wasm32")]
    return read_blob(unsafe { aici_host_token_trie() }, 0);

    #[cfg(not(target_arch = "wasm32"))]
    return std::fs::read("tokenizer.bin").unwrap();
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum StorageOp {
    Set,
    Append,
}

#[allow(dead_code)]
pub mod bin_string {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        let binstr = String::from_iter(v.iter().map(|b| *b as char));
        String::serialize(&binstr, s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let binstr = String::deserialize(d)?;
        Ok(binstr.chars().map(|c| c as u8).collect())
    }
}

pub mod hex_string {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use crate::bytes::{from_hex_string, to_hex_string};

    pub fn serialize<S: Serializer>(v: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        let hexstr = to_hex_string(v);
        String::serialize(&hexstr, s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let hexstr = String::deserialize(d)?;
        from_hex_string(&hexstr).map_err(serde::de::Error::custom)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
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
        #[serde(with = "hex_string")]
        value: Vec<u8>,
        op: StorageOp,
        when_version_is: Option<u64>,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub enum StorageResp {
    /// Upon handling the request the variable had the specified value and version number.
    ReadVar {
        version: u64,
        #[serde(with = "hex_string")]
        value: Vec<u8>,
    },
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
    // no fields (yet?)
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
            _ => panic!("unexpected response to write var"),
        }
    }

    fn get_with_version(&self, name: &str) -> Option<(u64, Vec<u8>)> {
        match storage_cmd(StorageCmd::ReadVar {
            name: name.to_string(),
        }) {
            StorageResp::ReadVar { version, value } => Some((version, value)),
            StorageResp::VariableMissing {} => None,
            StorageResp::WriteVar { .. } => panic!("unexpected response to read var"),
        }
    }
}

/// Tokenize given byte string.
pub fn tokenize_bytes(s: &[u8]) -> Vec<TokenId> {
    let id = unsafe { aici_host_tokenize(s.as_ptr(), s.len() as u32) };
    let r = read_blob(id, 4 * (s.len() / 3 + 10));
    let res = vec_from_bytes(&r);
    // println!(
    //     "tokenize_bytes: {:?} -> {:?}",
    //     String::from_utf8_lossy(s),
    //     res
    // );
    res
}

/// Tokenize given UTF8 string.
pub fn tokenize(s: &str) -> Vec<TokenId> {
    let id = unsafe { aici_host_tokenize(s.as_ptr(), s.len() as u32) };
    let r = read_blob(id, 4 * (s.len() / 3 + 10));
    let res = vec_from_bytes(&r);
    // println!("tokenize: {:?} -> {:?}", s, res);
    res
}

/// Return the ID of the current process.
pub fn self_seq_id() -> SeqId {
    unsafe { SeqId(aici_host_self_seq_id()) }
}

/// Return the ID of the EOS token.
pub fn eos_token() -> TokenId {
    unsafe { aici_host_eos_token() }
}

/// Stop the program - any error info is assumed to have been printed already.
pub fn aici_stop() -> ! {
    unsafe { aici_host_stop() };
    panic!("didn't stop");
}
