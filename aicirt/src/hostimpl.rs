use aici_abi::{
    bytes::{clone_vec_as_bytes, TokRxInfo},
    TokenId,
};
use anyhow::{anyhow, Result};
use log::info;
use std::sync::Arc;
use tokenizers::Tokenizer;

pub type ModuleInstId = usize;

#[derive(Debug, Clone)]
pub struct AiciLimits {
    pub max_memory_bytes: usize,
    pub max_step_epochs: u64,
    pub max_init_epochs: u64,
}

// this is available to functions called from wasm
pub struct ModuleData {
    pub id: ModuleInstId,
    log: Vec<u8>,
    printed_log: usize,
    pub globals: GlobalInfo,
    pub ff_tokens: Vec<TokenId>,
    pub module_arg: Arc<String>,
    tokenize_out: Vec<TokenId>,
    tokens_arg: Vec<TokenId>,
    pub linker: Arc<wasmtime::Linker<ModuleData>>,
    pub instance: Option<wasmtime::Instance>,
    pub memory: Option<wasmtime::Memory>,
    pub module: wasmtime::Module,
    tokenizer: Option<Tokenizer>,
    pub store_limits: wasmtime::StoreLimits,
}

const MAXLOG: usize = 32 * 1024;

pub struct BlobId(u32);

impl ModuleData {
    pub const ARG_ID: BlobId = BlobId(1);
    pub const TOKENIZE_ID: BlobId = BlobId(2);
    pub const TRIE_ID: BlobId = BlobId(3);
    pub const TOKENS_ID: BlobId = BlobId(4);

    pub fn new(
        id: ModuleInstId,
        limits: &AiciLimits,
        module: &wasmtime::Module,
        module_arg: Arc<String>,
        linker: &Arc<wasmtime::Linker<ModuleData>>,
        globals: GlobalInfo,
    ) -> Self {
        let store_limits = wasmtime::StoreLimitsBuilder::new()
            .memories(1)
            .memory_size(limits.max_memory_bytes)
            .tables(2)
            .table_elements(100000)
            .instances(1)
            .trap_on_grow_failure(true)
            .build();
        ModuleData {
            id,
            log: Vec::new(),
            printed_log: 0,
            globals,
            module_arg,
            module: module.clone(),
            linker: linker.clone(),
            instance: None,
            memory: None,
            tokenizer: None,
            store_limits,
            ff_tokens: Vec::new(),
            tokenize_out: Vec::new(),
            tokens_arg: Vec::new(),
        }
    }

    pub fn set_tokens(&mut self, tokens: &[u32]) {
        self.tokens_arg.clear();
        self.tokens_arg.extend_from_slice(tokens);
    }

    pub fn tokenize(&mut self, s: &str) -> Result<Vec<u32>> {
        if self.tokenizer.is_none() {
            let info = &self.globals;
            let tok = Tokenizer::from_bytes(info.hf_tokenizer_bytes).unwrap();
            self.tokenizer = Some(tok);
        };
        let tokens = self.tokenizer.as_ref().unwrap().encode(s, false);
        match tokens {
            Err(e) => Err(anyhow!(e)),
            Ok(tokens) => Ok(Vec::from(tokens.get_ids())),
        }
    }

    pub fn write_log(&mut self, bytes: &[u8]) {
        self.log.extend_from_slice(bytes);
        if self.log.len() > MAXLOG {
            let drop = MAXLOG / 4;
            self.printed_log = self.printed_log.saturating_sub(drop);
            self.log.drain(0..drop);
        }
    }

    pub fn string_log(&mut self) -> String {
        self.printed_log = 0;
        let logs = String::from_utf8_lossy(&self.log).to_string();
        self.log.clear();
        logs
    }

    pub fn flush_logs(&mut self, name: &str) {
        if !log::log_enabled!(log::Level::Info) {
            return;
        }

        let data = &self.log[self.printed_log..];
        if data.len() == 0 {
            return;
        }

        let logs = String::from_utf8_lossy(data).to_string();
        self.printed_log = self.log.len();

        for line in logs.lines() {
            info!("{}:{}> {}", self.id, name, line);
        }
    }
}

#[derive(Clone)]
pub struct GlobalInfo {
    pub tokrx_info: TokRxInfo,
    pub trie_bytes: Vec<u8>,
    pub hf_tokenizer_bytes: &'static [u8],
}

fn read_caller_mem(caller: &wasmtime::Caller<'_, ModuleData>, ptr: u32, len: u32) -> Vec<u8> {
    let mem = caller.data().memory.unwrap();
    let ptr = ptr as usize;
    Vec::from(&mem.data(&caller)[ptr..(ptr + len as usize)])
}

fn write_caller_mem(
    caller: &mut wasmtime::Caller<'_, ModuleData>,
    ptr: u32,
    len: u32,
    src: &[u8],
) -> u32 {
    if len > 0 {
        let mem = caller.data().memory.unwrap();
        let min_len = std::cmp::min(len as usize, src.len());
        mem.write(caller, ptr as usize, &src[..min_len]).unwrap();
    }
    src.len() as u32
}

pub fn setup_linker(engine: &wasmtime::Engine) -> Result<Arc<wasmtime::Linker<ModuleData>>> {
    let mut linker = wasmtime::Linker::<ModuleData>::new(engine);
    linker.func_wrap(
        "env",
        "aici_host_print",
        |mut caller: wasmtime::Caller<'_, ModuleData>, ptr: u32, len: u32| {
            let m = read_caller_mem(&caller, ptr, len);
            caller.data_mut().write_log(&m);
        },
    )?;

    linker.func_wrap(
        "env",
        "aici_host_ff_token",
        |mut caller: wasmtime::Caller<'_, ModuleData>, tok: u32| {
            caller.data_mut().ff_tokens.push(tok);
        },
    )?;

    linker.func_wrap(
        "env",
        "aici_host_read_blob",
        |mut caller: wasmtime::Caller<'_, ModuleData>, blob_id: u32, ptr: u32, len: u32| {
            if blob_id == ModuleData::TRIE_ID.0 {
                // TODO remove .clone()
                let trie_bytes = caller.data().globals.trie_bytes.clone();
                write_caller_mem(&mut caller, ptr, len, &trie_bytes)
            } else if blob_id == ModuleData::ARG_ID.0 {
                let arg = caller.data().module_arg.clone();
                write_caller_mem(&mut caller, ptr, len, arg.as_bytes())
            } else if blob_id == ModuleData::TOKENIZE_ID.0 {
                let arg = clone_vec_as_bytes(&caller.data().tokenize_out);
                write_caller_mem(&mut caller, ptr, len, &arg)
            } else if blob_id == ModuleData::TOKENS_ID.0 {
                let arg = clone_vec_as_bytes(&caller.data().tokens_arg);
                write_caller_mem(&mut caller, ptr, len, &arg)
            } else {
                0
            }
        },
    )?;

    linker.func_wrap("env", "aici_host_module_arg", || ModuleData::ARG_ID.0)?;
    linker.func_wrap("env", "aici_host_token_trie", || ModuleData::TRIE_ID.0)?;
    linker.func_wrap("env", "aici_host_tokens", || ModuleData::TOKENS_ID.0)?;

    // uint32_t aici_host_tokenize(const uint8_t *src, uint32_t src_size, uint32_t *dst, uint32_t dst_size);
    linker.func_wrap(
        "env",
        "aici_host_tokenize",
        |mut caller: wasmtime::Caller<'_, ModuleData>, src: u32, src_size: u32| {
            let m = read_caller_mem(&caller, src, src_size);
            let s = String::from_utf8_lossy(&m);
            let tokens = caller.data_mut().tokenize(&s);
            match tokens {
                Err(_) => caller.data_mut().tokenize_out.clear(),
                Ok(tokens) => caller.data_mut().tokenize_out = tokens,
            }
            ModuleData::TOKENIZE_ID.0
        },
    )?;

    let linker = Arc::new(linker);
    Ok(linker)
}
