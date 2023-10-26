use aici_abi::bytes::{clone_vec_as_bytes, TokRxInfo};
use anyhow::{anyhow, Result};
use log::debug;
use std::sync::{Arc, RwLock};
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
    id: ModuleInstId,
    log: Vec<u8>,
    printed_log: usize,
    globals: Arc<RwLock<GlobalInfo>>,
    pub module_arg: Arc<String>,
    pub linker: Arc<wasmtime::Linker<ModuleData>>,
    pub instance: Option<wasmtime::Instance>,
    pub memory: Option<wasmtime::Memory>,
    pub module: wasmtime::Module,
    tokenizer: Option<Tokenizer>,
    pub store_limits: wasmtime::StoreLimits,
}

const MAXLOG: usize = 32 * 1024;

impl ModuleData {
    pub fn new(
        id: ModuleInstId,
        limits: &AiciLimits,
        module: &wasmtime::Module,
        module_arg: Arc<String>,
        linker: &Arc<wasmtime::Linker<ModuleData>>,
        globals: &Arc<RwLock<GlobalInfo>>,
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
            globals: globals.clone(),
            module_arg,
            module: module.clone(),
            linker: linker.clone(),
            instance: None,
            memory: None,
            tokenizer: None,
            store_limits,
        }
    }

    pub fn tokenize(&mut self, s: &str) -> Result<Vec<u32>> {
        if self.tokenizer.is_none() {
            let lock = self.globals.clone();
            let info = lock.read().unwrap();
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
        if !log::log_enabled!(log::Level::Debug) {
            return;
        }

        let data = &self.log[self.printed_log..];
        if data.len() == 0 {
            return;
        }

        let logs = String::from_utf8_lossy(data).to_string();
        self.printed_log = self.log.len();

        for line in logs.lines() {
            debug!("{}:{}> {}", self.id, name, line);
        }
    }
}

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

    // uint32_t aici_host_read_token_trie(uint8_t *dst, uint32_t size);
    linker.func_wrap(
        "env",
        "aici_host_read_token_trie",
        |mut caller: wasmtime::Caller<'_, ModuleData>, ptr: u32, len: u32| {
            let lock = caller.data().globals.clone();
            let info = lock.read().unwrap();
            write_caller_mem(&mut caller, ptr, len, &info.trie_bytes)
        },
    )?;

    // uint32_t aici_host_read_arg(uint8_t *dst, uint32_t size);
    linker.func_wrap(
        "env",
        "aici_host_read_arg",
        |mut caller: wasmtime::Caller<'_, ModuleData>, ptr: u32, len: u32| {
            let arg = caller.data().module_arg.clone();
            write_caller_mem(&mut caller, ptr, len, arg.as_bytes())
        },
    )?;

    // uint32_t aici_host_tokenize(const uint8_t *src, uint32_t src_size, uint32_t *dst, uint32_t dst_size);
    linker.func_wrap(
        "env",
        "aici_host_tokenize",
        |mut caller: wasmtime::Caller<'_, ModuleData>,
         src: u32,
         src_size: u32,
         dst: u32,
         dst_size: u32| {
            let m = read_caller_mem(&caller, src, src_size);
            let s = String::from_utf8_lossy(&m);
            let tokens = caller.data_mut().tokenize(&s);
            match tokens {
                Err(_) => 0,
                Ok(tokens) => {
                    let bytes = clone_vec_as_bytes(&tokens);
                    write_caller_mem(&mut caller, dst, 4 * dst_size, &bytes) / 4
                }
            }
        },
    )?;

    let linker = Arc::new(linker);
    Ok(linker)
}
