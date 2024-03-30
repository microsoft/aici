use crate::{
    bindings::aici::abi::{self, tokenizer::TokenId},
    worker::{GroupCmd, GroupHandle, GroupResp},
};
use aici_abi::{
    bytes::limit_str,
    toktrie::{TokRxInfo, TokTrie},
    StorageCmd,
};
use aicirt::{api::InferenceCapabilities, bindings::SeqId, user_error};
use anyhow::{anyhow, Result};
use std::{
    ops::Deref,
    sync::Arc,
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct AiciLimits {
    pub ipc_shm_bytes: usize,

    pub timer_resolution_ns: u64,
    pub max_memory_bytes: usize,
    pub max_step_ms: u64,
    pub max_init_ms: u64,
    pub max_compile_ms: u64,
    pub max_timeout_steps: usize,
    pub logit_memory_bytes: usize,
    pub busy_wait_duration: Duration,
    pub max_forks: usize,

    pub module_upload: bool,
    pub gh_download: bool,
}

type ModuleInstId = crate::api::ModuleInstId;

// this is available to functions called from wasm
pub struct ModuleData {
    pub id: ModuleInstId,
    log: Vec<u8>,
    printed_log: usize,
    pub globals: GlobalInfo,
    pub group_channel: GroupHandle,
    pub store_limits: wasmtime::StoreLimits,
    pub had_error: bool,
    pub storage_log: Vec<StorageCmd>,
    pub start_time: Instant,
    pub wasi_ctx: wasmtime_wasi::WasiCtx,
    pub resource_table: wasmtime_wasi::ResourceTable,
}

const MAXLOG: usize = 64 * 1024;

impl ModuleData {
    pub fn new(
        id: ModuleInstId,
        limits: &AiciLimits,
        globals: GlobalInfo,
        group_channel: GroupHandle,
    ) -> Self {
        let store_limits = wasmtime::StoreLimitsBuilder::new()
            .memories(1)
            .memory_size(limits.max_memory_bytes)
            .tables(2)
            .table_elements(100000)
            .instances(100)
            .trap_on_grow_failure(true)
            .build();
        ModuleData {
            id,
            log: Vec::new(),
            printed_log: 0,
            globals,
            group_channel,
            store_limits,
            had_error: false,
            storage_log: Vec::new(),
            start_time: Instant::now(),
            wasi_ctx: wasmtime_wasi::WasiCtxBuilder::new().build(),
            resource_table: wasmtime_wasi::ResourceTable::new(),
        }
    }

    pub fn tokenize(&mut self, s: &str) -> Result<Vec<u32>> {
        let tokens = self.globals.hf_tokenizer.encode(s, false);
        match tokens {
            Err(e) => Err(anyhow!(e)),
            Ok(tokens) => Ok(Vec::from(tokens.get_ids())),
        }
    }

    pub fn fatal(&mut self, msg: &str) {
        log::warn!("{}: fatal error {}", self.id, msg);
        let msg = format!("FATAL ERROR: {}\n", msg);
        self.write_log(msg.as_bytes());
        self.had_error = true;
        // ideally, this should call into the module and cause panic
    }

    // pub fn warn(&mut self, msg: &str) {
    //     log::warn!("{}: {}", self.id, msg);
    //     let msg = format!("warning: {}\n", msg);
    //     self.write_log(msg.as_bytes());
    // }

    pub fn write_log(&mut self, bytes: &[u8]) {
        self.log.extend_from_slice(bytes);
        if self.log.len() > MAXLOG {
            let drop = MAXLOG / 4;
            if self.had_error {
                // normally, we drop prefix, but if "had_error" is set
                // we drop the suffix instead to avoid flushing out "FATAL ERROR" message
                self.log.truncate(self.log.len() - drop);
            } else {
                self.printed_log = self.printed_log.saturating_sub(drop);
                self.log.drain(0..drop);
            }
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
            log::debug!("{}:{}> {}", self.id, name, limit_str(line, 512));
        }
    }
}

impl wasmtime_wasi::WasiView for ModuleData {
    fn table(&mut self) -> &mut wasmtime_wasi::ResourceTable {
        &mut self.resource_table
    }

    fn ctx(&mut self) -> &mut wasmtime_wasi::WasiCtx {
        &mut self.wasi_ctx
    }
}

#[derive(Clone)]
pub struct GlobalInfo {
    pub inference_caps: InferenceCapabilities,
    pub tokrx_info: TokRxInfo,
    pub trie_bytes: Arc<Vec<u8>>,
    pub tok_trie: Arc<TokTrie>,
    pub hf_tokenizer: Arc<Tokenizer>,
}

fn check_fatal(caller: &mut wasmtime::Caller<'_, ModuleData>) {
    if caller.data().had_error {
        fatal_error(caller, "see above")
    }
}

fn fatal_error(caller: &mut wasmtime::Caller<'_, ModuleData>, msg: &str) {
    caller.data_mut().fatal(msg);
    match caller.get_export("aici_panic") {
        Some(wasmtime::Extern::Func(f)) => {
            let mut res = Vec::new();
            let _ = f.call(caller, &[], &mut res);
        }
        _ => {}
    }
}

impl abi::runtime::Host for ModuleData {
    fn sequence_id(&mut self) -> wasmtime::Result<SeqId> {
        Ok(self.id as SeqId)
    }

    fn stop(&mut self) -> wasmtime::Result<()> {
        Err::<(), _>(user_error!("*** aici_host_stop()"))
    }

    fn get_config(&mut self, key: String) -> wasmtime::Result<i32> {
        let caps = serde_json::to_value(self.globals.inference_caps.clone()).unwrap();
        if caps[&key].as_bool().unwrap_or(false) {
            Ok(1)
        } else {
            Ok(0)
        }
    }
}

impl abi::runtime_storage::Host for ModuleData {
    fn get(&mut self, name: String) -> wasmtime::Result<Option<Vec<u8>>> {
        let res = self.group_channel.send_cmd(GroupCmd::StorageCmd {
            cmd: StorageCmd::ReadVar { name },
        })?;

        match res {
            GroupResp::StorageResp { resp } => match resp {
                aici_abi::StorageResp::ReadVar { version: _, value } => Ok(Some(value)),
                aici_abi::StorageResp::VariableMissing {} => Ok(None),
                aici_abi::StorageResp::WriteVar { .. } => Err(anyhow!("unexpected WriteVar")),
            },
        }
    }

    fn set(&mut self, name: String, value: Vec<u8>) -> wasmtime::Result<()> {
        let cmd = StorageCmd::WriteVar {
            name,
            value,
            op: aici_abi::StorageOp::Set,
            when_version_is: None,
        };
        self.storage_log.push(cmd.clone());
        let res = self.group_channel.send_cmd(GroupCmd::StorageCmd { cmd })?;
        match res {
            GroupResp::StorageResp { resp } => match resp {
                aici_abi::StorageResp::ReadVar { .. } => Err(anyhow!("unexpected ReadVar")),
                aici_abi::StorageResp::VariableMissing { .. } => {
                    Err(anyhow!("unexpected VariableMissing"))
                }
                aici_abi::StorageResp::WriteVar { .. } => Ok(()),
            },
        }
    }

    fn append(&mut self, name: String, value: Vec<u8>) -> wasmtime::Result<()> {
        let cmd = StorageCmd::WriteVar {
            name,
            value,
            op: aici_abi::StorageOp::Append,
            when_version_is: None,
        };
        self.storage_log.push(cmd.clone());
        let res = self.group_channel.send_cmd(GroupCmd::StorageCmd { cmd })?;
        match res {
            GroupResp::StorageResp { resp } => match resp {
                aici_abi::StorageResp::ReadVar { .. } => Err(anyhow!("unexpected ReadVar")),
                aici_abi::StorageResp::VariableMissing { .. } => {
                    Err(anyhow!("unexpected VariableMissing"))
                }
                aici_abi::StorageResp::WriteVar { .. } => Ok(()),
            },
        }
    }
}

impl ModuleData {
    pub fn tokenize_str(&mut self, s: &str) -> Result<Vec<TokenId>> {
        let tokens = self.globals.hf_tokenizer.encode(s, false);
        match tokens {
            Err(e) => Err(anyhow!(e)),
            Ok(tokens) => Ok(Vec::from(tokens.get_ids())),
        }
    }
}

impl abi::tokenizer::Host for ModuleData {
    fn eos_token(&mut self) -> wasmtime::Result<abi::tokenizer::TokenId> {
        Ok(self.globals.tokrx_info.tok_eos)
    }

    fn tokenize(&mut self, s: String) -> wasmtime::Result<Vec<abi::tokenizer::TokenId>> {
        self.tokenize_str(&s)
    }

    fn tokenize_bytes(&mut self, bytes: Vec<u8>) -> wasmtime::Result<Vec<TokenId>> {
        let s = String::from_utf8_lossy(&bytes);
        self.tokenize(&s)
    }

    fn token_trie_bytes(&mut self) -> wasmtime::Result<Vec<u8>> {
        Ok(self.globals.trie_bytes.deref().clone())
    }
}

pub fn setup_component_linker(
    engine: &wasmtime::Engine,
) -> Result<Arc<wasmtime::component::Linker<ModuleData>>> {
    let mut linker = wasmtime::component::Linker::<ModuleData>::new(engine);

    crate::bindings::Aici::add_to_linker(&mut linker, |m| m)?;
    wasmtime_wasi::add_to_linker_sync(&mut linker)?;

    let linker = Arc::new(linker);
    Ok(linker)
}
