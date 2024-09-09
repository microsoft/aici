use crate::{
    bindings::aici::abi::{self, tokenizer::TokenId},
    worker::{GroupCmd, GroupHandle, GroupResp},
};
use aici_abi::{
    toktrie::{TokRxInfo, TokTrie},
    StorageCmd,
};
use aicirt::wasi::clock::BoundedResolutionClock;
use aicirt::{api::InferenceCapabilities, bindings::SeqId, user_error};
use anyhow::{anyhow, Result};
use std::{ops::Deref, sync::Arc, time::Duration};
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
    log: wasmtime_wasi::pipe::MemoryOutputPipe,
    printed_log: usize,
    pub globals: GlobalInfo,
    pub group_channel: GroupHandle,
    pub store_limits: wasmtime::StoreLimits,
    pub storage_log: Vec<StorageCmd>,
    pub wasi_ctx: wasmtime_wasi::WasiCtx,
    pub resource_table: wasmtime_wasi::ResourceTable,
}

const MAX_LOG: usize = 64 * 1024;

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

        let wall_clock =
            BoundedResolutionClock::new(Duration::from_nanos(limits.timer_resolution_ns));
        let monotonic_clock = wall_clock.clone();

        let log = wasmtime_wasi::pipe::MemoryOutputPipe::new(MAX_LOG);
        let stdout = log.clone();
        let stderr = log.clone();
        ModuleData {
            id,
            log: log,
            printed_log: 0,
            globals,
            group_channel,
            store_limits,
            storage_log: Vec::new(),
            wasi_ctx: wasmtime_wasi::WasiCtxBuilder::new()
                .wall_clock(wall_clock)
                .monotonic_clock(monotonic_clock)
                .stdout(stdout)
                .stderr(stderr)
                .build(),
            resource_table: wasmtime_wasi::ResourceTable::new(),
        }
    }

    pub fn string_log(&mut self) -> String {
        self.printed_log = 0;
        let logs = String::from_utf8_lossy(&self.log.contents()).to_string();
        logs
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
    pub fn tokenize_bytes_greedy(&mut self, s: impl AsRef<[u8]>) -> Result<Vec<TokenId>> {
        Ok(self.globals.tok_trie.tokenize_with_greedy_fallback(s.as_ref(), |s| {
            self.globals
                .hf_tokenizer
                .encode(s, false)
                .expect("tokenizer error")
                .get_ids()
                .to_vec()
        }))
    }
}

impl abi::tokenizer::Host for ModuleData {
    fn eos_token(&mut self) -> wasmtime::Result<abi::tokenizer::TokenId> {
        Ok(self.globals.tokrx_info.tok_eos)
    }

    fn tokenize(&mut self, s: String) -> wasmtime::Result<Vec<abi::tokenizer::TokenId>> {
        self.tokenize_bytes_greedy(s.as_bytes())
    }

    fn tokenize_bytes(&mut self, bytes: Vec<u8>) -> wasmtime::Result<Vec<TokenId>> {
        self.tokenize_bytes_greedy(&bytes)
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
