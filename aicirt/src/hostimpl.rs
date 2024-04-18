use crate::worker::{GroupCmd, GroupHandle, GroupResp, RtMidProcessArg};
use aici_abi::{
    bytes::{clone_vec_as_bytes, limit_str, vec_from_bytes, TokRxInfo},
    StorageCmd,
};
use aicirt::{api::InferenceCapabilities, shm::ShmAllocator, user_error};
use anyhow::{anyhow, Result};
use std::{
    rc::Rc,
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
    pub process_result: Vec<u8>,
    pub logit_shm: Rc<ShmAllocator>,
    pub logit_offsets: Vec<u32>,
    pub limits: AiciLimits,
    pub linker: Arc<wasmtime::Linker<ModuleData>>,
    pub instance: Option<wasmtime::Instance>,
    pub memory: Option<wasmtime::Memory>,
    pub module: wasmtime::Module,
    pub store_limits: wasmtime::StoreLimits,
    pub had_error: bool,
    pub storage_log: Vec<StorageCmd>,
    pub start_time: Instant,
    blobs: Vec<Rc<Vec<u8>>>,
}

const MAXLOG: usize = 64 * 1024;

pub const LOGIT_BIAS_ALLOW: f32 = 0.0;
pub const LOGIT_BIAS_DISALLOW: f32 = -100.0;

pub struct BlobId(u32);

impl BlobId {
    pub const MODULE_ARG: BlobId = BlobId(1);
    pub const TOKENIZE: BlobId = BlobId(2);
    pub const TOKENS: BlobId = BlobId(3);
    pub const PROCESS_ARG: BlobId = BlobId(4);
    pub const STORAGE_RESULT: BlobId = BlobId(5);

    pub const MAX_BLOB_ID: u32 = 20;

    // these have special handling:
    pub const TRIE: BlobId = BlobId(100);
}

impl ModuleData {
    pub fn new(
        id: ModuleInstId,
        limits: &AiciLimits,
        module: &wasmtime::Module,
        module_arg: String,
        linker: &Arc<wasmtime::Linker<ModuleData>>,
        globals: GlobalInfo,
        group_channel: GroupHandle,
        logit_shm: Rc<ShmAllocator>,
    ) -> Self {
        let store_limits = wasmtime::StoreLimitsBuilder::new()
            .memories(1)
            .memory_size(limits.max_memory_bytes)
            .tables(2)
            .table_elements(100000)
            .instances(1)
            .trap_on_grow_failure(true)
            .build();
        let mut r = ModuleData {
            id,
            log: Vec::new(),
            printed_log: 0,
            globals,
            group_channel,
            module: module.clone(),
            limits: limits.clone(),
            linker: linker.clone(),
            instance: None,
            memory: None,
            store_limits,
            process_result: Vec::new(),
            logit_shm,
            logit_offsets: Vec::new(),
            had_error: false,
            storage_log: Vec::new(),
            start_time: Instant::now(),
            blobs: vec![Rc::new(Vec::new()); BlobId::MAX_BLOB_ID as usize],
        };
        r.set_blob(BlobId::MODULE_ARG, module_arg.as_bytes().to_vec());
        r
    }

    fn clear_blob(&mut self, blob_id: BlobId) {
        self.set_blob(blob_id, vec![])
    }

    fn set_blob(&mut self, blob_id: BlobId, bytes: Vec<u8>) {
        self.blobs[blob_id.0 as usize] = Rc::new(bytes);
    }

    pub fn set_process_arg(&mut self, bytes: Vec<u8>) {
        self.process_result.clear();
        self.set_blob(BlobId::PROCESS_ARG, bytes);
    }

    pub fn set_mid_process_data(&mut self, data: RtMidProcessArg) {
        let bytes = serde_json::to_vec(&data.op).unwrap();
        self.set_process_arg(bytes);
        self.logit_offsets.clear();
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

    pub fn warn(&mut self, msg: &str) {
        log::warn!("{}: {}", self.id, msg);
        let msg = format!("warning: {}\n", msg);
        self.write_log(msg.as_bytes());
    }

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

    pub fn aici_host_storage_cmd(&mut self, m: Vec<u8>) -> BlobId {
        self.clear_blob(BlobId::STORAGE_RESULT);
        match serde_json::from_slice(&m) {
            Ok(cmd) => {
                let save = match &cmd {
                    StorageCmd::WriteVar { .. } => Some(cmd.clone()),
                    StorageCmd::ReadVar { .. } => None,
                };
                let res = self.group_channel.send_cmd(GroupCmd::StorageCmd { cmd });
                match res {
                    Ok(GroupResp::StorageResp { resp }) => {
                        if let Some(log) = save {
                            self.storage_log.push(log)
                        }
                        let res_bytes = serde_json::to_vec(&resp).unwrap();
                        self.set_blob(BlobId::STORAGE_RESULT, res_bytes);
                    }
                    // Ok(r) => self.fatal(&format!("storage_cmd invalid resp: {r:?}")),
                    Err(msg) => self.fatal(&format!("storage_cmd send error: {msg:?}")),
                }
            }
            Err(e) => self.fatal(&format!("storage_cmd error: {e:?}")),
        }
        BlobId::STORAGE_RESULT
    }
}

#[derive(Clone)]
pub struct GlobalInfo {
    pub inference_caps: InferenceCapabilities,
    pub tokrx_info: TokRxInfo,
    pub trie_bytes: Arc<Vec<u8>>,
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

macro_rules! fake_wasi {
    ($linker:ident, $func_name:ident, $($arg_type:ty)+) => {
        $linker.func_wrap(
            "wasi_snapshot_preview1",
            stringify!($func_name),
            |$(_: $arg_type),+| -> i32 {
                8 // BADF
                // 52 // NOSYS
            },
        )?;
    };
}

pub fn setup_linker(engine: &wasmtime::Engine) -> Result<Arc<wasmtime::Linker<ModuleData>>> {
    let mut linker = wasmtime::Linker::<ModuleData>::new(engine);

    fake_wasi!(linker, environ_get, i32 i32);
    fake_wasi!(linker, path_create_directory, i32 i32 i32);
    fake_wasi!(linker, path_filestat_get, i32 i32 i32 i32 i32);
    fake_wasi!(linker, path_link, i32 i32 i32 i32 i32 i32 i32);
    fake_wasi!(linker, path_open, i32 i32 i32 i32 i32 i64 i64 i32 i32);
    fake_wasi!(linker, path_readlink, i32 i32 i32 i32 i32 i32);
    fake_wasi!(linker, path_remove_directory, i32 i32 i32);
    fake_wasi!(linker, path_rename, i32 i32 i32 i32 i32 i32);
    fake_wasi!(linker, path_unlink_file, i32 i32 i32);
    fake_wasi!(linker, poll_oneoff, i32 i32 i32 i32);
    fake_wasi!(linker, fd_filestat_set_size, i32 i64);
    fake_wasi!(linker, fd_read, i32 i32 i32 i32);
    fake_wasi!(linker, fd_readdir, i32 i32 i32 i64 i32);
    fake_wasi!(linker, fd_close, i32);
    fake_wasi!(linker, fd_filestat_get, i32 i32);
    fake_wasi!(linker, fd_prestat_get, i32 i32);
    fake_wasi!(linker, fd_prestat_dir_name, i32 i32 i32);
    fake_wasi!(linker, fd_seek, i32 i64 i32 i32);
    fake_wasi!(linker, path_filestat_set_times, i32 i32 i32 i32 i64 i64 i32);

    linker.func_wrap("wasi_snapshot_preview1", "sched_yield", || 0)?;
    linker.func_wrap("wasi_snapshot_preview1", "fd_sync", |_: i32| 0)?;

    linker.func_wrap(
        "wasi_snapshot_preview1",
        "proc_exit",
        |code: i32| -> Result<()> { Err(user_error!("proc_exit: {code}")) },
    )?;

    linker.func_wrap(
        "wasi_snapshot_preview1",
        "fd_fdstat_get",
        |mut caller: wasmtime::Caller<'_, ModuleData>, fd: i32, stat_ptr: u32| -> Result<i32> {
            if fd != 0 && fd != 1 && fd != 2 {
                return Ok(8); // BADF
            }
            // pretend file isatty()
            let mut char_device = vec![0u8; 24];
            char_device[0] = 2;
            write_caller_mem(&mut caller, stat_ptr, 24, &char_device);
            Ok(0)
        },
    )?;

    linker.func_wrap(
        "wasi_snapshot_preview1",
        "clock_time_get",
        |mut caller: wasmtime::Caller<'_, ModuleData>,
         clock_id: i32,
         _precision: i64,
         dst_ptr: u32|
         -> Result<i32> {
            if clock_id != 1 {
                return Ok(63); // EPERM
            }
            let res = caller.data().limits.timer_resolution_ns as u64;
            let now = std::time::Instant::now();
            let nanos = now.duration_since(caller.data().start_time).as_nanos() as u64;
            let nanos = if res == 0 { 0 } else { nanos / res * res };
            let bytes = nanos.to_le_bytes();
            write_caller_mem(&mut caller, dst_ptr, 8, &bytes);
            Ok(0)
        },
    )?;

    linker.func_wrap(
        "wasi_snapshot_preview1",
        "fd_write",
        |mut caller: wasmtime::Caller<'_, ModuleData>,
         fd: i32,
         iovs_ptr: u32,
         niovs: u32,
         nwrittenptr: u32| {
            if fd != 1 && fd != 2 {
                return 8; // BADF
            }
            let iovs = read_caller_mem(&caller, iovs_ptr, niovs * 8);
            let ptr_lens = vec_from_bytes::<(u32, u32)>(&iovs);
            let mut nwr = 0;
            for (ptr, len) in ptr_lens {
                let m = read_caller_mem(&caller, ptr, len);
                nwr += m.len();
                caller.data_mut().write_log(&m);
            }
            if nwrittenptr != 0 {
                write_caller_mem(&mut caller, nwrittenptr, 4, &nwr.to_le_bytes());
            }
            0
        },
    )?;

    linker.func_wrap(
        "wasi_snapshot_preview1",
        "random_get",
        |mut caller: wasmtime::Caller<'_, ModuleData>, ptr: u32, len: u32| {
            write_caller_mem(&mut caller, ptr, len, &[]);
            0
        },
    )?;

    linker.func_wrap(
        "wasi_snapshot_preview1",
        "args_sizes_get",
        |mut caller: wasmtime::Caller<'_, ModuleData>, p1: u32, p2: u32| {
            let z = vec![0u8; 4];
            write_caller_mem(&mut caller, p1, 4, &z);
            write_caller_mem(&mut caller, p2, 4, &z);
            0
        },
    )?;

    linker.func_wrap(
        "wasi_snapshot_preview1",
        "environ_sizes_get",
        |mut caller: wasmtime::Caller<'_, ModuleData>, p1: u32, p2: u32| {
            let z = vec![0u8; 4];
            write_caller_mem(&mut caller, p1, 4, &z);
            write_caller_mem(&mut caller, p2, 4, &z);
            0
        },
    )?;
    linker.func_wrap("wasi_snapshot_preview1", "args_get", |_: u32, _: u32| 0)?;

    linker.func_wrap(
        "env",
        "aici_host_read_blob",
        |mut caller: wasmtime::Caller<'_, ModuleData>, blob_id: u32, ptr: u32, len: u32| {
            if blob_id == BlobId::TRIE.0 {
                let trie_bytes = caller.data().globals.trie_bytes.clone();
                write_caller_mem(&mut caller, ptr, len, &trie_bytes)
            } else if blob_id < BlobId::MAX_BLOB_ID {
                let blob = caller.data().blobs[blob_id as usize].clone();
                write_caller_mem(&mut caller, ptr, len, &blob)
            } else {
                fatal_error(&mut caller, "invalid blob_id");
                0
            }
        },
    )?;

    linker.func_wrap("env", "aici_host_module_arg", || BlobId::MODULE_ARG.0)?;
    linker.func_wrap("env", "aici_host_process_arg", || BlobId::PROCESS_ARG.0)?;
    linker.func_wrap("env", "aici_host_token_trie", || BlobId::TRIE.0)?;
    linker.func_wrap("env", "aici_host_tokens", || BlobId::TOKENS.0)?;

    // uint32_t aici_host_tokenize(const uint8_t *src, uint32_t src_size, uint32_t *dst, uint32_t dst_size);
    linker.func_wrap(
        "env",
        "aici_host_tokenize",
        |mut caller: wasmtime::Caller<'_, ModuleData>, src: u32, src_size: u32| {
            let m = read_caller_mem(&caller, src, src_size);
            let s = String::from_utf8_lossy(&m);
            let tokens = caller.data_mut().tokenize(&s);
            match tokens {
                Err(e) => {
                    caller.data_mut().warn(&format!("tokenize error: {e:?}"));
                    caller.data_mut().clear_blob(BlobId::TOKENIZE);
                }
                Ok(tokens) => {
                    caller
                        .data_mut()
                        .set_blob(BlobId::TOKENIZE, clone_vec_as_bytes(&tokens));
                }
            }
            BlobId::TOKENIZE.0
        },
    )?;

    linker.func_wrap(
        "env",
        "aici_host_return_logit_bias",
        |mut caller: wasmtime::Caller<'_, ModuleData>, src: u32| {
            let data = caller.data();

            let numtok = data.globals.tokrx_info.vocab_size as usize;
            let shm = data.logit_shm.clone();
            let id: u32 = data.id.try_into().unwrap();
            let numbytes = 4 * ((numtok + 31) / 32);

            // TODO two unnecessary copies here
            let m = read_caller_mem(&caller, src, numbytes as u32);
            let masks = vec_from_bytes::<u32>(&m);

            let off = shm.alloc(id).unwrap();
            let ptr = shm.slice_at_byte_offset::<f32>(off, numtok);

            for idx in 0..numtok {
                let mask = masks[idx / 32];
                let bit = 1 << (idx % 32);
                if mask & bit != 0 {
                    ptr[idx] = LOGIT_BIAS_ALLOW;
                } else {
                    ptr[idx] = LOGIT_BIAS_DISALLOW;
                }
            }

            caller.data_mut().logit_offsets.push(off as u32);
        },
    )?;

    linker.func_wrap(
        "env",
        "aici_host_self_seq_id",
        |caller: wasmtime::Caller<'_, ModuleData>| caller.data().id as u32,
    )?;

    linker.func_wrap(
        "env",
        "aici_host_get_config",
        |caller: wasmtime::Caller<'_, ModuleData>, name: u32, name_size: u32| {
            let m = read_caller_mem(&caller, name, name_size);
            let name = String::from_utf8_lossy(&m);
            let caps = serde_json::to_value(caller.data().globals.inference_caps.clone()).unwrap();
            if caps[name.as_ref()].as_bool().unwrap_or(false) {
                return 1;
            }
            return 0;
        },
    )?;

    linker.func_wrap(
        "env",
        "aici_host_eos_token",
        |caller: wasmtime::Caller<'_, ModuleData>| caller.data().globals.tokrx_info.tok_eos,
    )?;

    linker.func_wrap(
        "env",
        "aici_host_return_process_result",
        |mut caller: wasmtime::Caller<'_, ModuleData>, src: u32, src_size: u32| {
            let m = read_caller_mem(&caller, src, src_size);
            caller.data_mut().process_result = m;
        },
    )?;

    linker.func_wrap(
        "env",
        "aici_host_storage_cmd",
        |mut caller: wasmtime::Caller<'_, ModuleData>, src: u32, src_size: u32| {
            let m = read_caller_mem(&caller, src, src_size);
            let r = caller.data_mut().aici_host_storage_cmd(m);
            check_fatal(&mut caller);
            r.0
        },
    )?;

    linker.func_wrap("env", "aici_host_stop", || {
        Err::<(), _>(user_error!("*** aici_host_stop()"))
    })?;

    let linker = Arc::new(linker);
    Ok(linker)
}
