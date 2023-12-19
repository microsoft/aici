mod bench;
mod hostimpl;
mod moduleinstance;
mod worker;

use aici_abi::bytes::limit_str;
use aici_abi::toktree::TokTrie;
use aici_abi::{MidProcessArg, PostProcessArg, PreProcessArg, SeqId};
use aici_tokenizers::find_tokenizer;
use aicirt::*;
use anyhow::{anyhow, ensure, Result};
use base64;
use base64::Engine as _;
use bench::{TimerRef, TimerSet};
use clap::Parser;
use hex;
use hostimpl::GlobalInfo;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thread_priority::*;
use worker::{fork_child, RtPostProcessArg, RtPreProcessArg, SeqWorkerHandle};

use crate::hostimpl::*;
use crate::moduleinstance::*;
use crate::msgchannel::MessageChannel;
use crate::shm::Shm;
use crate::worker::{bench_ipc, RtMidProcessArg, WorkerForker};

use crate::api::*;

// Both of these are percentage of available cores
const BG_THREADS_FRACTION: usize = 50;
const STEP_THREADS_FRACTION: usize = 90;

const MEGABYTE: usize = 1024 * 1024;

#[derive(Parser, Clone)]
struct Cli {
    /// Path to .wasm module to install
    #[arg(short, long)]
    module: Option<String>,

    /// Path to .json metadata for module to install
    #[arg(short = 'j', long)]
    module_meta: Option<String>,

    /// Tokenizer to use; try --tokenizer list to see options
    #[arg(short, long, default_value = "llama")]
    tokenizer: String,

    /// Save the --tokenizer=... to specified file
    #[arg(long)]
    save_tokenizer: Option<String>,

    /// Run main() from the module just added
    #[arg(short, long)]
    run: bool,

    /// Path to argument to pass.
    #[arg(long)]
    run_arg: Option<PathBuf>,

    /// Run with POSIX shared memory interface
    #[arg(short, long)]
    server: bool,

    /// Run benchmarks
    #[arg(long)]
    bench: bool,

    /// Fork test
    #[arg(long)]
    fork: bool,

    /// Size of JSON comm buffer in megabytes
    #[arg(long, default_value = "8")]
    json_size: usize,

    /// Size of binary comm buffer in megabytes
    #[arg(long, default_value = "16")]
    bin_size: usize,

    /// How many milliseconds to spin-wait for a message over IPC and SHM.
    #[arg(long, default_value = "200")]
    busy_wait_time: u64,

    /// Maximum number of concurrent forks of a WASM module in a single request
    #[arg(long, default_value = "16")]
    wasm_max_forks: usize,

    /// Maximum size of WASM module memory in megabytes
    #[arg(long, default_value = "64")]
    wasm_max_memory: usize,

    /// Maximum time WASM module can execute step preparation in milliseconds
    #[arg(long, default_value = "4")]
    wasm_max_pre_step_time: u64,

    /// Maximum time WASM module can execute step in milliseconds
    #[arg(long, default_value = "150")]
    wasm_max_step_time: u64,

    /// Maximum time WASM module can execute initialization code in milliseconds
    #[arg(long, default_value = "1000")]
    wasm_max_init_time: u64,

    /// Shm/semaphore name prefix
    #[arg(long, short, default_value = "/aici0-")]
    name: String,
}

impl Cli {
    pub fn prefixed_name(&self, name: &str, name2: &str) -> String {
        format!("{}{}{}", self.name, name, name2)
    }
}

enum ModuleStatus {
    Missing,
    Locked,
    Ready,
}

// this is cloned for every module-level request, so don't go overboard with fields
#[derive(Clone)]
struct ModuleRegistry {
    wasm_ctx: Arc<WasmContext>,
    cache_path: PathBuf,
    // maps module_id (sha256 string) to module status
    modules: Arc<Mutex<HashMap<String, ModuleStatus>>>,
    req_instances: Arc<Mutex<HashMap<String, SeqWorkerHandle>>>,
    // note sure Mutex is needed
    forker: Arc<Mutex<WorkerForker>>,
}

struct Stepper {
    req_instances: Arc<Mutex<HashMap<String, SeqWorkerHandle>>>,
    instances: HashMap<ModuleInstId, SeqWorkerHandle>,
    limits: AiciLimits,
    globals: GlobalInfo,
    // for debugging
    shm: Shm,
    token_bytes: Vec<Vec<u8>>,
    pre_timer: TimerRef,
    pre_recv_timer: TimerRef,
}

fn is_hex_string(s: &str) -> bool {
    s.chars().all(|c| c.is_digit(16))
}

impl ModuleRegistry {
    pub fn new(wasm_ctx: WasmContext, shm: Shm) -> Result<Self> {
        let forker = WorkerForker::new(wasm_ctx.clone(), shm);

        Ok(Self {
            forker: Arc::new(Mutex::new(forker)),
            cache_path: PathBuf::from("./cache"),
            wasm_ctx: Arc::new(wasm_ctx),
            modules: Arc::new(Mutex::new(HashMap::new())),
            req_instances: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn module_needs_check(&self, module_id: &str) -> bool {
        loop {
            let mut lck = self.modules.lock().unwrap();
            match *lck.get(module_id).unwrap_or(&ModuleStatus::Missing) {
                ModuleStatus::Locked => std::thread::sleep(std::time::Duration::from_millis(50)),
                ModuleStatus::Ready => return false,
                ModuleStatus::Missing => {
                    // we lock it
                    lck.insert(module_id.to_string(), ModuleStatus::Locked);
                    return true;
                }
            }
        }
    }

    fn wasm_path(&self, module_id: &str) -> PathBuf {
        self.cache_path.join(format!("{}.wasm", module_id))
    }

    fn elf_path(&self, module_id: &str) -> PathBuf {
        self.cache_path.join(format!("{}.elf", module_id))
    }

    fn compile_module(&self, module_id: &str, force: bool) -> Result<()> {
        let module = if force {
            Err(anyhow!("force"))
        } else {
            self.wasm_ctx.deserialize_module(self.elf_path(module_id))
        };

        match module {
            Err(e) => {
                let wasm_bytes = fs::read(self.wasm_path(module_id))?;
                log::info!("compiling {}; {}", module_id, e);
                let compiled = self.wasm_ctx.engine.precompile_module(&wasm_bytes)?;
                fs::write(self.elf_path(module_id), compiled)?;
                // make sure we can deserialize it
                let _ = self.wasm_ctx.deserialize_module(self.elf_path(module_id))?;
            }
            Ok(_) => {}
        };

        let mut lck = self.modules.lock().unwrap();
        lck.insert(module_id.to_string(), ModuleStatus::Ready);
        return Ok(());
    }

    fn ensure_module_in_fs(&self, module_id: &str) -> Result<PathBuf> {
        if self.module_needs_check(module_id) {
            match self.compile_module(module_id, false) {
                Ok(_) => {}
                Err(e) => {
                    let mut lck = self.modules.lock().unwrap();
                    lck.remove(module_id);
                    return Err(e);
                }
            }
        }

        Ok(self.elf_path(module_id))
    }

    fn create_module(&self, wasm_bytes: Vec<u8>, meta_bytes: Vec<u8>) -> Result<Value> {
        let timer = Instant::now();

        // make sure meta_bytes is valid JSON
        let _: Value = serde_json::from_slice(&meta_bytes)?;

        let mut hasher = Sha256::new();
        hasher.update(&meta_bytes);
        hasher.update(&wasm_bytes);

        let module_id = hex::encode(hasher.finalize());
        let module_id = &module_id;

        if self.module_needs_check(module_id) {
            match self.write_and_compile(module_id, &meta_bytes, &wasm_bytes) {
                Err(e) => {
                    let mut lck = self.modules.lock().unwrap();
                    lck.remove(module_id);
                    return Err(e);
                }
                Ok(_) => {}
            }
        }

        let compiled_size = fs::metadata(self.elf_path(module_id))?.len() as usize;
        let time = timer.elapsed().as_millis() as u64;

        log::info!(
            "module {}: {}kB -> {}kB; {}ms",
            module_id,
            wasm_bytes.len() / 1024,
            compiled_size / 1024,
            time
        );

        Ok(serde_json::to_value(MkModuleResp {
            module_id: module_id.to_string(),
            wasm_size: wasm_bytes.len(),
            meta_size: meta_bytes.len(),
            compiled_size,
            time,
        })?)
    }

    fn write_and_compile(
        &self,
        module_id: &String,
        meta_bytes: &Vec<u8>,
        wasm_bytes: &Vec<u8>,
    ) -> Result<()> {
        fs::create_dir_all(&self.cache_path)?;
        Ok(if !self.wasm_path(module_id).exists() {
            let jsonpath = self.cache_path.join(format!("{}.json", module_id));
            fs::write(jsonpath, meta_bytes)?;
            fs::write(self.wasm_path(module_id), wasm_bytes)?;
            self.compile_module(module_id, true)?
        } else {
            self.compile_module(module_id, false)?
        })
    }

    fn mk_module(&self, req: MkModuleReq) -> Result<Value> {
        let wasm_bytes = base64::engine::general_purpose::STANDARD.decode(req.binary)?;
        let meta_bytes = serde_json::to_vec(&req.meta)?;
        self.create_module(wasm_bytes, meta_bytes)
    }

    fn instantiate(&mut self, req: InstantiateReq) -> Result<Value> {
        ensure!(is_hex_string(&req.module_id), "invalid module_id");
        let module_path = self.ensure_module_in_fs(&req.module_id)?;
        log::debug!("instance {} -> {}", req.module_id, req.req_id);
        let (handle, res) = self
            .forker
            .lock()
            .unwrap()
            .instantiate(req.clone(), module_path)?;
        let mut req_instances = self.req_instances.lock().unwrap();
        req_instances.insert(req.req_id, handle);
        Ok(serde_json::to_value(res)?)
    }

    fn run_main(&self, req_id: &String) -> Result<()> {
        let req_instances = self.req_instances.lock().unwrap();
        let inst = req_instances.get(req_id).ok_or(anyhow!("invalid req_id"))?;
        inst.run_main()
    }

    pub fn dispatch_loop(&self, ch: CmdRespChannel) -> ! {
        loop {
            let msg = ch.recv();
            let mut s2 = self.clone();
            let resp_lck = ch.resp_ch.clone();
            rayon::spawn(move || {
                let r = s2.exec_wrapped(&msg);
                resp_lck
                    .lock()
                    .unwrap()
                    .send(serde_json::to_vec(&r).unwrap().as_slice())
                    .unwrap();
            });
        }
    }
}

impl Stepper {
    pub fn new(
        reg: &ModuleRegistry,
        limits: AiciLimits,
        shm: Shm,
        token_bytes: Vec<Vec<u8>>,
    ) -> Result<Self> {
        Ok(Self {
            req_instances: reg.req_instances.clone(),
            instances: HashMap::new(),
            limits,
            globals: reg.wasm_ctx.globals.clone(),
            shm,
            token_bytes,
            pre_timer: reg.wasm_ctx.timers.new_timer("pre"),
            pre_recv_timer: reg.wasm_ctx.timers.new_timer("pre_recv"),
        })
    }

    fn get_worker(&self, id: ModuleInstId) -> Result<&SeqWorkerHandle> {
        Ok(self
            .instances
            .get(&id)
            .ok_or(anyhow!("invalid id {}", id))?)
    }

    fn token_name(&self, idx: usize) -> String {
        if idx >= self.token_bytes.len() {
            format!("<{idx}>")
        } else {
            format!(
                "{:?}",
                String::from_utf8_lossy(&self.token_bytes[idx as usize])
            )
        }
    }

    // returns the parent id (if any) or current module id otherwise
    fn maybe_fork(
        &mut self,
        id: ModuleInstId,
        clone_id: Option<ModuleInstId>,
    ) -> Result<ModuleInstId> {
        if let Some(parent_id) = clone_id {
            ensure!(
                !self.instances.contains_key(&id),
                "duplicate id {id} (cloning {parent_id})"
            );
            let parent = self.get_worker(parent_id)?;
            let num_forks = self
                .instances
                .values()
                .filter(|r| r.req_id == parent.req_id)
                .count();
            if num_forks + 1 > self.limits.max_forks {
                anyhow::bail!("too many forks (max={})", self.limits.max_forks)
            }
            log::debug!("fork {} -> ({})", parent_id, id);
            // TODO the forks should be done in parallel, best in tree-like fashion
            let h = parent.fork(id)?;
            self.instances.insert(id, h);
            Ok(parent_id)
        } else {
            // make sure worker exists
            self.get_worker(id)?;
            Ok(id)
        }
    }

    fn mk_instance(&mut self, op: &AiciPreOp) -> Result<()> {
        if let Some(req_id) = op.req_id.clone() {
            let e = { self.req_instances.lock().unwrap().remove(&req_id) };
            ensure!(e.is_some(), "invalid req_id {req_id}");
            let id = op.id;
            ensure!(!self.instances.contains_key(&id), "duplicate id {id}");
            let h = e.unwrap();
            log::debug!("prompt {} ({})", id, req_id);
            h.set_id(id)?;
            self.instances.insert(id, h);
        }
        Ok(())
    }

    fn aici_pre_process(
        &mut self,
        is_all: bool,
        req: AiciPreProcessReq,
    ) -> Result<AiciPreProcessResp> {
        for id in req.freed {
            log::debug!("free module {}", id);
            self.instances.remove(&id);
        }

        // first, start instances and link clones
        for op in req.ops.iter() {
            self.mk_instance(&op)?;
        }

        let op_ids: Vec<ModuleInstId> = if is_all {
            self.instances.keys().map(|x| *x).collect()
        } else {
            req.ops.iter().map(|op| op.id).collect()
        };

        let mut used_ids = Vec::new();
        let mut outputs = HashMap::new();
        let block_elts = req.max_context_len;
        let mut idx = 0;

        for instid in op_ids {
            if let Ok(h) = self.get_worker(instid) {
                let op = RtPreProcessArg {
                    op: PreProcessArg {},
                    max_context_size: req.max_context_len,
                    allow_ff_tokens: is_all,
                };
                match h.start_pre_process(op) {
                    Ok(_) => used_ids.push((idx, instid)),
                    Err(e) => self.worker_error(instid, &mut outputs, e),
                };
            } else {
                log::warn!("invalid id {}", instid);
            }
            idx += 1;
        }

        let deadline =
            Instant::now() + std::time::Duration::from_millis(self.limits.max_pre_step_ms);

        let mut all_masks = Vec::new();
        let mut curr_req_masks = Vec::new();
        let mut curr_req_id = "".to_string();
        let mut suspend_ids = Vec::new();

        for (op_idx, id) in used_ids {
            let h = self.get_worker(id).unwrap();
            if h.req_id != curr_req_id {
                all_masks.append(&mut curr_req_masks);
                curr_req_id = h.req_id.clone();
            }
            let timeout = deadline.saturating_duration_since(Instant::now());
            match h.check_pre_process(timeout, &self.pre_recv_timer) {
                Ok(mut data) => {
                    outputs.insert(
                        id,
                        data.json.clone_with(Some(AiciPreProcessResultInner {
                            suspend: data.suspend,
                            num_forks: data.attn_masks.len(),
                            ff_tokens: data.ff_tokens,
                        })),
                    );
                    let len = data.attn_masks.len();
                    if data.suspend {
                        suspend_ids.push(op_idx);
                        assert!(len == 1);
                        assert!(data.attn_masks[0].len() == 0);
                    } else if len >= 1 {
                        // first mask goes in place of the current sequence
                        all_masks.push((op_idx, data.attn_masks.remove(0)));

                        // other masks need to go after all sequences of the current sequence group (req_id)
                        for e in data.attn_masks {
                            curr_req_masks.push((op_idx, e));
                        }
                    }
                }
                Err(e) => self.worker_error(id, &mut outputs, e),
            }
        }

        // add masks of the last req
        all_masks.append(&mut curr_req_masks);

        let mut block_off = 0;
        let mut fork_map = Vec::<usize>::new();
        for (op_idx, mask) in &all_masks {
            let dst = self.shm.slice_at_byte_offset(block_off, block_elts);
            block_off += block_elts * 4;
            dst.iter_mut().for_each(|v| *v = 1.0 as f32);
            let len = std::cmp::min(mask.len(), block_elts);
            dst[0..len].copy_from_slice(&mask[0..len]);
            fork_map.push(*op_idx);
        }

        Ok(AiciPreProcessResp {
            seqs: outputs,
            fork_map,
            suspend_ids,
        })
    }

    fn aici_mid_process(&mut self, req: AiciMidProcessReq) -> Result<AiciMidProcessResp> {
        let block_elts = self.globals.tokrx_info.vocab_size as usize;
        let mut outputs = HashMap::new();

        // first, execute forks
        let mut parents = HashMap::new();
        let mut child_lists = HashMap::new();

        for op in req.ops.iter() {
            let id = op.id;
            match self.maybe_fork(id, op.clone_id) {
                Ok(parent_id) => {
                    child_lists
                        .entry(parent_id)
                        .or_insert_with(Vec::new)
                        .push(id);
                    parents.insert(id, parent_id);
                }
                Err(e) => {
                    self.worker_error(id, &mut outputs, e);
                }
            }
        }

        let num_seqs = req.ops.len();
        let logit_size = block_elts * 4;

        ensure!(
            self.limits.logit_memory_bytes > num_seqs * logit_size,
            "shm size too small"
        );

        let mut logit_offset = 0;
        let mut used_ids = Vec::new();

        // initialize shm
        let slice = self
            .shm
            .slice_at_byte_offset::<f32>(0, num_seqs * block_elts);
        slice.iter_mut().for_each(|v| *v = 0.0);

        for op in req.ops.into_iter() {
            let instid = op.id;
            if let Ok(h) = self.get_worker(instid) {
                let par = *parents.get(&instid).unwrap();
                let fork_group = child_lists
                    .get(&par)
                    .unwrap()
                    .iter()
                    .map(|id| SeqId(*id as u32))
                    .collect::<Vec<_>>();
                let op = RtMidProcessArg {
                    op: MidProcessArg { fork_group },
                    logit_offset,
                    logit_size,
                };
                match h.start_process(op) {
                    Ok(_) => used_ids.push((logit_offset, instid)),
                    Err(e) => self.worker_error(instid, &mut outputs, e),
                };
            } else {
                log::info!("invalid id {}", instid);
            }
            logit_offset += logit_size;
        }

        let deadline = Instant::now() + std::time::Duration::from_millis(self.limits.max_step_ms);

        for (off, id) in used_ids {
            let h = self.get_worker(id).unwrap();
            let timeout = deadline.saturating_duration_since(Instant::now());
            match h.check_process(timeout) {
                Ok(data) => {
                    outputs.insert(id, data);
                    if log::log_enabled!(log::Level::Debug) {
                        let slice = self.logit_bias_at_byte_offset(off);
                        let allow_set = slice
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, v)| if *v >= 0.0 { Some(idx) } else { None })
                            .collect::<Vec<_>>();
                        let list = if allow_set.len() > 10000 {
                            "...".to_string()
                        } else {
                            allow_set
                                .iter()
                                .take(50)
                                .map(|idx| self.token_name(*idx))
                                .collect::<Vec<_>>()
                                .join(", ")
                        };
                        log::trace!("logits: {} allow; tokens: {}", allow_set.len(), list);
                    }
                }
                Err(e) => self.worker_error(id, &mut outputs, e),
            }
        }

        Ok(AiciMidProcessResp {
            seqs: outputs,
            num_seqs,
        })
    }

    fn aici_post_process(&mut self, req: AiciPostProcessReq) -> Result<AiciPostProcessResp> {
        // this if forking due to n= parameter in sampling
        // in general, we want to avoid that and instead use forking in the program,
        // as it is executed with a long time limit
        for op in req.ops.iter() {
            self.maybe_fork(op.id, op.clone_id)?;
        }

        let mut used_ids = Vec::new();
        let mut outputs = HashMap::new();

        for op in req.ops.into_iter() {
            let instid = op.id;
            if let Ok(h) = self.get_worker(instid) {
                let tokens = op.tokens;
                let op = RtPostProcessArg {
                    op: PostProcessArg {
                        tokens,
                        backtrack: op.backtrack.saturating_sub(1),
                    },
                };
                match h.start_post_process(op) {
                    Ok(_) => used_ids.push(instid),
                    Err(e) => self.worker_error(instid, &mut outputs, e),
                };
            } else {
                log::warn!("invalid id {}", instid);
            }
        }

        let deadline =
            Instant::now() + std::time::Duration::from_millis(self.limits.max_pre_step_ms);

        for id in used_ids {
            let h = self.get_worker(id).unwrap();
            let timeout = deadline.saturating_duration_since(Instant::now());
            match h.check_post_process(timeout) {
                Ok(data) => {
                    outputs.insert(id, data);
                }
                Err(e) => self.worker_error(id, &mut outputs, e),
            }
        }

        Ok(AiciPostProcessResp { seqs: outputs })
    }

    fn logit_bias_at_byte_offset(&self, off: usize) -> &'static mut [f32] {
        self.shm
            .slice_at_byte_offset(off, self.globals.tokrx_info.vocab_size as usize)
    }

    fn worker_error<T>(
        &mut self,
        instid: usize,
        map: &mut HashMap<usize, SequenceResult<T>>,
        e: anyhow::Error,
    ) {
        let err = format!("Worker: {e:?}");
        log::warn!("error: {err}");
        map.insert(instid, SequenceResult::from_error(err));
        self.instances.remove(&instid);
    }
}

impl Exec for Stepper {
    #[inline(never)]
    fn exec(&mut self, json: Value) -> Result<Value> {
        match json["op"].as_str() {
            Some("tokens") => Ok(json!({ "vocab_size": self.globals.tokrx_info.vocab_size })),
            Some("pre_process") => {
                let json = serde_json::from_value(json)?;
                with_timer!(self.pre_timer, {
                    Ok(serde_json::to_value(&self.aici_pre_process(false, json)?)?)
                })
            }
            Some("pre_process_all") => {
                let json = serde_json::from_value(json)?;
                with_timer!(self.pre_timer, {
                    Ok(serde_json::to_value(&self.aici_pre_process(true, json)?)?)
                })
            }
            Some("mid_process") => Ok(serde_json::to_value(
                &self.aici_mid_process(serde_json::from_value(json)?)?,
            )?),
            Some("post_process") => Ok(serde_json::to_value(
                &self.aici_post_process(serde_json::from_value(json)?)?,
            )?),
            _ => return Err(anyhow!("bad op")),
        }
    }
}

impl Exec for ModuleRegistry {
    #[inline(never)]
    fn exec(&mut self, json: Value) -> Result<Value> {
        match json["op"].as_str() {
            Some("mk_module") => self.mk_module(serde_json::from_value(json)?),
            Some("instantiate") => self.instantiate(serde_json::from_value(json)?),
            _ => return Err(anyhow!("bad op")),
        }
    }
}

trait Exec {
    fn exec(&mut self, json: Value) -> Result<Value>;

    fn exec_wrapped(&mut self, msg: &[u8]) -> Value {
        match serde_json::from_slice::<Value>(msg) {
            Ok(json) => {
                let rid = json["$rid"].as_str().map(|v| v.to_string());
                log::trace!("dispatch: rid={:?} op={:?}", rid, json["op"]);
                let val = match json["op"].as_str() {
                    Some("ping") => Ok(json!({ "pong": 1 })),
                    Some("stop") => worker::stop_process(),
                    _ => self.exec(json),
                };
                let mut resp = match val {
                    Ok(v) => {
                        log::trace!(
                            "dispatch ok: {}",
                            limit_str(&serde_json::to_string(&v).unwrap(), 200)
                        );
                        json!({
                            "type": "ok",
                            "data": v
                        })
                    }
                    Err(err) => {
                        let err = format!("{:?}", err);
                        log::warn!("dispatch error: {}", err);
                        log::info!(
                            "for data: {}",
                            String::from_utf8_lossy(&msg[0..std::cmp::min(100, msg.len())])
                        );
                        json!({
                            "type": "error",
                            "error": err
                        })
                    }
                };
                match rid {
                    Some(rid) => {
                        resp["$rid"] = Value::String(rid);
                        resp
                    }
                    None => resp,
                }
            }
            Err(err) => {
                let err = format!("{:?}", err);
                json!({
                    "type": "json-error",
                    "error": err,
                })
            }
        }
    }
}

struct CmdRespChannel {
    cmd_ch: MessageChannel,
    resp_ch: Arc<Mutex<MessageChannel>>,
    busy_wait_duration: Duration,
}

impl CmdRespChannel {
    pub fn new(suff: &str, cli: &Cli) -> Result<Self> {
        let cmd_ch =
            MessageChannel::new(&cli.prefixed_name("cmd", suff), cli.json_size * MEGABYTE)?;
        let resp_ch = Arc::new(Mutex::new(MessageChannel::new(
            &cli.prefixed_name("resp", suff),
            cli.json_size * MEGABYTE,
        )?));

        Ok(Self {
            cmd_ch,
            resp_ch,
            busy_wait_duration: Duration::from_millis(cli.busy_wait_time),
        })
    }

    #[allow(dead_code)]
    pub fn busy_reset(&self) {
        self.cmd_ch.busy_reset();
        self.resp_ch.lock().unwrap().busy_reset();
    }

    pub fn respond(&self, json: Value) {
        self.resp_ch
            .lock()
            .unwrap()
            .send(serde_json::to_vec(&json).unwrap().as_slice())
            .unwrap();
    }

    pub fn recv(&self) -> Vec<u8> {
        self.cmd_ch.recv(&self.busy_wait_duration).unwrap()
    }

    pub fn dispatch_loop(&self, mut exec: impl Exec) -> ! {
        loop {
            let msg = self.recv();
            let val = exec.exec_wrapped(&msg);
            self.respond(val)
        }
    }
}

fn bench_cmd_resp_busy(cli: &Cli, limits: &AiciLimits) {
    match fork_child::<u8, u8>(limits).unwrap() {
        worker::ForkResult::Parent { handle } => {
            let ch = CmdRespChannel::new("", cli).unwrap();
            ch.busy_reset();
            let resp_ch = ch.resp_ch.lock().unwrap();
            let timers = TimerSet::new();
            let timer = timers.new_timer("cmd_resp_busy");
            let cnt = 100;
            for idx in 0..cnt {
                let q = (idx & 0xf0) as u8;
                let v = vec![q];
                let resp = timer.with(|| {
                    ch.cmd_ch.busy_send(&v).unwrap();
                    resp_ch.busy_recv().unwrap()
                });
                assert!(resp[0] == v[0] + 1, "failed at idx={} {}", idx, resp[0]);
                std::thread::sleep(Duration::from_millis(10));
            }
            println!("MessageChannel {}", timers);
            handle.kill();
        }
        worker::ForkResult::Child { .. } => {
            let ch = CmdRespChannel::new("", cli).unwrap();
            let resp_ch = ch.resp_ch.lock().unwrap();
            loop {
                let msg = ch.cmd_ch.busy_recv().unwrap();
                let resp = vec![msg[0] + 1, 12];
                for _ in 0..20 {
                    std::hint::black_box(());
                }
                resp_ch.busy_send(&resp).unwrap();
            }
        }
    }
}

fn bench_cmd_resp(cli: &Cli, limits: &AiciLimits) {
    let wait_time = limits.busy_wait_duration;
    match fork_child::<u8, u8>(limits).unwrap() {
        worker::ForkResult::Parent { handle } => {
            let ch = CmdRespChannel::new("", cli).unwrap();
            let resp_ch = ch.resp_ch.lock().unwrap();
            let timers = TimerSet::new();
            let timer = timers.new_timer("cmd_resp_sem");
            let cnt = 100;
            for idx in 0..cnt {
                let q = (idx & 0xf0) as u8;
                let v = vec![q];
                let resp = timer.with(|| {
                    ch.cmd_ch.send(&v).unwrap();
                    resp_ch.recv(&wait_time).unwrap()
                });
                assert!(resp[0] == v[0] + 1, "failed at idx={} {}", idx, resp[0]);
                std::thread::sleep(Duration::from_millis(10));
            }
            println!("MessageChannel {}", timers);
            handle.kill();
        }
        worker::ForkResult::Child { .. } => {
            let ch = CmdRespChannel::new("", cli).unwrap();
            let resp_ch = ch.resp_ch.lock().unwrap();
            loop {
                let msg = ch.cmd_ch.recv(&wait_time).unwrap();
                let resp = vec![msg[0] + 1, 12];
                resp_ch.send(&resp).unwrap();
            }
        }
    }
}

fn set_priority(pri: ThreadPriority) {
    set_thread_priority_and_policy(
        thread_native_id(),
        pri,
        ThreadSchedulePolicy::Realtime(RealtimeThreadSchedulePolicy::Fifo),
    )
    .unwrap();
}

fn save_tokenizer(cli: &Cli) {
    let filename = cli.save_tokenizer.as_deref().unwrap();
    let tokenizer = find_tokenizer(&cli.tokenizer).unwrap();
    let tokens = tokenizer.token_bytes();

    let trie = TokTrie::from(&tokenizer.tokrx_info(), &tokens);
    trie.check_against(&tokens);

    let bytes = trie.serialize();

    // validate
    let trie2 = TokTrie::from_bytes(&bytes);
    assert!(trie.info() == trie2.info());
    trie2.check_against(&tokens);

    std::fs::write(filename, &bytes).unwrap();
    println!("wrote {}, {} bytes", filename, bytes.len());
}

fn install_from_cmdline(cli: &Cli, wasm_ctx: WasmContext, shm: Shm) {
    let name = cli.module.as_deref().unwrap();
    let mut reg = ModuleRegistry::new(wasm_ctx, shm).unwrap();
    let module_id = if name.len() == 64 && name.chars().all(|c| c.is_digit(16)) {
        name.to_string()
    } else {
        let wasm_bytes = fs::read(name).unwrap();
        let meta_bytes = match cli.module_meta.as_deref() {
            Some(name) => fs::read(name).unwrap(),
            None => serde_json::to_vec(&Value::Null).unwrap(),
        };

        let json = reg.create_module(wasm_bytes, meta_bytes).unwrap();
        json["module_id"].as_str().unwrap().to_string()
    };

    println!("{}", module_id);

    if cli.run {
        let req_id = "main".to_string();
        let arg = match cli.run_arg {
            Some(ref path) => json!(fs::read_to_string(path).unwrap()),
            None => json!({"steps":[]}),
        };
        reg.instantiate(InstantiateReq {
            req_id: req_id.clone(),
            prompt: json!("Hello world!"),
            module_id: module_id.clone(),
            module_arg: arg,
        })
        .unwrap();
        reg.run_main(&req_id).unwrap();
    }

    worker::stop_process();
}

fn main() -> () {
    setup_log();

    let cli = Cli::parse();

    if !cli.name.starts_with("/") {
        eprintln!("--name must start with /");
        std::process::exit(1);
    }

    let limits = AiciLimits {
        max_memory_bytes: cli.wasm_max_memory * MEGABYTE,
        max_init_ms: cli.wasm_max_init_time,
        max_step_ms: cli.wasm_max_step_time,
        max_pre_step_ms: cli.wasm_max_pre_step_time,
        logit_memory_bytes: cli.bin_size * MEGABYTE,
        busy_wait_duration: Duration::from_millis(cli.busy_wait_time),
        max_forks: cli.wasm_max_forks,
    };

    if cli.bench {
        bench_ipc(&limits);
        bench_cmd_resp_busy(&cli, &limits);
        bench_cmd_resp(&cli, &limits);
        return ();
    }

    let tokenizer = find_tokenizer(&cli.tokenizer).unwrap();
    let token_bytes = tokenizer.token_bytes();
    let wasm_ctx = WasmContext::new(limits.clone(), tokenizer).unwrap();

    if cli.save_tokenizer.is_some() {
        save_tokenizer(&cli);
        return ();
    }

    let bin_shm = Shm::new(
        &MessageChannel::shm_name(&cli.prefixed_name("bin", "")),
        limits.logit_memory_bytes,
        false,
    )
    .unwrap();

    if cli.module.is_some() {
        install_from_cmdline(&cli, wasm_ctx, bin_shm);
        return ();
    }

    if !cli.server {
        println!("missing --server");
        std::process::exit(1);
    }

    let num_cores: usize = std::thread::available_parallelism().unwrap().into();
    let num_bg_threads = BG_THREADS_FRACTION * num_cores / 100;
    let num_step_threads = STEP_THREADS_FRACTION * num_cores / 100;

    log::info!(
        "rayon with {} bg and {} step workers ({} cores)",
        num_bg_threads, num_step_threads, num_cores
    );

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_bg_threads)
        .start_handler(|_| set_priority(ThreadPriority::Min))
        .build_global()
        .unwrap();

    let step_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_step_threads)
        .start_handler(|_| set_priority(ThreadPriority::Max))
        .build()
        .unwrap();

    let reg = ModuleRegistry::new(wasm_ctx, bin_shm).unwrap();
    let debug_shm = Shm::new(
        &MessageChannel::shm_name(&cli.prefixed_name("bin", "")),
        limits.logit_memory_bytes,
        false,
    )
    .unwrap();
    let exec = Stepper::new(&reg, limits, debug_shm, token_bytes).unwrap();
    let cli2 = cli.clone();
    rayon::spawn(move || {
        let reg_disp = CmdRespChannel::new("-side", &cli2).unwrap();
        reg.dispatch_loop(reg_disp);
    });

    set_priority(ThreadPriority::Max);
    step_pool.install(|| {
        let exec_disp = CmdRespChannel::new("", &cli).unwrap();
        exec_disp.dispatch_loop(exec);
    })
}
