mod hostimpl;
mod moduleinstance;
mod msgchannel;
mod semaphore;
mod shm;
mod worker;

use aici_abi::bytes::limit_str;
use aici_abi::toktree::TokTrie;
use aici_abi::{PreProcessArg, TokenId};
use aici_tokenizers::find_tokenizer;
use anyhow::{anyhow, ensure, Result};
use base64;
use base64::Engine as _;
use clap::Parser;
use hex;
use hostimpl::GlobalInfo;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use thread_priority::*;
use worker::{fork_child, SeqWorkerHandle};

use crate::hostimpl::*;
use crate::moduleinstance::*;
use crate::msgchannel::MessageChannel;
use crate::shm::Shm;
use crate::worker::{bench_ipc, ExecOp, WorkerForker};

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

    /// Maximum size of WASM module memory in megabytes
    #[arg(long, default_value = "64")]
    wasm_max_memory: usize,

    /// Maximum time WASM module can execute step in milliseconds
    #[arg(long, default_value = "50")]
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
}

fn is_hex_string(s: &str) -> bool {
    s.chars().all(|c| c.is_digit(16))
}

#[derive(Serialize, Deserialize)]
struct AiciPreProcessReq {
    max_context_len: usize, // in tokens
    freed: Vec<ModuleInstId>,
    ops: Vec<AiciOp>,
}

#[derive(Serialize, Deserialize)]
struct AiciProcessReq {
    ops: Vec<AiciOp>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum AiciOp {
    Prompt {
        id: ModuleInstId,
        req_id: String,
    },
    Gen {
        id: ModuleInstId,
        tokens: Vec<Token>,
        clone_id: Option<ModuleInstId>,
    },
}

impl AiciOp {
    pub fn to_thread_op(self) -> PreProcessArg {
        match self {
            AiciOp::Prompt { .. } => PreProcessArg { tokens: vec![] },
            AiciOp::Gen { tokens, .. } => PreProcessArg { tokens },
        }
    }
}

#[derive(Serialize, Deserialize)]
struct MkModuleReq {
    binary: String,
    #[serde(default = "mk_null")]
    meta: Value,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct InstantiateReq {
    req_id: String,
    // [TokenId] or str
    prompt: Value,
    module_id: String,
    #[serde(default = "mk_null")]
    module_arg: Value,
}

fn mk_null() -> Value {
    Value::Null
}

type Token = TokenId;

#[derive(Serialize, Deserialize)]
struct SpecialTokenIds {
    pub bos: Option<Token>,
    pub eos: Option<Token>,
    pub unk: Option<Token>,
    pub sep: Option<Token>,
    pub pad: Option<Token>,
    pub cls: Option<Token>,
}

#[derive(Serialize, Deserialize)]
struct TokensReq {
    tokens: Vec<String>,
    special: SpecialTokenIds,
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
                info!("compiling {}; {}", module_id, e);
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
        let time = timer.elapsed().as_millis();

        info!(
            "module {}: {}kB -> {}kB; {}ms",
            module_id,
            wasm_bytes.len() / 1024,
            compiled_size / 1024,
            time
        );

        Ok(json!({
            "module_id": module_id,
            "wasm_size": wasm_bytes.len(),
            "meta_size": meta_bytes.len(),
            "compiled_size": compiled_size,
            "time": time
        }))
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
        info!("instance {} -> {}", req.module_id, req.req_id);
        let handle = self
            .forker
            .lock()
            .unwrap()
            .instantiate(req.clone(), module_path)?;
        let mut req_instances = self.req_instances.lock().unwrap();
        req_instances.insert(req.req_id, handle);
        Ok(json!({}))
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

    fn mk_instance(&mut self, op: &AiciOp) -> Result<()> {
        // TODO the forks should be done in parallel, best in tree-like fashion
        match op {
            AiciOp::Gen { id, clone_id, .. } => {
                if let Some(cid) = clone_id {
                    ensure!(!self.instances.contains_key(id));
                    let parent = self.get_worker(*cid)?;
                    info!("fork {} -> ({})", cid, id);
                    let h = parent.fork(*cid)?;
                    self.instances.insert(*id, h);
                } else {
                    // make sure worker exists
                    self.get_worker(*id)?;
                }
            }
            AiciOp::Prompt { id, req_id, .. } => {
                let e = { self.req_instances.lock().unwrap().remove(req_id) };
                ensure!(e.is_some(), format!("invalid req_id {}", req_id));
                ensure!(
                    !self.instances.contains_key(id),
                    format!("duplicate id {}", id)
                );
                let h = e.unwrap();
                info!("prompt {} ({})", id, req_id);
                h.set_id(*id)?;
                self.instances.insert(*id, h);
            }
        };

        Ok(())
    }

    fn aici_step(&mut self, ops: Vec<AiciOp>, block_elts: usize, is_pre: bool) -> Result<Value> {
        let numops = ops.len();
        let block_bytes = block_elts * 4;

        ensure!(
            self.limits.logit_memory_bytes > numops * block_bytes,
            "shm size too small"
        );

        let mut logit_offset = 0;
        let mut used_ids = Vec::new();
        let mut map = serde_json::Map::new();

        // initialize shm
        let slice = self.shm.slice_at_byte_offset::<f32>(0, numops * block_elts);
        let init_v = if is_pre { 1.0 } else { 0.0 };
        slice.iter_mut().for_each(|v| *v = init_v);

        for op in ops.into_iter() {
            let instid = match op {
                AiciOp::Gen { id, .. } => id,
                AiciOp::Prompt { id, .. } => id,
            };
            if let Ok(h) = self.get_worker(instid) {
                let op = op.to_thread_op();
                let op = ExecOp {
                    op,
                    logit_offset,
                    logit_size: block_bytes,
                };
                let res = if is_pre {
                    h.start_pre_process(op)
                } else {
                    h.start_process(op)
                };
                match res {
                    Ok(_) => used_ids.push((logit_offset, instid)),
                    Err(e) => self.worker_error(instid, &mut map, e),
                };
            } else {
                warn!("invalid id {}", instid);
            }
            logit_offset += block_bytes;
        }

        let deadline = Instant::now() + std::time::Duration::from_millis(self.limits.max_step_ms);

        for (off, id) in used_ids {
            let h = self.get_worker(id).unwrap();
            let timeout = deadline.saturating_duration_since(Instant::now());
            match h.check_exec(timeout) {
                Ok(json) => {
                    map.insert(id.to_string(), json);
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
                                .take(40)
                                .map(|idx| self.token_name(*idx))
                                .collect::<Vec<_>>()
                                .join(", ")
                        };
                        debug!("logits: {} allow; tokens: {}", allow_set.len(), list);
                    }
                }
                Err(e) => self.worker_error(id, &mut map, e),
            }
        }

        Ok(Value::Object(map))
    }

    fn aici_pre_process(&mut self, req: AiciPreProcessReq) -> Result<Value> {
        for id in req.freed {
            info!("free module {}", id);
            self.instances.remove(&id);
        }

        // first, start instances and link clones
        for op in req.ops.iter() {
            self.mk_instance(&op)?
        }

        self.aici_step(req.ops, req.max_context_len, true)
    }

    fn aici_process(&mut self, req: AiciProcessReq) -> Result<Value> {
        let vocab_block_len = self.globals.tokrx_info.vocab_size as usize;
        self.aici_step(req.ops, vocab_block_len, false)
    }

    fn logit_bias_at_byte_offset(&self, off: usize) -> &'static mut [f32] {
        self.shm
            .slice_at_byte_offset(off, self.globals.tokrx_info.vocab_size as usize)
    }

    fn worker_error(
        &mut self,
        instid: usize,
        map: &mut serde_json::Map<String, Value>,
        e: anyhow::Error,
    ) {
        warn!("worker error: {e:?}");
        map.insert(instid.to_string(), json!({ "error": format!("{e:?}") }));
        self.instances.remove(&instid);
    }
}

impl Exec for Stepper {
    #[inline(never)]
    fn exec(&mut self, json: Value) -> Result<Value> {
        match json["op"].as_str() {
            Some("tokens") => Ok(json!({ "vocab_size": self.globals.tokrx_info.vocab_size })),
            Some("pre_process") => self.aici_pre_process(serde_json::from_value(json)?),
            Some("process") => self.aici_process(serde_json::from_value(json)?),
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
                debug!("dispatch: rid={:?} op={:?}", rid, json["op"]);
                let val = match json["op"].as_str() {
                    Some("ping") => Ok(json!({ "pong": 1 })),
                    Some("stop") => worker::stop_process(),
                    _ => self.exec(json),
                };
                let mut resp = match val {
                    Ok(v) => {
                        debug!(
                            "dispatch ok: {}",
                            limit_str(&serde_json::to_string(&v).unwrap(), 200)
                        );
                        json!({
                            "type": "ok",
                            "data": v
                        })
                    }
                    Err(err) => {
                        info!(
                            "data: {}",
                            String::from_utf8_lossy(&msg[0..std::cmp::min(100, msg.len())])
                        );
                        let err = format!("{:?}", err);
                        warn!("dispatch error: {}", err);
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
}

impl CmdRespChannel {
    pub fn new(suff: &str, cli: &Cli) -> Result<Self> {
        let cmd_ch =
            MessageChannel::new(&cli.prefixed_name("cmd", suff), cli.json_size * MEGABYTE)?;
        let resp_ch = Arc::new(Mutex::new(MessageChannel::new(
            &cli.prefixed_name("resp", suff),
            cli.json_size * MEGABYTE,
        )?));

        Ok(Self { cmd_ch, resp_ch })
    }

    pub fn respond(&self, json: Value) {
        self.resp_ch
            .lock()
            .unwrap()
            .send(serde_json::to_vec(&json).unwrap().as_slice())
            .unwrap();
    }

    pub fn recv(&self) -> Vec<u8> {
        self.cmd_ch.recv().unwrap()
    }

    pub fn dispatch_loop(&self, mut exec: impl Exec) -> ! {
        loop {
            let msg = self.recv();
            let val = exec.exec_wrapped(&msg);
            self.respond(val)
        }
    }
}

fn bench_cmd_resp(cli: &Cli) {
    match fork_child::<u8, u8>().unwrap() {
        worker::ForkResult::Parent { handle } => {
            let t0 = Instant::now();
            let ch = CmdRespChannel::new("", cli).unwrap();
            let resp_ch = ch.resp_ch.lock().unwrap();
            let cnt = 10_000;
            let mut sum0 = 0u64;
            let mut sum1 = 0u64;
            for idx in 0..cnt {
                let q = (idx & 0xf0) as u8;
                sum0 += q as u64 + 1;
                let v = vec![q];
                ch.cmd_ch.send(&v).unwrap();
                let resp = resp_ch.recv().unwrap();
                sum1 += resp[0] as u64;
            }
            assert!(sum0 == sum1);
            println!("MessageChannel {:?}", t0.elapsed() / cnt);
            handle.kill();
        }
        worker::ForkResult::Child { .. } => {
            let ch = CmdRespChannel::new("", cli).unwrap();
            let resp_ch = ch.resp_ch.lock().unwrap();
            loop {
                let mut msg = ch.recv();
                msg[0] += 1;
                resp_ch.send(&msg).unwrap();
            }
        }
    }
}

fn set_priority(pri: ThreadPriority) {
    set_thread_priority_and_policy(
        thread_native_id(),
        pri,
        ThreadSchedulePolicy::Realtime(RealtimeThreadSchedulePolicy::RoundRobin),
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
        reg.instantiate(InstantiateReq {
            req_id: req_id.clone(),
            prompt: json!(""),
            module_id: module_id.clone(),
            module_arg: json!({"steps":[]}),
        })
        .unwrap();
        reg.run_main(&req_id).unwrap();
    }

    worker::stop_process();
}

fn main() -> () {
    env_logger::init();

    let cli = Cli::parse();

    if !cli.name.starts_with("/") {
        eprintln!("--name must start with /");
        std::process::exit(1);
    }

    if cli.bench {
        bench_ipc();
        bench_cmd_resp(&cli);
        return ();
    }

    let limits = AiciLimits {
        max_memory_bytes: cli.wasm_max_memory * MEGABYTE,
        max_init_ms: cli.wasm_max_init_time,
        max_step_ms: cli.wasm_max_step_time,
        logit_memory_bytes: cli.bin_size * MEGABYTE,
    };

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

    info!(
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
