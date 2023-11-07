mod hostimpl;
mod moduleinstance;
mod msgchannel;
mod semaphore;
mod shm;
mod worker;

use aici_abi::bytes::limit_str;
use aici_abi::toktree::TokTrie;
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
use std::hash::Hash;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use thread_priority::*;
use worker::SeqGroupWorkerHandle;

use crate::hostimpl::*;
use crate::moduleinstance::*;
use crate::msgchannel::MessageChannel;
use crate::shm::Shm;
use crate::worker::WorkerForker;

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
    req_instances: Arc<Mutex<HashMap<String, SeqGroupWorkerHandle>>>,
    // note sure Mutex is needed
    forker: Arc<Mutex<WorkerForker>>,
}

struct Stepper {
    req_instances: Arc<Mutex<HashMap<String, SeqGroupWorkerHandle>>>,
    instances: HashMap<ModuleInstId, SeqGroupWorkerHandle>,
    top_workers: HashMap<ModuleInstId, ModuleInstId>,
    globals: GlobalInfo,
    bin_shm: Shm,
}

fn is_hex_string(s: &str) -> bool {
    s.chars().all(|c| c.is_digit(16))
}

#[derive(Serialize, Deserialize)]
struct AiciStepReq {
    freed: Vec<ModuleInstId>,
    ops: Vec<AiciOp>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AiciOp {
    Prompt {
        id: ModuleInstId,
        // the prompt normally comes from InstantiateReq
        // we currently ignore this one
        prompt: Option<Vec<Token>>,
        req_id: String,
    },
    Gen {
        id: ModuleInstId,
        tokens: Vec<Token>,
        clone_id: Option<ModuleInstId>,
    },
}

impl AiciOp {
    pub fn to_thread_op(self) -> ThreadOp {
        match self {
            AiciOp::Prompt { .. } => ThreadOp::Prompt {},
            AiciOp::Gen { tokens, .. } => ThreadOp::Gen { tokens },
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
    pub fn new(wasm_ctx: WasmContext) -> Result<Self> {
        let forker = WorkerForker::new(wasm_ctx.clone());

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
        let handle = self
            .forker
            .lock()
            .unwrap()
            .instantiate(req.clone(), module_path)?;
        info!("instance {} -> {}", req.module_id, req.req_id);
        let mut req_instances = self.req_instances.lock().unwrap();
        req_instances.insert(req.req_id, handle);
        Ok(json!({}))
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
    pub fn new(reg: &ModuleRegistry, bin_shm: Shm) -> Result<Self> {
        Ok(Self {
            req_instances: reg.req_instances.clone(),
            instances: HashMap::new(),
            top_workers: HashMap::new(),
            globals: reg.wasm_ctx.globals.clone(),
            bin_shm,
        })
    }

    fn top_worker(&self, id: ModuleInstId) -> Result<&SeqGroupWorkerHandle> {
        let id = self.top_worker_id(id)?;
        Ok(self.instances.get(&id).unwrap())
    }

    fn top_worker_id(&self, id: ModuleInstId) -> Result<ModuleInstId> {
        self.top_workers
            .get(&id)
            .map(|x| *x)
            .ok_or(anyhow!("invalid id {}", id))
    }

    fn mk_instance(&mut self, op: &AiciOp) -> Result<()> {
        // TODO the forks should be done in parallel, best in tree-like fashion
        match op {
            AiciOp::Gen { id, clone_id, .. } => {
                if let Some(cid) = clone_id {
                    ensure!(!self.top_workers.contains_key(id));
                    let parent = self.top_worker(*cid)?;
                    info!("fork {} -> ({})", cid, id);
                    parent.create_clone(*cid, *id);
                    self.top_workers.insert(*cid, self.top_worker_id(*id)?);
                }
            }
            AiciOp::Prompt { id, req_id, .. } => {
                let e = { self.req_instances.lock().unwrap().remove(req_id) };
                ensure!(e.is_some(), format!("invalid req_id {}", req_id));
                ensure!(
                    !self.instances.contains_key(id),
                    format!("duplicate id {}", id)
                );
                let modinst = e.unwrap();
                info!("prompt {} ({})", id, req_id);
                modinst.set_id(*id);
                self.instances.insert(*id, modinst);
            }
        };

        Ok(())
    }

    fn aici_step(&mut self, req: AiciStepReq) -> Result<Value> {
        for id in req.freed {
            info!("free module {}", id);
            let _ = self.instances.remove(&id);
        }

        // first, start instances and link clones
        for op in req.ops.iter() {
            self.mk_instance(&op)?
        }

        let vocab_block_len = self.globals.tokrx_info.vocab_size as usize * 4;

        let mut slices = self.bin_shm.split(vocab_block_len)?;
        slices.reverse();

        let numops = req.ops.len();

        ensure!(
            self.bin_shm.len() / vocab_block_len - 1 >= numops,
            "shm size too small"
        );

        let mut reqs = HashMap::new();
        let mut used_ids = Vec::new();
        let mut off = 0;

        for op in req.ops.into_iter() {
            let instid = match op {
                AiciOp::Gen { id, .. } => id,
                AiciOp::Prompt { id, .. } => id,
            };
            let topid = self.top_worker_id(instid).unwrap();
            if !reqs.contains_key(&topid) {
                used_ids.push(topid);
                reqs.insert(topid, Vec::new());
            }
            reqs.get_mut(&topid).unwrap().push((op, off));
            off += vocab_block_len;
        }

        for id in &used_ids {
            let h = self.top_worker(*id).unwrap();
            h.start_exec(reqs.remove(id).unwrap());
        }

        let mut map = serde_json::Map::new();
        for id in &used_ids {
            let h = self.top_worker(*id).unwrap();
            for (id, result) in h.finish_exec() {
                map.insert(id.to_string(), result);
            }
        }

        Ok(Value::Object(map))
    }
}

impl Exec for Stepper {
    fn exec(&mut self, json: Value) -> Result<Value> {
        match json["op"].as_str() {
            Some("tokens") => Ok(json!({ "vocab_size": self.globals.tokrx_info.vocab_size })),
            Some("step") => self.aici_step(serde_json::from_value(json)?),
            _ => return Err(anyhow!("bad op")),
        }
    }
}

impl Exec for ModuleRegistry {
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
                    Some("stop") => std::process::exit(0),
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

    std::fs::write(filename.clone(), &bytes).unwrap();
    println!("wrote {}, {} bytes", filename, bytes.len());
}

fn install_from_cmdline(cli: &Cli, wasm_ctx: WasmContext) {
    let name = cli.module.as_deref().unwrap();
    let reg = ModuleRegistry::new(wasm_ctx).unwrap();
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

    // if cli.run {
    //     let mut modinst = reg.new_instance(42, &module_id, "{}".to_string()).unwrap();
    //     modinst.run_main().unwrap();
    // }
}

fn main() -> () {
    env_logger::init();

    let cli = Cli::parse();

    if !cli.name.starts_with("/") {
        eprintln!("--name must start with /");
        std::process::exit(1);
    }

    let limits = AiciLimits {
        max_memory_bytes: cli.wasm_max_memory * MEGABYTE,
        max_init_epochs: (cli.wasm_max_init_time / WASMTIME_EPOCH_MS) + 1,
        max_step_epochs: (cli.wasm_max_step_time / WASMTIME_EPOCH_MS) + 1,
    };

    let wasm_ctx = WasmContext::new(limits, find_tokenizer(&cli.tokenizer).unwrap()).unwrap();

    if cli.save_tokenizer.is_some() {
        save_tokenizer(&cli);
        return ();
    }

    if cli.module.is_some() {
        install_from_cmdline(&cli, wasm_ctx);
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

    let bin_shm = Shm::new(
        &MessageChannel::shm_name(&cli.prefixed_name("bin", "")),
        cli.bin_size * MEGABYTE,
    )
    .unwrap();
    let reg = ModuleRegistry::new(wasm_ctx).unwrap();
    let exec = Stepper::new(&reg, bin_shm).unwrap();
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
