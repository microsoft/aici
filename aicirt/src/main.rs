mod hostimpl;
mod moduleinstance;
mod msgchannel;
mod semaphore;
mod shm;

use aici_abi::toktree::TokTrie;
use aici_tokenizers::{find_tokenizer, Tokenizer};
use anyhow::{anyhow, ensure, Result};
use base64;
use base64::Engine as _;
use clap::Parser;
use hex;
use hostimpl::{GlobalInfo, ModuleData};
use log::{info, warn};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use wasmtime::{self, Config};

use crate::hostimpl::*;
use crate::moduleinstance::*;
use crate::msgchannel::MessageChannel;
use crate::shm::Shm;

const N_THREADS: usize = 10;

/// How often to check for timeout of WASM; should be between 1 and 10
const WASMTIME_EPOCH_MS: u64 = 1;

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

    /// Run main() from the module just added
    #[arg(short, long)]
    run: bool,

    /// Run with POSIX shared memory interface
    #[arg(short, long)]
    server: bool,

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

struct ModuleRegistry {
    cache_path: PathBuf,
    engine: wasmtime::Engine,
    linker: Arc<wasmtime::Linker<ModuleData>>,
    modules: HashMap<String, wasmtime::Module>,
    req_instances: Arc<Mutex<HashMap<String, ModuleInstance>>>,
    globals: Arc<RwLock<GlobalInfo>>,
    limits: AiciLimits,
}

struct Stepper {
    req_instances: Arc<Mutex<HashMap<String, ModuleInstance>>>,
    instances: HashMap<ModuleInstId, Arc<Mutex<ModuleInstance>>>,
    globals: Arc<RwLock<GlobalInfo>>,
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

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum AiciOp {
    Prompt {
        id: ModuleInstId,
        prompt: Vec<Token>,
        req_id: String,
    },
    Gen {
        id: ModuleInstId,
        gen: Token,
        clone_id: Option<ModuleInstId>,
    },
}

impl AiciOp {
    pub fn to_thread_op(self) -> ThreadOp {
        match self {
            AiciOp::Prompt { prompt, .. } => ThreadOp::Prompt { prompt },
            AiciOp::Gen { gen, .. } => ThreadOp::Gen { gen },
        }
    }
}

#[derive(Serialize, Deserialize)]
struct MkModuleReq {
    binary: String,
    #[serde(default = "mk_null")]
    meta: Value,
}

#[derive(Serialize, Deserialize)]
struct InstantiateReq {
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
    pub fn new(limits: AiciLimits, mut tokenizer: Tokenizer) -> Result<Self> {
        let mut cfg = Config::default();
        // these are defaults as of 13.0.0, but we specify them anyways for stability
        cfg.debug_info(false)
            .wasm_backtrace(true)
            .native_unwind_info(true)
            .consume_fuel(false)
            .max_wasm_stack(512 * 1024)
            .wasm_tail_call(false)
            .wasm_threads(false)
            .wasm_simd(true)
            .wasm_relaxed_simd(false)
            .wasm_bulk_memory(true)
            .wasm_multi_value(true)
            .wasm_memory64(false)
            .strategy(wasmtime::Strategy::Auto)
            .cranelift_nan_canonicalization(false)
            .parallel_compilation(true);

        // disable stuff we don't need
        cfg.wasm_backtrace_details(wasmtime::WasmBacktraceDetails::Disable)
            .wasm_reference_types(false);

        // compilation in Speed mode seems to be ~10% slower but the generated code is 20-30% faster
        cfg.cranelift_opt_level(wasmtime::OptLevel::Speed);

        cfg.epoch_interruption(true);

        let engine = wasmtime::Engine::new(&cfg)?;
        let linker = setup_linker(&engine)?;

        tokenizer.load();
        let tokens = tokenizer.token_bytes();
        let trie = TokTrie::from(&tokenizer.tokrx_info(), &tokens);
        trie.check_against(&tokens);
        let bytes = trie.serialize();
        // validate
        let trie2 = TokTrie::from_bytes(&bytes);
        assert!(trie.info() == trie2.info());
        trie2.check_against(&tokens);

        // let tok = tokenizers::Tokenizer::from_bytes(tokenizer.hf_bytes).unwrap();
        // let tokens = tok.encode("I am something", false).unwrap();
        // println!("tokens: {:?}", tokens);

        let globals = GlobalInfo {
            tokrx_info: tokenizer.tokrx_info(),
            trie_bytes: bytes,
            hf_tokenizer_bytes: tokenizer.hf_bytes,
        };

        {
            let engine = engine.clone();
            std::thread::spawn(move || {
                let period = std::time::Duration::from_millis(WASMTIME_EPOCH_MS);
                loop {
                    std::thread::sleep(period);
                    engine.increment_epoch();
                }
            });
        }

        Ok(Self {
            cache_path: PathBuf::from("./cache"),
            engine,
            linker,
            modules: HashMap::new(),
            req_instances: Arc::new(Mutex::new(HashMap::new())),
            globals: Arc::new(RwLock::new(globals)),
            limits,
        })
    }

    fn create_module(&self, wasm_bytes: Vec<u8>, meta_bytes: Vec<u8>) -> Result<Value> {
        // make sure meta_bytes is valid JSON
        let _: Value = serde_json::from_slice(&meta_bytes)?;

        let mut hasher = Sha256::new();
        hasher.update(&meta_bytes);
        hasher.update(&wasm_bytes);

        let id = hex::encode(hasher.finalize());

        let filepath = self.cache_path.join(format!("{}.bin", id));
        let mut time = 0;
        let compiled_size = match fs::metadata(&filepath) {
            Ok(m) => m.len() as usize,
            Err(_) => {
                let timer = Instant::now();

                fs::create_dir_all(&self.cache_path)?;
                let compiled = self.engine.precompile_module(&wasm_bytes)?;
                let clen = compiled.len();
                fs::write(filepath, compiled)?;

                let jsonpath = self.cache_path.join(format!("{}.json", id));
                fs::write(jsonpath, &meta_bytes)?;

                let wasmpath = self.cache_path.join(format!("{}.wasm", id));
                fs::write(wasmpath, &wasm_bytes)?;

                time = timer.elapsed().as_millis();
                clen
            }
        };

        info!(
            "module {}: {}kB -> {}kB; {}ms",
            id,
            wasm_bytes.len() / 1024,
            compiled_size / 1024,
            time
        );

        Ok(json!({
            "module_id": id,
            "wasm_size": wasm_bytes.len(),
            "meta_size": meta_bytes.len(),
            "compiled_size": compiled_size,
            "time": time
        }))
    }

    fn mk_module(&self, req: MkModuleReq) -> Result<Value> {
        let wasm_bytes = base64::engine::general_purpose::STANDARD.decode(req.binary)?;
        let meta_bytes = serde_json::to_vec(&req.meta)?;
        self.create_module(wasm_bytes, meta_bytes)
    }

    fn instantiate(&mut self, req: InstantiateReq) -> Result<Value> {
        let arg = match req.module_arg.as_str() {
            Some(a) => a.to_string(),
            None => serde_json::to_string(&req.module_arg)?,
        };
        let modinst = self.new_instance(0x100000, req.module_id.as_str(), arg)?;
        let mut req_instances = self.req_instances.lock().unwrap();
        info!("instance {} -> {}", req.module_id, req.req_id);
        req_instances.insert(req.req_id, modinst);
        Ok(json!({}))
    }

    pub fn new_instance(
        &mut self,
        id: ModuleInstId,
        module_id: &str,
        module_arg: String,
    ) -> Result<ModuleInstance> {
        ensure!(is_hex_string(module_id), "invalid module_id");

        let module = match self.modules.get(module_id) {
            None => {
                let filepath = self.cache_path.join(format!("{}.bin", module_id));
                ensure!(filepath.exists(), "{} not found", module_id);
                let module = unsafe { wasmtime::Module::deserialize_file(&self.engine, filepath)? };
                self.modules.insert(String::from(module_id), module.clone());
                module
            }
            Some(v) => v.clone(),
        };

        let modinst = ModuleInstance::new(
            id,
            &self.limits,
            module,
            Arc::new(module_arg),
            self.linker.clone(),
            self.globals.clone(),
        )?;
        Ok(modinst)
    }
}

impl Stepper {
    pub fn new(reg: &ModuleRegistry, bin_shm: Shm) -> Result<Self> {
        Ok(Self {
            req_instances: reg.req_instances.clone(),
            instances: HashMap::new(),
            globals: reg.globals.clone(),
            bin_shm,
        })
    }

    fn mk_instance(&mut self, op: &AiciOp) -> Result<()> {
        // TODO the forks should be done in parallel, best in tree-like fashion
        match op {
            AiciOp::Gen { id, clone_id, .. } => {
                if let Some(cid) = clone_id {
                    ensure!(!self.instances.contains_key(id));
                    let parent = self
                        .instances
                        .get(cid)
                        .ok_or(anyhow!("invalid clone_id {}", cid))?;
                    info!("fork {} -> ({})", cid, id);
                    let copy = parent.lock().unwrap().fork(*id)?;
                    self.instances.insert(*id, Arc::new(Mutex::new(copy)));
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
                self.instances.insert(*id, Arc::new(Mutex::new(modinst)));
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

        let vocab_block_len = { self.globals.read().unwrap().tokrx_info.vocab_size * 4 } as usize;

        let mut slices = self.bin_shm.split(vocab_block_len)?;
        slices.reverse();

        let numops = req.ops.len();

        ensure!(slices.len() >= numops, "shm size too small");

        let mut ids = Vec::new();

        let reqs = req
            .ops
            .into_iter()
            .map(|op| -> Arc<Mutex<ModuleInstance>> {
                let instid = match op {
                    AiciOp::Gen { id, .. } => id,
                    AiciOp::Prompt { id, .. } => id,
                };
                ids.push(instid);
                let modinst_rc = self.instances.get(&instid).unwrap();
                let slice = slices.pop().unwrap();

                let mut lck = modinst_rc.lock();
                lck.as_deref_mut().unwrap().add_op(slice, op.to_thread_op());

                modinst_rc.clone()
            })
            .collect::<Vec<_>>();

        let results = reqs
            .into_par_iter()
            .map(|req| req.lock().as_deref_mut().unwrap().exec())
            .collect::<Vec<_>>();

        let mut map = serde_json::Map::new();
        for (id, result) in ids.into_iter().zip(results.into_iter()) {
            map.insert(id.to_string(), result);
        }

        Ok(Value::Object(map))
    }
}

impl Exec for Stepper {
    fn exec(&mut self, json: Value) -> Result<Value> {
        match json["op"].as_str() {
            Some("tokens") => {
                Ok(json!({ "vocab_size": self.globals.read().unwrap().tokrx_info.vocab_size }))
            }
            Some("step") => self.aici_step(serde_json::from_value(json)?),
            _ => return Err(anyhow!("bad op")),
        }
    }
}

impl Exec for ModuleRegistry {
    fn exec(&mut self, json: Value) -> Result<Value> {
        match json["op"].as_str() {
            Some("tokens") => {
                Ok(json!({ "vocab_size": self.globals.read().unwrap().tokrx_info.vocab_size }))
            }
            Some("mk_module") => self.mk_module(serde_json::from_value(json)?),
            Some("instantiate") => self.instantiate(serde_json::from_value(json)?),
            _ => return Err(anyhow!("bad op")),
        }
    }
}

trait BgExec {
    fn bg_exec(&mut self, json: Value) -> Result<Value>;
    fn bg_exec_wrapped(&mut self, msg: &[u8], resp: Box<dyn Fn(&[u8]) -> ()>) {

    }

        fn outer_exec(&mut self, msg: &[u8], resp: Box<dyn Fn(&[u8]) -> ()>) {
        match serde_json::from_slice::<Value>(msg) {
            Ok(json) => {
                let val = match json["op"].as_str() {
                    Some("ping") => Ok(json!({ "pong": 1 })),
                    Some("stop") => std::process::exit(0),
                    _ => self.exec(json),
                };
                match val {
                    Ok(v) => json!({
                        "type": "ok",
                        "data": v
                    }),
                    Err(err) => {
                        let err = format!("{:?}", err);
                        warn!("dispatch error: {}", err);
                        json!({
                            "type": "error",
                            "error": err
                        })
                    }
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

trait Exec {
    fn exec(&mut self, json: Value) -> Result<Value>;

    fn exec_wrapped(&mut self, msg: &[u8]) -> Value {
        match serde_json::from_slice::<Value>(msg) {
            Ok(json) => {
                let val = match json["op"].as_str() {
                    Some("ping") => Ok(json!({ "pong": 1 })),
                    Some("stop") => std::process::exit(0),
                    _ => self.exec(json),
                };
                match val {
                    Ok(v) => json!({
                        "type": "ok",
                        "data": v
                    }),
                    Err(err) => {
                        let err = format!("{:?}", err);
                        warn!("dispatch error: {}", err);
                        json!({
                            "type": "error",
                            "error": err
                        })
                    }
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

struct Dispatcher<T: Exec> {
    cmd_ch: MessageChannel,
    resp_ch: MessageChannel,
    executor: T,
}

impl<T: Exec> Dispatcher<T> {
    pub fn new(executor: T, suff: &str, cli: &Cli) -> Result<Self> {
        let cmd_ch =
            MessageChannel::new(&cli.prefixed_name("cmd", suff), cli.json_size * MEGABYTE)?;
        let resp_ch =
            MessageChannel::new(&cli.prefixed_name("resp", suff), cli.json_size * MEGABYTE)?;

        Ok(Self {
            cmd_ch,
            resp_ch,
            executor,
        })
    }

    fn respond(&self, json: Value) -> Result<()> {
        self.resp_ch.send(serde_json::to_vec(&json)?.as_slice())?;
        Ok(())
    }

    pub fn dispatch_loop(&mut self) -> ! {
        loop {
            let msg = self.cmd_ch.recv().unwrap();
            let resp = self.executor.exec_wrapped(&msg);
            self.respond(resp).unwrap();
        }
    }
}

struct BgDispatcher<T: BgExec> {
    cmd_ch: MessageChannel,
    resp_ch: Arc<Mutex<MessageChannel>>,
    executor: T,
}

impl<T: BgExec> BgDispatcher<T> {
    pub fn new(executor: T, suff: &str, cli: &Cli) -> Result<Self> {
        let cmd_ch =
            MessageChannel::new(&cli.prefixed_name("cmd", suff), cli.json_size * MEGABYTE)?;
        let resp_ch = Arc::new(Mutex::new(MessageChannel::new(
            &cli.prefixed_name("resp", suff),
            cli.json_size * MEGABYTE,
        )?));

        Ok(Self {
            cmd_ch,
            resp_ch,
            executor,
        })
    }

    pub fn dispatch_loop(&mut self) -> ! {
        loop {
            let msg = self.cmd_ch.recv().unwrap();
            let resp_lck = self.resp_ch.clone();
            self.executor.outer_exec(
                &msg,
                Box::new(move |resp| resp_lck.lock().unwrap().send(resp).unwrap()),
            )
        }
    }
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

    // You can check the value provided by positional arguments, or option arguments
    if let Some(name) = cli.module.as_deref() {
        let mut reg = ModuleRegistry::new(limits, find_tokenizer(&cli.tokenizer).unwrap()).unwrap();
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
            let mut modinst = reg.new_instance(42, &module_id, "{}".to_string()).unwrap();
            modinst.run_main().unwrap();
        }

        return ();
    }

    if !cli.server {
        println!("missing --server");
        std::process::exit(1);
    }

    info!("rayon with {} workers", N_THREADS);

    rayon::ThreadPoolBuilder::new()
        .num_threads(N_THREADS)
        .build_global()
        .unwrap();

    let bin_shm = Shm::new(
        &MessageChannel::shm_name(&cli.prefixed_name("bin", "")),
        cli.bin_size * MEGABYTE,
    )
    .unwrap();
    let reg = ModuleRegistry::new(limits, find_tokenizer(&cli.tokenizer).unwrap()).unwrap();
    let exec = Stepper::new(&reg, bin_shm).unwrap();
    let cli2 = cli.clone();
    std::thread::spawn(move || {
        let mut reg_disp = Dispatcher::new(reg, "-side", &cli2).unwrap();
        reg_disp.dispatch_loop();
    });

    let mut exec_disp = Dispatcher::new(exec, "", &cli).unwrap();
    exec_disp.dispatch_loop();
}
