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
use wasmtime;

use crate::moduleinstance::*;
use crate::msgchannel::MessageChannel;
use crate::shm::Shm;

const N_THREADS: usize = 10;

#[derive(Parser)]
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

    /// Shm/semaphore name prefix
    #[arg(long, short, default_value = "/aici0-")]
    name: String,
}

impl Cli {
    pub fn with_prefix(&self, name: &str) -> String {
        format!("{}{}", self.name, name)
    }
}

struct Executor {
    cache_path: PathBuf,
    engine: wasmtime::Engine,
    linker: Arc<wasmtime::Linker<ModuleData>>,
    modules: HashMap<String, wasmtime::Module>,
    instances: HashMap<Id, Arc<Mutex<ModuleInstance>>>,
    globals: Arc<RwLock<GlobalInfo>>,
    bin_shm: Option<Shm>,
}

fn is_hex_string(s: &str) -> bool {
    s.chars().all(|c| c.is_digit(16))
}

#[derive(Serialize, Deserialize)]
struct AiciStepReq {
    freed: Vec<Id>,
    ops: Vec<AiciOp>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum AiciOp {
    Prompt {
        id: Id,
        prompt: Vec<Token>,
        module_id: String,
        module_arg: String,
    },
    Gen {
        id: Id,
        gen: Token,
        clone_id: Option<Id>,
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

impl Executor {
    pub fn new(bin_shm: Option<Shm>, mut tokenizer: Tokenizer) -> Result<Self> {
        let engine = wasmtime::Engine::default();
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

        Ok(Self {
            cache_path: PathBuf::from("./cache"),
            engine,
            linker,
            modules: HashMap::new(),
            instances: HashMap::new(),
            globals: Arc::new(RwLock::new(globals)),
            bin_shm,
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

                time = timer.elapsed().as_millis();
                clen
            }
        };

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

    pub fn new_instance(
        &mut self,
        id: Id,
        module_id: &str,
        module_arg: &str,
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
            module,
            Arc::new(module_arg.to_string()),
            self.linker.clone(),
            self.globals.clone(),
        )?;
        Ok(modinst)
    }

    fn mk_instance(&mut self, op: &AiciOp) -> Result<()> {
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
            AiciOp::Prompt {
                id,
                module_id,
                module_arg,
                ..
            } => {
                ensure!(!self.instances.contains_key(id));
                let modinst = self.new_instance(*id, module_id, module_arg)?;
                info!("new module {} ({})", id, module_id);
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

        let binresp_ch = self.bin_shm.as_ref().unwrap();
        let mut slices = binresp_ch.split(vocab_block_len)?;
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

struct Dispatcher {
    cmd_ch: MessageChannel,
    resp_ch: MessageChannel,
    executor: Executor,
}

impl Dispatcher {
    pub fn new(cli: &Cli) -> Result<Self> {
        const M: usize = 1024 * 1024;

        let cmd_ch = MessageChannel::new(&cli.with_prefix("cmd"), cli.json_size * M)?;
        let resp_ch = MessageChannel::new(&cli.with_prefix("resp"), cli.json_size * M)?;
        let bin_shm = Shm::new(
            &MessageChannel::shm_name(&cli.with_prefix("bin")),
            cli.bin_size * M,
        )?;

        Ok(Self {
            cmd_ch,
            resp_ch,
            executor: Executor::new(Some(bin_shm), find_tokenizer(&cli.tokenizer)?)?,
        })
    }

    fn respond(&self, json: Value) -> Result<()> {
        self.resp_ch.send(serde_json::to_vec(&json)?.as_slice())?;
        Ok(())
    }

    fn dispatch_one(&mut self, json: Value) -> Result<Value> {
        match json["op"].as_str() {
            Some("ping") => Ok(json!({ "pong": 1 })),
            Some("tokens") => Ok(
                json!({ "vocab_size": self.executor.globals.read().unwrap().tokrx_info.vocab_size }),
            ),
            Some("mk_module") => self.executor.mk_module(serde_json::from_value(json)?),
            Some("step") => self.executor.aici_step(serde_json::from_value(json)?),
            Some("stop") => std::process::exit(0),
            _ => return Err(anyhow!("bad op")),
        }
    }

    pub fn dispatch_loop(&mut self) -> ! {
        loop {
            let msg = self.cmd_ch.recv().unwrap();
            match serde_json::from_slice(msg.as_slice()) {
                Ok(json) => match self.dispatch_one(json) {
                    Ok(v) => self
                        .respond(json!({
                            "type": "ok",
                            "data": v
                        }))
                        .unwrap(),
                    Err(err) => {
                        warn!("dispatch error: {}", err.to_string());
                        self.respond(json!({
                            "type": "error",
                            "error": err.to_string()
                        }))
                        .unwrap()
                    }
                },
                Err(err) => self
                    .respond(json!({
                        "type": "json-error",
                        "error": err.to_string(),
                    }))
                    .unwrap(),
            }
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

    // You can check the value provided by positional arguments, or option arguments
    if let Some(name) = cli.module.as_deref() {
        let mut exec = Executor::new(None, find_tokenizer(&cli.tokenizer).unwrap()).unwrap();
        let module_id = if name.len() == 64 && name.chars().all(|c| c.is_digit(16)) {
            name.to_string()
        } else {
            let wasm_bytes = fs::read(name).unwrap();
            let meta_bytes = match cli.module_meta.as_deref() {
                Some(name) => fs::read(name).unwrap(),
                None => serde_json::to_vec(&Value::Null).unwrap(),
            };

            let json = exec.create_module(wasm_bytes, meta_bytes).unwrap();
            json["module_id"].as_str().unwrap().to_string()
        };

        println!("{}", module_id);

        if cli.run {
            let mut modinst = exec.new_instance(42, &module_id, "").unwrap();
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

    let mut dispatcher = Dispatcher::new(&cli).unwrap();
    dispatcher.dispatch_loop();
}
