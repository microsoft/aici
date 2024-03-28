mod hostimpl;
mod moduleinstance;
mod worker;

use crate::{
    api::*,
    hostimpl::*,
    moduleinstance::*,
    msgchannel::MessageChannel,
    shm::Shm,
    worker::{RtMidProcessArg, WorkerForker}, TimerSet,
};
use aici_abi::{bytes::limit_str, toktree::TokTrie, MidProcessArg, SeqId};
use aicirt::{bintokens::find_tokenizer, futexshm::ServerChannel, *};
use anyhow::{anyhow, ensure, Result};
use base64::{self, Engine as _};
use clap::Parser;
use hex;
use hostimpl::GlobalInfo;
use regex::Regex;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    ops::Sub,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};
use worker::SeqWorkerHandle;

// percentage of available cores
const BG_THREADS_FRACTION: usize = 50;
// how much memory any process can allocate; doesn't apply to mmap()
// mostly applies to module compilation - 256M is just about enough for
// compiling 12M WASM module (RustPython)
const MAX_MALLOC: usize = 512 * MEGABYTE;

const MEGABYTE: usize = 1024 * 1024;

#[global_allocator]
pub static ALLOCATOR: cap::Cap<std::alloc::System> =
    cap::Cap::new(std::alloc::System, usize::max_value());

#[derive(Parser, Clone)]
struct Cli {
    /// Path to .wasm module to install
    #[arg(short, long)]
    module: Option<String>,

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

    /// Enable futex comms
    #[arg(long, default_value_t = false)]
    futex: bool,

    /// Size of JSON comm buffer in megabytes
    #[arg(long, default_value = "128")]
    json_size: usize,

    /// Size of binary comm buffer in megabytes
    #[arg(long, default_value = "64")]
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

    /// Maximum time WASM module can execute step in milliseconds
    #[arg(long, default_value = "150")]
    wasm_max_step_time: u64,

    /// Maximum time WASM module can execute initialization code in milliseconds
    #[arg(long, default_value = "1000")]
    wasm_max_init_time: u64,

    /// Resolution of timer exposed to WASM modules in microseconds; 0 to disable timer
    #[arg(long, default_value = "0")]
    wasm_timer_resolution_us: u64,

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

fn hex_hash_string(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s);
    hex::encode(hasher.finalize())
}

fn read_json(filename: &PathBuf) -> Result<Value> {
    let bytes = fs::read(filename)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn write_json<T: Serialize>(filename: &PathBuf, json: &T) -> Result<()> {
    fs::write(filename, serde_json::to_vec(json)?)?;
    Ok(())
}

impl ModuleRegistry {
    pub fn new(wasm_ctx: WasmContext, shm: Shm) -> Result<Self> {
        let forker = WorkerForker::new(wasm_ctx.clone(), shm);

        Ok(Self {
            forker: Arc::new(Mutex::new(forker)),
            cache_path: PathBuf::from("./cache"),
            wasm_ctx: Arc::new(wasm_ctx),
            modules: Arc::new(Mutex::new(HashMap::default())),
            req_instances: Arc::new(Mutex::new(HashMap::default())),
        })
    }

    fn module_needs_check(&self, module_id: &str) -> bool {
        loop {
            let mut lck = self.modules.lock().unwrap();
            match *lck.get(module_id).unwrap_or(&ModuleStatus::Missing) {
                ModuleStatus::Locked => {
                    drop(lck);
                    std::thread::sleep(std::time::Duration::from_millis(50))
                }
                ModuleStatus::Ready => return false,
                ModuleStatus::Missing => {
                    // we lock it
                    lck.insert(module_id.to_string(), ModuleStatus::Locked);
                    return true;
                }
            }
        }
    }

    fn sys_meta_path(&self, module_id: &str) -> PathBuf {
        self.cache_path.join(format!("{}-sys.json", module_id))
    }

    fn wasm_path(&self, module_id: &str) -> PathBuf {
        self.cache_path.join(format!("{}.wasm", module_id))
    }

    fn url_path(&self, url: &str) -> PathBuf {
        let hex = hex_hash_string(url);
        self.cache_path.join(format!("url-{}.json", hex))
    }

    fn elf_path(&self, module_id: &str) -> PathBuf {
        self.cache_path.join(format!("{}.elf", module_id))
    }

    fn tag_path(&self, tagname: &str) -> PathBuf {
        assert!(valid_tagname(tagname));
        self.cache_path.join(format!("tags/{}.json", tagname))
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
                let compiled = self.forker.lock().unwrap().compile(wasm_bytes)?;
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

    fn create_module(&self, wasm_bytes: Vec<u8>, auth: AuthInfo) -> Result<MkModuleResp> {
        let timer = Instant::now();

        let mut hasher = <Sha256 as Digest>::new();
        hasher.update(&wasm_bytes);

        let module_id = hex::encode(hasher.finalize());
        let module_id = &module_id;

        if self.module_needs_check(module_id) {
            match self.write_and_compile(module_id, &wasm_bytes, &auth) {
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

        Ok(MkModuleResp {
            module_id: module_id.to_string(),
            wasm_size: wasm_bytes.len(),
            compiled_size,
            time,
        })
    }

    fn write_and_compile(
        &self,
        module_id: &String,
        wasm_bytes: &Vec<u8>,
        auth: &AuthInfo,
    ) -> Result<()> {
        fs::create_dir_all(&self.cache_path)?;
        let meta = self.wasm_path(module_id).metadata();
        Ok(
            if meta.is_err() || meta.unwrap().len() != wasm_bytes.len() as u64 {
                fs::write(self.wasm_path(module_id), wasm_bytes)?;
                write_json(
                    &self.sys_meta_path(module_id),
                    &json!({
                        "created": get_unix_time(),
                        "auth": auth,
                    }),
                )?;
                self.compile_module(module_id, true)?
            } else {
                self.compile_module(module_id, false)?
            },
        )
    }

    fn mk_module(&self, req: MkModuleReq, auth: AuthInfo) -> Result<Value> {
        let wasm_bytes = base64::engine::general_purpose::STANDARD.decode(req.binary)?;
        Ok(serde_json::to_value(
            &self.create_module(wasm_bytes, auth)?,
        )?)
    }

    fn set_tags(&self, req: SetTagsReq, auth: AuthInfo) -> Result<Value> {
        ensure!(valid_module_id(&req.module_id), "invalid module_id");
        let _ = self.ensure_module_in_fs(&req.module_id)?;

        let user_pref = if auth.is_admin {
            // admins can do any prefix
            String::new()
        } else {
            // other users can only do myself.something
            auth.user.clone() + "."
        };

        for tagname in &req.tags {
            if tagname.len() > 50 {
                bail_user!("tag name too long");
            }
            if tagname.len() > 20 && is_hex_string(tagname) {
                bail_user!("tag name looks too hex")
            }
            if !valid_tagname(tagname) {
                bail_user!("tag name not identifier")
            }
            if !tagname.starts_with(&user_pref) {
                bail_user!("permission denied for tag name")
            }
        }

        fs::create_dir_all(&self.cache_path.join("tags"))?;

        let info = TagInfo {
            tag: String::new(),
            module_id: req.module_id.clone(),
            updated_at: get_unix_time(),
            updated_by: auth.user.clone(),
            wasm_size: self.wasm_path(&req.module_id).metadata()?.len(),
            compiled_size: self.elf_path(&req.module_id).metadata()?.len(),
        };

        let mut resp = GetTagsResp { tags: vec![] };
        for tagname in &req.tags {
            log::info!("tag {} -> {} by {}", tagname, req.module_id, auth.user);
            let mut info = info.clone();
            info.tag = tagname.clone();
            write_json(&self.tag_path(tagname), &info)?;
            resp.tags.push(info)
        }

        Ok(json!(resp))
    }

    fn read_tag(&self, tag_name: &str) -> Result<TagInfo> {
        let path = self.tag_path(tag_name);
        match fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(anyhow::Error::from),
            Err(_) => bail_user!("tag {tag_name} not found"),
        }
    }

    fn get_tags(&self, _req: Value) -> Result<Value> {
        let tagspath = self.cache_path.join("tags");
        fs::create_dir_all(&tagspath)?;
        let mut resp = GetTagsResp { tags: vec![] };
        for file in fs::read_dir(&tagspath)? {
            let file = file?.path();
            if file.to_string_lossy().ends_with(".json") {
                let bytes = fs::read(file)?;
                resp.tags.push(serde_json::from_slice(&bytes)?);
            }
        }
        resp.tags.sort_by_key(|e| e.updated_at);
        resp.tags.reverse();
        Ok(json!(resp))
    }

    fn resolve_gh_module(&self, module_id: &str) -> Result<String> {
        if !module_id.starts_with("gh:") {
            return Ok(module_id.to_string());
        }
        ensure_user!(
            Regex::new(r"^gh:[\./a-zA-Z0-9_-]+$")
                .unwrap()
                .is_match(module_id),
            "invalid gh: module_id"
        );
        let mut parts = module_id[3..]
            .split('/')
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        ensure_user!(
            2 <= parts.len() && parts.len() <= 4,
            "invalid gh: module_id (parts)"
        );
        let mut ver = "latest".to_string();
        let last_part = parts.last().unwrap();
        let mut selector = "".to_string();
        if parts.len() > 2
            && (last_part == "latest"
                || regex::Regex::new(r"^v\d+\.\d+")
                    .unwrap()
                    .is_match(last_part))
        {
            ver = format!("tags/{}", parts.pop().unwrap());
        }
        if parts.len() > 2 {
            selector = parts.pop().unwrap();
        }
        ensure_user!(parts.len() == 2, "invalid gh: module_id (parts2)");

        let url = format!(
            "https://api.github.com/repos/{}/{}/releases/{}",
            parts[0], parts[1], ver
        );
        let cache_path = self.url_path(&url);
        let meta = cache_path.metadata();
        if !(meta.is_ok()
            && meta.unwrap().modified()? > SystemTime::now().sub(Duration::from_secs(120)))
        {
            log::info!("fetching {} to {:?}", url, cache_path);
            let resp = ureq::get(&url)
                .set("User-Agent", "AICI")
                .set("Accept", "application/vnd.github+json")
                .set("X-GitHub-Api-Version", "2022-11-28")
                .call()
                .map_err(|e| user_error!("gh: fetch failed: {}", e))?;
            fs::create_dir_all(&self.cache_path)?;
            std::fs::write(cache_path.clone(), resp.into_string()?)?;
        }
        let release = read_json(&cache_path)?;
        let wasm_files = release["assets"]
            .as_array()
            .ok_or_else(|| anyhow!("no assets"))?
            .iter()
            .filter(|a| {
                a["name"]
                    .as_str()
                    .map(|s| s.ends_with(".wasm") && s.contains(&selector))
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();

        ensure_user!(
            wasm_files.len() > 0,
            "no wasm files found (selector={:?})",
            selector
        );
        ensure_user!(
            wasm_files.len() == 1,
            "too many wasm files found (selector={:?})",
            selector
        );

        let wasm_file = wasm_files[0];
        let upd = wasm_file["updated_at"]
            .as_str()
            .ok_or_else(|| anyhow!("no updated_at"))?;
        let wasm_url = wasm_file["browser_download_url"]
            .as_str()
            .ok_or_else(|| anyhow!("no browser_download_url"))?;
        let link_path = self.url_path(&format!("{}---{}", upd, wasm_url));
        if link_path.exists() {
            let link = read_json(&link_path)?;
            return Ok(link["module_id"]
                .as_str()
                .ok_or_else(|| anyhow!("invalid json"))?
                .to_string());
        }
        log::info!("downloading {}", wasm_url);
        let mut wasm_bytes = vec![];
        ureq::get(wasm_url)
            .set("User-Agent", "AICI")
            .call()
            .map_err(|e| anyhow!("gh: download failed: {}", e))?
            .into_reader()
            .read_to_end(&mut wasm_bytes)?;
        log::info!("downloaded {} bytes", wasm_bytes.len());
        let resp = self.create_module(
            wasm_bytes,
            AuthInfo {
                user: wasm_url.to_string(),
                is_admin: true,
            },
        )?;
        write_json(&link_path, &resp)?;
        Ok(resp.module_id)
    }

    fn instantiate(&mut self, mut req: InstantiateReq) -> Result<Value> {
        req.module_id = self.resolve_gh_module(&req.module_id)?;
        if valid_tagname(&req.module_id) {
            let taginfo = self.read_tag(&req.module_id)?;
            req.module_id = taginfo.module_id;
        }
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
        let inst = req_instances
            .get(req_id)
            .ok_or_else(|| anyhow!("invalid req_id"))?;
        inst.run_main()
    }

    pub fn dispatch_loop(&self, mut ch: CmdRespChannel) -> ! {
        loop {
            let msg = ch.recv();
            let mut s2 = self.clone();

            //println!("exec side: {}", &String::from_utf8_lossy(&msg));
            match &ch {
                CmdRespChannel::Futex { resp_ch, .. } => {
                    let resp_ch = resp_ch.clone();
                    rayon::spawn(move || {
                        let r = s2.exec_wrapped(&msg);
                        //println!("resp side: {}", serde_json::to_string(&r).unwrap());
                        resp_ch
                            .lock()
                            .unwrap()
                            .send_resp(serde_json::to_vec(&r).unwrap().as_slice())
                            .unwrap();
                    });
                }
                CmdRespChannel::Sem { resp_ch, .. } => {
                    let resp_ch = resp_ch.clone();
                    rayon::spawn(move || {
                        let r = s2.exec_wrapped(&msg);
                        resp_ch
                            .lock()
                            .unwrap()
                            .send(serde_json::to_vec(&r).unwrap().as_slice())
                            .unwrap();
                    });
                }
            }
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
            instances: HashMap::default(),
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
            .ok_or_else(|| anyhow!("invalid id {}", id))?)
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
    fn maybe_fork(&mut self, op: &AiciMidOp) -> Result<ModuleInstId> {
        self.mk_instance(&op)?;
        let id = op.id;
        if let Some(parent_id) = op.clone_id {
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

    fn mk_instance(&mut self, op: &AiciMidOp) -> Result<()> {
        if let Some(req_id) = &op.req_id {
            let req_id = req_id.clone();
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

    fn aici_mid_process(&mut self, req: AiciMidProcessReq) -> Result<AiciMidProcessResp> {
        let block_elts = self.globals.tokrx_info.vocab_size as usize;
        let mut outputs = HashMap::default();

        // first, execute forks
        let mut parents = HashMap::default();
        let mut child_lists = HashMap::default();

        for op in req.ops.iter() {
            assert!(op.clone_id.is_none() == op.clone_idx.is_none());
            assert!(op.req_id.is_none() || op.clone_id.is_none());

            let id = op.id;
            match self.maybe_fork(op) {
                Ok(parent_id) => {
                    let lst = child_lists.entry(parent_id).or_insert_with(Vec::new);
                    let idx = op.clone_idx.unwrap_or(0);
                    while lst.len() <= idx {
                        lst.push(0);
                    }
                    lst[idx] = id;
                    parents.insert(id, parent_id);
                }
                Err(e) => {
                    self.worker_error(id, &mut outputs, e);
                }
            }
        }

        for lst in child_lists.values() {
            assert!(lst.iter().all(|id| self.instances.contains_key(&id)));
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
                    op: MidProcessArg {
                        backtrack: op.backtrack,
                        tokens: op.tokens.clone(),
                        fork_group,
                    },
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

        for id in req.freed {
            log::debug!("free module {}", id);
            self.instances.remove(&id);
        }

        Ok(AiciMidProcessResp { seqs: outputs })
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
    fn exec(&mut self, json: Value, _auth: AuthInfo) -> Result<Value> {
        match json["op"].as_str() {
            Some("tokens") => Ok(json!({ "vocab_size": self.globals.tokrx_info.vocab_size })),
            Some("mid_process") => Ok(serde_json::to_value(
                &self.aici_mid_process(serde_json::from_value(json)?)?,
            )?),
            _ => return Err(anyhow!("bad op")),
        }
    }
}

impl Exec for ModuleRegistry {
    #[inline(never)]
    fn exec(&mut self, json: Value, auth: AuthInfo) -> Result<Value> {
        match json["op"].as_str() {
            Some("set_tags") => self.set_tags(serde_json::from_value(json)?, auth),
            Some("get_tags") => self.get_tags(serde_json::from_value(json)?),
            Some("mk_module") => self.mk_module(serde_json::from_value(json)?, auth),
            Some("instantiate") => self.instantiate(serde_json::from_value(json)?),
            _ => return Err(anyhow!("bad op")),
        }
    }
}

trait Exec {
    fn exec(&mut self, json: Value, auth: AuthInfo) -> Result<Value>;

    fn exec_wrapped(&mut self, msg: &[u8]) -> Value {
        match serde_json::from_slice::<Value>(msg) {
            Ok(json) => {
                let rid = json["$rid"].as_str().map(|v| v.to_string());

                log::trace!("dispatch: rid={:?} op={:?}", rid, json["op"]);
                let val = match json["op"].as_str() {
                    Some("ping") => Ok(json!({ "pong": 1 })),
                    Some("stop") => worker::stop_process(),
                    _ => {
                        let auth = if json["$auth"].as_object().is_none() {
                            Ok(AuthInfo::local_user())
                        } else {
                            serde_json::from_value(json["$auth"].clone())
                        };
                        match auth {
                            Err(e) => Err(anyhow!(e)),
                            Ok(auth) => self.exec(json, auth),
                        }
                    }
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
                        let errmsg = UserError::maybe_stacktrace(&err);
                        log::warn!("dispatch error: {}", errmsg);
                        log::info!(
                            "for data: {}",
                            String::from_utf8_lossy(&msg[0..std::cmp::min(100, msg.len())])
                        );
                        json!({
                            "type": "error",
                            "error": errmsg,
                            "is_user_error": UserError::is_self(&err)
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

enum CmdRespChannel {
    Sem {
        cmd_ch: MessageChannel,
        resp_ch: Arc<Mutex<MessageChannel>>,
        busy_wait_duration: Duration,
    },
    Futex {
        cmd_ch: ServerChannel,
        resp_ch: Arc<Mutex<ServerChannel>>,
        busy_wait_duration: Duration,
    },
}

impl CmdRespChannel {
    pub fn new(suff: &str, cli: &Cli) -> Result<Self> {
        let busy_wait_duration = Duration::from_millis(cli.busy_wait_time);
        if cli.futex {
            let cmd_shm = Shm::new(
                &cli.prefixed_name("cmd", suff),
                cli.json_size * MEGABYTE,
                shm::Unlink::Post,
            )?;
            let resp_shm = Shm::new(
                &cli.prefixed_name("resp", suff),
                cli.json_size * MEGABYTE,
                shm::Unlink::Post,
            )?;
            Ok(Self::Futex {
                cmd_ch: ServerChannel::new(cmd_shm),
                resp_ch: Arc::new(Mutex::new(ServerChannel::new(resp_shm))),
                busy_wait_duration,
            })
        } else {
            let cmd_ch =
                MessageChannel::new(&cli.prefixed_name("cmd", suff), cli.json_size * MEGABYTE)?;
            let resp_ch = Arc::new(Mutex::new(MessageChannel::new(
                &cli.prefixed_name("resp", suff),
                cli.json_size * MEGABYTE,
            )?));

            Ok(Self::Sem {
                cmd_ch,
                resp_ch,
                busy_wait_duration,
            })
        }
    }

    #[allow(dead_code)]
    pub fn busy_reset(&self) {
        match self {
            Self::Sem {
                cmd_ch, resp_ch, ..
            } => {
                cmd_ch.busy_reset();
                resp_ch.lock().unwrap().busy_reset();
            }
            Self::Futex { .. } => {}
        }
    }

    pub fn respond(&self, json: Value) {
        let slice = serde_json::to_vec(&json).unwrap();
        match self {
            Self::Sem { resp_ch, .. } => {
                resp_ch.lock().unwrap().send(&slice).unwrap();
            }
            Self::Futex { resp_ch, .. } => {
                resp_ch.lock().unwrap().send_resp(&slice).unwrap();
            }
        }
    }

    pub fn recv(&mut self) -> Vec<u8> {
        match self {
            Self::Sem {
                cmd_ch,
                busy_wait_duration,
                ..
            } => cmd_ch.recv(busy_wait_duration).unwrap(),
            Self::Futex {
                cmd_ch,
                busy_wait_duration,
                ..
            } => cmd_ch.recv_req(busy_wait_duration.clone()),
        }
    }

    pub fn dispatch_loop(&mut self, mut exec: impl Exec) -> ! {
        loop {
            let msg = self.recv();
            //println!("exec main: {}", String::from_utf8_lossy(&msg));
            let val = exec.exec_wrapped(&msg);
            //println!("resp main: {}", serde_json::to_string(&val).unwrap());
            self.respond(val)
        }
    }
}

fn bench_hashmap() {
    let mut h = HashMap::<u64, u64>::default();
    for x in 10..50 {
        h.insert(x, x * x);
    }
    for _ in 0..10 {
        let t0 = Instant::now();
        let mut sum = 0;
        for x in 10..50 {
            let v = h.get(&x).unwrap();
            sum += v;
        }
        println!("hashmap: {:?} {}", t0.elapsed(), sum);
    }
}

fn save_tokenizer(cli: &Cli) {
    let filename = cli.save_tokenizer.as_deref().unwrap();
    let tokenizer = find_tokenizer(&cli.tokenizer).unwrap();
    let tokens = tokenizer.token_bytes();

    log::info!(
        "TokTrie building: {:?} wl={}",
        tokenizer.tokrx_info(),
        tokens.len()
    );
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
    let module_id = if name.ends_with(".wasm") {
        let wasm_bytes = fs::read(name).unwrap();
        let json = reg
            .create_module(wasm_bytes, AuthInfo::local_user())
            .unwrap();
        json.module_id
    } else {
        name.to_string()
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
            prompt: json!(""),
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
        ipc_shm_bytes: cli.json_size * MEGABYTE,
        timer_resolution_ns: cli.wasm_timer_resolution_us * 1000,
        max_memory_bytes: cli.wasm_max_memory * MEGABYTE,
        max_init_ms: cli.wasm_max_init_time,
        max_step_ms: cli.wasm_max_step_time,
        max_compile_ms: 10_000,
        logit_memory_bytes: cli.bin_size * MEGABYTE,
        busy_wait_duration: Duration::from_millis(cli.busy_wait_time),
        max_forks: cli.wasm_max_forks,
    };

    if cli.bench {
        bench_hashmap();
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
        if cli.module.is_none() {
            shm::Unlink::None
        } else {
            shm::Unlink::Pre
        },
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

    ALLOCATOR.set_limit(MAX_MALLOC).expect("set memory limit");

    set_max_priority();

    let reg = ModuleRegistry::new(wasm_ctx, bin_shm).unwrap();

    // needs to be done after WorkerForker is spawned
    setup_bg_worker_pool();

    let debug_shm = Shm::new(
        &MessageChannel::shm_name(&cli.prefixed_name("bin", "")),
        limits.logit_memory_bytes,
        shm::Unlink::None,
    )
    .unwrap();
    let exec = Stepper::new(&reg, limits, debug_shm, token_bytes).unwrap();
    let cli2 = cli.clone();
    rayon::spawn(move || {
        let reg_disp = CmdRespChannel::new("-side", &cli2).unwrap();
        reg.dispatch_loop(reg_disp);
    });

    let mut exec_disp = CmdRespChannel::new("", &cli).unwrap();
    exec_disp.dispatch_loop(exec);
}

pub fn setup_bg_worker_pool() {
    let num_cores: usize = std::thread::available_parallelism().unwrap().into();
    let num_bg_threads = BG_THREADS_FRACTION * num_cores / 100;
    log::debug!(
        "rayon with {} bg workers ({} cores)",
        num_bg_threads,
        num_cores
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_bg_threads)
        .start_handler(|_| set_min_priority())
        .build_global()
        .unwrap();
}
