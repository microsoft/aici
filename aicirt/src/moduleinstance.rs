use aici_abi::bytes::{clone_vec_as_bytes, TokRxInfo};
use anyhow::{anyhow, ensure, Result};
use log::{debug, info};
use serde_json::{json, Value};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokenizers::Tokenizer;
use wasmtime;

// this is avaiable to functions called from wasm
pub struct ModuleData {
    id: Id,
    log: Vec<u8>,
    printed_log: usize,
    globals: Arc<RwLock<GlobalInfo>>,
    module_arg: Arc<String>,
    linker: Arc<wasmtime::Linker<ModuleData>>,
    instance: Option<wasmtime::Instance>,
    memory: Option<wasmtime::Memory>,
    module: wasmtime::Module,
    tokenizer: Option<Tokenizer>,
}

pub struct ModuleInstance {
    store: wasmtime::Store<ModuleData>,
    memory: wasmtime::Memory,
    instance: wasmtime::Instance,
    handle: WasmAici,
    logit_ptr: WasmPtr,
    globals: Arc<RwLock<GlobalInfo>>,
    op: Option<IdxOp>,
    had_error: bool,
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

const MAXLINE: usize = 512;
const MAXLOG: usize = 2048;

impl ModuleData {
    pub fn append_line(&mut self, s: &str) {
        let bytes = s.as_bytes();
        if bytes.len() > MAXLINE {
            self.log.extend_from_slice(&bytes[..MAXLINE]);
            self.log.push(46);
            self.log.push(46);
            self.log.push(46);
        } else {
            self.log.extend_from_slice(bytes);
        }
        self.log.push(10);
        if self.log.len() > MAXLOG {
            let drop = MAXLINE + 64;
            self.printed_log = std::cmp::max(0, self.printed_log as isize - drop as isize) as usize;
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

pub struct IdxOp {
    dst_slice: &'static mut [u8],
    op: ThreadOp,
}

pub type Id = usize;
pub type Token = u32;

pub enum ThreadOp {
    Prompt { prompt: Vec<Token> },
    Gen { gen: Token },
}

type WasmPtr = u32;
type WasmAici = u32;

impl ModuleInstance {
    fn call_func<Params, Results>(&mut self, name: &str, params: Params) -> Result<Results>
    where
        Params: wasmtime::WasmParams,
        Results: wasmtime::WasmResults,
    {
        if self.had_error {
            return Err(anyhow!("Previous WASM Error"));
        }
        let f = self
            .instance
            .get_typed_func::<Params, Results>(&mut self.store, name)?;
        let r = f.call(&mut self.store, params);
        let ctx = self.store.data_mut();
        match &r {
            Err(e) => {
                self.had_error = true;
                ctx.append_line(&format!("WASM Error: {}", e.to_string()))
            }
            _ => {}
        }
        ctx.flush_logs(name);
        r
    }

    fn write_mem<T>(&mut self, src: &[T], ptr: WasmPtr) -> Result<()> {
        let len = src.len();
        let numbytes = len * std::mem::size_of::<T>();

        let dest_slice = &mut self.memory.data_mut(&mut self.store)[ptr as usize..];

        ensure!(dest_slice.len() >= numbytes);

        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr() as *const u8,
                dest_slice.as_mut_ptr(),
                numbytes,
            );
        }

        Ok(())
    }

    fn read_mem<T>(&self, ptr: WasmPtr, target: &mut [T]) -> Result<()> {
        let numbytes = target.len() * std::mem::size_of::<T>();
        let src_slice = &self.memory.data(&self.store)[ptr as usize..];
        ensure!(src_slice.len() >= numbytes);
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_slice.as_ptr(),
                target.as_mut_ptr() as *mut u8,
                numbytes,
            )
        }
        Ok(())
    }
}

impl ModuleInstance {
    pub fn new(
        id: Id,
        module: wasmtime::Module,
        module_arg: Arc<String>,
        linker: Arc<wasmtime::Linker<ModuleData>>,
        globals: Arc<RwLock<GlobalInfo>>,
    ) -> Result<Self> {
        let engine = module.engine();

        let mut store = wasmtime::Store::new(
            engine,
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
            },
        );
        let instance = linker.instantiate(&mut store, &module)?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or(anyhow!("memory missing"))?;
        store.data_mut().instance = Some(instance);
        store.data_mut().memory = Some(memory);

        Ok(ModuleInstance {
            handle: 0,
            logit_ptr: 0,
            op: None,
            store,
            memory,
            instance,
            globals,
            had_error: false,
        })
    }

    #[inline(never)]
    pub fn fork(&mut self, id: Id) -> Result<Self> {
        let t0 = Instant::now();
        let mut fork = Self::new(
            id,
            self.store.data().module.clone(),
            self.store.data().module_arg.clone(),
            self.store.data().linker.clone(),
            self.globals.clone(),
        )?;
        fork.handle = self.handle;
        fork.logit_ptr = self.logit_ptr;
        let src = self.memory;
        let dst = fork.memory;
        let missing_size = src.data_size(&self.store) - dst.data_size(&fork.store);
        dst.grow(&mut fork.store, (missing_size >> 16) as u64)?;
        // TIME: 1-2ms at ~4MB
        dst.data_mut(&mut fork.store)
            .copy_from_slice(src.data(&self.store));
        info!(
            "fork time: {:?}, mem={}kB",
            t0.elapsed(),
            dst.data_size(&fork.store) / 1024
        );
        Ok(fork)
    }

    fn run_init(&mut self) -> Result<()> {
        self.call_func::<(), ()>("aici_init", ())?;
        Ok(())
    }

    pub fn run_main(&mut self) -> Result<()> {
        self.run_init()?;
        let t0 = Instant::now();
        let _ = self.call_func::<(i32, i32), i32>("main", (0, 0))?;
        println!("{}\n", self.store.data_mut().string_log());
        println!("time: {:?}", t0.elapsed());
        Ok(())
    }

    pub fn add_op(&mut self, dst_slice: &'static mut [u8], op: ThreadOp) {
        assert!(self.op.is_none());
        self.op = Some(IdxOp { dst_slice, op });
    }

    fn setup_logit_bias(&mut self, handle: WasmAici) -> Result<u32> {
        let vocab_size = { self.globals.read().unwrap().tokrx_info.vocab_size };
        let logit_ptr = self.call_func::<(WasmAici, u32), WasmPtr>(
            "aici_get_logit_bias_buffer",
            (handle, vocab_size),
        )?;

        self.handle = handle;
        self.logit_ptr = logit_ptr;

        Ok(logit_ptr)
    }

    pub fn exec(&mut self) -> Value {
        let mut json_type = "ok";
        let mut suffix = "".to_string();
        let t0 = Instant::now();

        match self.exec_inner() {
            Ok(_) => {}
            Err(e) => {
                json_type = "error";
                suffix = format!("\nError: {}", e.to_string());
            }
        };

        let logs = self.store.data_mut().string_log();
        json!({
            "type": json_type,
            "millis": t0.elapsed().as_millis() as u64,
            "logs": logs + &suffix,
        })
    }

    fn exec_inner(&mut self) -> Result<()> {
        let opidx = std::mem::replace(&mut self.op, None).unwrap();

        match opidx.op {
            ThreadOp::Prompt { prompt, .. } => {
                self.run_init()?;

                let handle = self.call_func::<(), WasmAici>("aici_create", ())?;
                let logit_ptr = self.setup_logit_bias(handle)?;

                let prompt_ptr = self.call_func::<(WasmAici, u32), WasmPtr>(
                    "aici_get_prompt_buffer",
                    (handle, prompt.len().try_into().unwrap()),
                )?;

                self.write_mem(&prompt, prompt_ptr)?;
                self.call_func::<WasmAici, ()>("aici_process_prompt", handle)?;
                self.read_mem(logit_ptr, opidx.dst_slice)?;
            }

            ThreadOp::Gen { gen, .. } => {
                self.call_func::<(WasmAici, Token), ()>("aici_append_token", (self.handle, gen))?;
                self.read_mem(self.logit_ptr, opidx.dst_slice)?;
            }
        }

        Ok(())
    }
}

pub fn setup_linker(engine: &wasmtime::Engine) -> Result<Arc<wasmtime::Linker<ModuleData>>> {
    let mut linker = wasmtime::Linker::<ModuleData>::new(engine);
    linker.func_wrap(
        "env",
        "aici_host_print",
        |mut caller: wasmtime::Caller<'_, ModuleData>, ptr: u32, len: u32| {
            let m = read_caller_mem(&caller, ptr, len);
            caller.data_mut().log.extend_from_slice(&m);
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
            if caller.data().tokenizer.is_none() {
                let lock = caller.data().globals.clone();
                let info = lock.read().unwrap();
                let tok = Tokenizer::from_bytes(info.hf_tokenizer_bytes).unwrap();
                caller.data_mut().tokenizer = Some(tok);
            };
            let m = read_caller_mem(&caller, src, src_size);
            let s = String::from_utf8_lossy(&m);
            let tokens = caller.data().tokenizer.as_ref().unwrap().encode(s, false);
            match tokens {
                Err(_) => 0,
                Ok(tokens) => {
                    let bytes = clone_vec_as_bytes(&tokens.get_ids());
                    write_caller_mem(&mut caller, dst, 4 * dst_size, &bytes) / 4
                }
            }
        },
    )?;

    let linker = Arc::new(linker);
    Ok(linker)
}
