use aici_abi::bytes::TokRxInfo;
use anyhow::{anyhow, ensure, Result};
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use wasmtime;

// this is avaiable to functions called from wasm
pub struct ModuleData {
    id: Id,
    log: Vec<u8>,
    globals: Arc<RwLock<GlobalInfo>>,
    module_arg: Arc<String>,
    linker: Arc<wasmtime::Linker<ModuleData>>,
    instance: Option<wasmtime::Instance>,
    memory: Option<wasmtime::Memory>,
    module: wasmtime::Module,
}

pub struct ModuleInstance {
    store: wasmtime::Store<ModuleData>,
    memory: wasmtime::Memory,
    instance: wasmtime::Instance,
    info: AiciInfo,
    globals: Arc<RwLock<GlobalInfo>>,
    ops: Vec<IdxOp>, // for next req
    error: bool,
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

impl ModuleData {
    pub fn append_line(&mut self, s: &str) {
        self.log.extend_from_slice(s.as_bytes());
        self.log.push(10)
    }

    pub fn flush_logs(&mut self, name: &str) {
        if self.log.len() == 0 {
            return;
        }

        let logs = String::from_utf8_lossy(&self.log).to_string();
        self.log.clear();

        for line in logs.lines() {
            println!("{}:{}> {}", self.id, name, line)
        }
    }
}

pub struct GlobalInfo {
    pub tokrx_info: TokRxInfo,
    pub trie_bytes: Vec<u8>,
}

#[derive(Clone)]
pub struct AiciInfo {
    id: Id,
    handle: WasmAici,
    logit_ptr: WasmPtr,
}

pub struct IdxOp {
    dst_slice: &'static mut [u8],
    op: AiciOp,
}

pub type Id = usize;
pub type Token = u32;

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

type WasmPtr = u32;
type WasmAici = u32;

impl ModuleInstance {
    fn call_func<Params, Results>(&mut self, name: &str, params: Params) -> Result<Results>
    where
        Params: wasmtime::WasmParams,
        Results: wasmtime::WasmResults,
    {
        if self.error {
            return Err(anyhow!("Previous WASM Error"));
        }
        let f = self
            .instance
            .get_typed_func::<Params, Results>(&mut self.store, name)?;
        let r = f.call(&mut self.store, params);
        let ctx = self.store.data_mut();
        match &r {
            Err(e) => {
                self.error = true;
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
                globals: globals.clone(),
                module_arg,
                module: module.clone(),
                linker: linker.clone(),
                instance: None,
                memory: None,
            },
        );
        let instance = linker.instantiate(&mut store, &module)?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or(anyhow!("memory missing"))?;
        store.data_mut().instance = Some(instance);
        store.data_mut().memory = Some(memory);

        Ok(ModuleInstance {
            info: AiciInfo {
                id,
                handle: 0,
                logit_ptr: 0,
            },
            ops: Vec::new(),
            store,
            memory,
            instance,
            globals,
            error: false,
        })
    }

    pub fn fork(&mut self, id: Id) -> Result<Self> {
        let mut fork = Self::new(
            id,
            self.store.data().module.clone(),
            self.store.data().module_arg.clone(),
            self.store.data().linker.clone(),
            self.globals.clone(),
        )?;
        fork.info = self.info.clone();
        let src = self.memory;
        let dst = fork.memory;
        info!(
            "grow mem to: {} from {}",
            src.data_size(&self.store),
            dst.data_size(&fork.store)
        );
        let missing_size = src.data_size(&self.store) - dst.data_size(&fork.store);
        dst.grow(&mut fork.store, (missing_size >> 16) as u64)?;
        dst.data_mut(&mut fork.store)
            .copy_from_slice(src.data(&self.store));
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
        println!("time: {:?}", t0.elapsed());
        Ok(())
    }

    pub fn add_op(&mut self, dst_slice: &'static mut [u8], op: AiciOp) -> bool {
        self.ops.push(IdxOp { dst_slice, op });
        self.ops.len() == 1
    }

    fn setup_logit_bias(&mut self, handle: WasmAici) -> Result<u32> {
        let vocab_size = { self.globals.read().unwrap().tokrx_info.vocab_size };
        let logit_ptr = self.call_func::<(WasmAici, u32), WasmPtr>(
            "aici_get_logit_bias_buffer",
            (handle, vocab_size),
        )?;

        self.info.handle = handle;
        self.info.logit_ptr = logit_ptr;

        Ok(logit_ptr)
    }

    pub fn exec(&mut self) -> Result<Value> {
        let ops = std::mem::replace(&mut self.ops, Vec::new());

        assert!(ops.len() == 1);

        for opidx in ops {
            match &opidx.op {
                AiciOp::Prompt { prompt, .. } => {
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

                AiciOp::Gen { gen, .. } => {
                    self.call_func::<(WasmAici, Token), ()>(
                        "aici_append_token",
                        (self.info.handle, *gen),
                    )?;
                    self.read_mem(self.info.logit_ptr, opidx.dst_slice)?;
                }
            };
        }

        Ok(json!({}))
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

    let linker = Arc::new(linker);
    Ok(linker)
}
