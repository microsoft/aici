use aici_abi::bytes::TokRxInfo;
use anyhow::{anyhow, ensure, Result};
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use wasmtime;

#[derive(Clone)]
pub struct AiciContext {
    id: Id,
    log: Vec<u8>,
    globals: Arc<RwLock<GlobalInfo>>,
    // TODO there is way too much cloning of this thing
    module_arg: String,
}

impl AiciContext {
    pub fn from(id: Id, module_arg: String, globals: Arc<RwLock<GlobalInfo>>) -> Self {
        AiciContext {
            id,
            log: Vec::new(),
            globals,
            module_arg,
        }
    }
    pub fn fake(globals: Arc<RwLock<GlobalInfo>>) -> Self {
        Self::from(1000000, "".to_string(), globals)
    }

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

struct WasmCtx {
    store: wasmtime::Store<AiciContext>,
    linker: Arc<wasmtime::Linker<AiciContext>>,
    instance: wasmtime::Instance,
    memory: wasmtime::Memory,
    module: wasmtime::Module,
    error: bool,
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

impl AiciInfo {
    pub fn context(&self, module_arg: String, globals: Arc<RwLock<GlobalInfo>>) -> AiciContext {
        AiciContext::from(self.id, module_arg, globals)
    }
}

pub struct ModuleInstance {
    globals: Arc<RwLock<GlobalInfo>>,
    info: AiciInfo,
    module_arg: String,
    wasm: WasmCtx,
    ops: Vec<IdxOp>, // for next req
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

impl WasmCtx {
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

    fn call_func_in<Params, Results>(
        &mut self,
        ginfo: &AiciContext,
        name: &str,
        params: Params,
    ) -> Result<Results>
    where
        Params: wasmtime::WasmParams,
        Results: wasmtime::WasmResults,
    {
        let mut ginfo = ginfo.clone();
        ginfo = std::mem::replace(self.store.data_mut(), ginfo);
        let r = self.call_func(name, params);
        let _ = std::mem::replace(self.store.data_mut(), ginfo);
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
        module: wasmtime::Module,
        module_arg: String,
        linker: Arc<wasmtime::Linker<AiciContext>>,
        globals: Arc<RwLock<GlobalInfo>>,
    ) -> Result<Self> {
        let engine = module.engine();

        let mut store = wasmtime::Store::new(engine, AiciContext::fake(globals.clone()));
        let instance = linker.instantiate(&mut store, &module)?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or(anyhow!("memory missing"))?;

        Ok(ModuleInstance {
            info: AiciInfo {
                id: 0,
                handle: 0,
                logit_ptr: 0,
            },
            ops: Vec::new(),
            module_arg,
            wasm: WasmCtx {
                store,
                instance,
                memory,
                module,
                linker,
                error: false,
            },
            globals,
        })
    }

    pub fn fork(&mut self) -> Result<Self> {
        let mut fork = Self::new(
            self.wasm.module.clone(),
            self.module_arg.clone(),
            self.wasm.linker.clone(),
            self.globals.clone(),
        )?;
        fork.info = self.info.clone();
        let src = self.wasm.memory;
        let dst = fork.wasm.memory;
        info!(
            "grow mem to: {} from {}",
            src.data_size(&self.wasm.store),
            dst.data_size(&fork.wasm.store)
        );
        let missing_size = src.data_size(&self.wasm.store) - dst.data_size(&fork.wasm.store);
        dst.grow(&mut fork.wasm.store, (missing_size >> 16) as u64)?;
        dst.data_mut(&mut fork.wasm.store)
            .copy_from_slice(src.data(&self.wasm.store));
        Ok(fork)
    }

    fn run_init(&mut self) -> Result<()> {
        self.wasm.call_func::<(), ()>("aici_init", ())?;
        Ok(())
    }

    pub fn run_main(&mut self) -> Result<()> {
        self.run_init()?;
        let t0 = Instant::now();
        let _ = self.wasm.call_func::<(i32, i32), i32>("main", (0, 0))?;
        println!("time: {:?}", t0.elapsed());
        Ok(())
    }

    pub fn add_op(&mut self, dst_slice: &'static mut [u8], op: AiciOp) -> bool {
        self.ops.push(IdxOp { dst_slice, op });
        self.ops.len() == 1
    }

    fn setup_logit_bias(&mut self, id: Id, handle: WasmAici) -> Result<u32> {
        let vocab_size = { self.globals.read().unwrap().tokrx_info.vocab_size };
        let logit_ptr = self.wasm.call_func_in::<(WasmAici, u32), WasmPtr>(
            &AiciContext::from(id, self.module_arg.clone(), self.globals.clone()),
            "aici_get_logit_bias_buffer",
            (handle, vocab_size),
        )?;

        self.info.id = id;
        self.info.handle = handle;
        self.info.logit_ptr = logit_ptr;

        Ok(logit_ptr)
    }

    pub fn exec(&mut self) -> Result<Value> {
        let ops = std::mem::replace(&mut self.ops, Vec::new());

        assert!(ops.len() == 1);

        for opidx in ops {
            match &opidx.op {
                AiciOp::Prompt { id, prompt, .. } => {
                    self.run_init()?;

                    let ctx = AiciContext::from(*id, self.module_arg.clone(), self.globals.clone());
                    let handle = self
                        .wasm
                        .call_func_in::<(), WasmAici>(&ctx, "aici_create", ())?;
                    let logit_ptr = self.setup_logit_bias(*id, handle)?;

                    let prompt_ptr = self.wasm.call_func_in::<(WasmAici, u32), WasmPtr>(
                        &ctx,
                        "aici_get_prompt_buffer",
                        (handle, prompt.len().try_into().unwrap()),
                    )?;

                    self.wasm.write_mem(&prompt, prompt_ptr)?;
                    self.wasm
                        .call_func_in::<WasmAici, ()>(&ctx, "aici_process_prompt", handle)?;
                    self.wasm.read_mem(logit_ptr, opidx.dst_slice)?;
                }

                AiciOp::Gen { id, gen, .. } => {
                    self.info.id = *id;
                    let ginfo = &self.info;
                    self.wasm.call_func_in::<(WasmAici, Token), ()>(
                        &ginfo.context(self.module_arg.clone(), self.globals.clone()),
                        "aici_append_token",
                        (ginfo.handle, *gen),
                    )?;
                    self.wasm.read_mem(ginfo.logit_ptr, opidx.dst_slice)?;
                }
            };
        }

        Ok(json!({}))
    }
}

pub fn setup_linker(engine: &wasmtime::Engine) -> Result<Arc<wasmtime::Linker<AiciContext>>> {
    let mut linker = wasmtime::Linker::<AiciContext>::new(engine);
    linker.func_wrap(
        "env",
        "aici_host_print",
        |mut caller: wasmtime::Caller<'_, AiciContext>, ptr: u32, len: u32| {
            let mut bytes = if let Some(wasmtime::Extern::Memory(mem)) = caller.get_export("memory")
            {
                let ptr = ptr as usize;
                let len = len as usize;
                let m = &mem.data(&caller)[ptr..(ptr + len)];
                Vec::from(m)
            } else {
                panic!("no memory")
            };
            caller.data_mut().log.append(&mut bytes);
        },
    )?;

    // uint32_t aici_host_read_token_trie(uint8_t *dst, uint32_t size);
    linker.func_wrap(
        "env",
        "aici_host_read_token_trie",
        |mut caller: wasmtime::Caller<'_, AiciContext>, ptr: u32, len: u32| {
            let lock = caller.data().globals.clone();
            let info = lock.read().unwrap();
            if let Some(wasmtime::Extern::Memory(mem)) = caller.get_export("memory") {
                if len > 0 {
                    let ptr = ptr as usize;
                    let len = len as usize;
                    let min_len = std::cmp::min(len as usize, info.trie_bytes.len());
                    mem.write(&mut caller, ptr, &info.trie_bytes[..min_len])
                        .unwrap();
                }
                info.trie_bytes.len() as u32
            } else {
                panic!("no memory")
            }
        },
    )?;

    // uint32_t aici_host_read_arg(uint8_t *dst, uint32_t size);
    linker.func_wrap(
        "env",
        "aici_host_read_arg",
        |mut caller: wasmtime::Caller<'_, AiciContext>, ptr: u32, len: u32| {
            if let Some(wasmtime::Extern::Memory(mem)) = caller.get_export("memory") {
                let rlen = caller.data().module_arg.as_bytes().len();
                if len > 0 {
                    let ptr = ptr as usize;
                    let len = len as usize;
                    let min_len = std::cmp::min(len as usize, rlen);
                    let data = caller.data().module_arg.clone();
                    let data = data.as_bytes();
                    mem.write(&mut caller, ptr, &data[..min_len]).unwrap();
                }
                rlen as u32
            } else {
                panic!("no memory")
            }
        },
    )?;

    let linker = Arc::new(linker);
    Ok(linker)
}
