use anyhow::{anyhow, ensure, Result};
use gvm_abi::rx::TokRxInfo;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use wasmtime;

#[derive(Clone)]
pub struct GvmContext {
    id: Id,
    log: Vec<u8>,
    globals: Arc<RwLock<GlobalInfo>>,
}

impl GvmContext {
    pub fn from(id: Id, globals: Arc<RwLock<GlobalInfo>>) -> Self {
        GvmContext {
            id,
            log: Vec::new(),
            globals,
        }
    }
    pub fn fake(globals: Arc<RwLock<GlobalInfo>>) -> Self {
        Self::from(1000000, globals)
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
    store: wasmtime::Store<GvmContext>,
    instance: wasmtime::Instance,
    memory: wasmtime::Memory,
}

pub struct GlobalInfo {
    pub tokrx_info: TokRxInfo,
    pub trie_bytes: Vec<u8>,
}

pub struct GvmInfo {
    id: Id,
    handle: WasmGvm,
    logit_ptr: WasmPtr,
}

impl GvmInfo {
    pub fn context(&self, globals: Arc<RwLock<GlobalInfo>>) -> GvmContext {
        GvmContext::from(self.id, globals)
    }
}

pub struct ModuleInstance {
    globals: Arc<RwLock<GlobalInfo>>,
    gvm_handles: HashMap<Id, GvmInfo>,
    wasm: WasmCtx,
    ops: Vec<IdxOp>, // for next req
}

pub struct IdxOp {
    dst_slice: &'static mut [u8],
    op: GvmOp,
}

pub type Id = usize;
pub type Token = u32;

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum GvmOp {
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
type WasmGvm = u32;

impl WasmCtx {
    fn call_func<Params, Results>(&mut self, name: &str, params: Params) -> Result<Results>
    where
        Params: wasmtime::WasmParams,
        Results: wasmtime::WasmResults,
    {
        let f = self
            .instance
            .get_typed_func::<Params, Results>(&mut self.store, name)?;
        let r = f.call(&mut self.store, params);
        let ctx = self.store.data_mut();
        match &r {
            Err(e) => ctx.append_line(&format!("WASM Error: {}", e.to_string())),
            _ => {}
        }
        ctx.flush_logs(name);
        r
    }

    fn call_func_in<Params, Results>(
        &mut self,
        ginfo: &GvmContext,
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
        linker: Arc<wasmtime::Linker<GvmContext>>,
        globals: Arc<RwLock<GlobalInfo>>,
    ) -> Result<Self> {
        let engine = module.engine();

        let mut store = wasmtime::Store::new(engine, GvmContext::fake(globals.clone()));
        let instance = linker.instantiate(&mut store, &module)?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or(anyhow!("memory missing"))?;

        Ok(ModuleInstance {
            gvm_handles: HashMap::new(),
            ops: Vec::new(),
            wasm: WasmCtx {
                store,
                instance,
                memory,
            },
            globals,
        })
    }

    fn run_init(&mut self) -> Result<()> {
        self.wasm.call_func::<(), ()>("gvm_init", ())?;
        Ok(())
    }

    pub fn run_main(&mut self) -> Result<()> {
        self.run_init()?;
        let t0 = Instant::now();
        let _ = self.wasm.call_func::<(i32, i32), i32>("main", (0, 0))?;
        println!("time: {:?}", t0.elapsed());
        Ok(())
    }

    pub fn add_op(&mut self, dst_slice: &'static mut [u8], op: GvmOp) -> bool {
        self.ops.push(IdxOp { dst_slice, op });
        self.ops.len() == 1
    }

    fn setup_logit_bias(&mut self, id: Id, handle: WasmGvm) -> Result<u32> {
        let vocab_size = { self.globals.read().unwrap().tokrx_info.vocab_size };
        let logit_ptr = self.wasm.call_func_in::<(WasmGvm, u32), WasmPtr>(
            &GvmContext::from(id, self.globals.clone()),
            "gvm_get_logit_bias_buffer",
            (handle, vocab_size),
        )?;

        let ginfo = GvmInfo {
            id,
            handle,
            logit_ptr,
        };
        self.gvm_handles.insert(id, ginfo);

        Ok(logit_ptr)
    }

    pub fn exec(&mut self) -> Result<()> {
        let ops = std::mem::replace(&mut self.ops, Vec::new());
        let mut todo = Vec::new();

        for opidx in ops {
            match &opidx.op {
                GvmOp::Prompt {
                    id,
                    prompt,
                    module_arg,
                    ..
                } => {
                    ensure!(module_arg == "", "module_arg not supported yet");

                    self.run_init()?;

                    let ctx = GvmContext::from(*id, self.globals.clone());
                    let handle = self
                        .wasm
                        .call_func_in::<(), WasmGvm>(&ctx, "gvm_create", ())?;
                    let logit_ptr = self.setup_logit_bias(*id, handle)?;

                    let prompt_ptr = self.wasm.call_func_in::<(WasmGvm, u32), WasmPtr>(
                        &ctx,
                        "gvm_get_prompt_buffer",
                        (handle, prompt.len().try_into().unwrap()),
                    )?;

                    self.wasm.write_mem(&prompt, prompt_ptr)?;
                    self.wasm
                        .call_func_in::<WasmGvm, ()>(&ctx, "gvm_process_prompt", handle)?;
                    self.wasm.read_mem(logit_ptr, opidx.dst_slice)?;
                }

                GvmOp::Gen { id, gen, clone_id } => {
                    match clone_id {
                        None => {}
                        Some(cid) => {
                            let handles = &mut self.gvm_handles;
                            let parent = handles
                                .get(&cid)
                                .ok_or(anyhow!("invalid clone_id {} (inner)", cid))?;
                            let child = self.wasm.call_func_in::<WasmGvm, WasmGvm>(
                                &GvmContext::from(*id, self.globals.clone()),
                                "gvm_clone",
                                parent.handle,
                            )?;
                            self.setup_logit_bias(*id, child)?;
                        }
                    }

                    todo.push((*id, *gen, opidx.dst_slice));
                }
            };
        }

        // only append tokens after everything has been cloned
        for (id, gen, dst_slice) in todo {
            let ginfo = self
                .gvm_handles
                .get(&id)
                .ok_or(anyhow!("invalid Gen id {}", id))?;
            self.wasm.call_func_in::<(WasmGvm, Token), ()>(
                &ginfo.context(self.globals.clone()),
                "gvm_append_token",
                (ginfo.handle, gen),
            )?;
            self.wasm.read_mem(ginfo.logit_ptr, dst_slice)?;
        }

        Ok(())
    }
}

pub fn setup_linker(engine: &wasmtime::Engine) -> Result<Arc<wasmtime::Linker<GvmContext>>> {
    let mut linker = wasmtime::Linker::<GvmContext>::new(engine);
    linker.func_wrap(
        "env",
        "gvm_host_print",
        |mut caller: wasmtime::Caller<'_, GvmContext>, ptr: u32, len: u32| {
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

    // uint32_t gvm_host_read_token_trie(uint8_t *dst, uint32_t size);
    linker.func_wrap(
        "env",
        "gvm_host_read_token_trie",
        |mut caller: wasmtime::Caller<'_, GvmContext>, ptr: u32, len: u32| {
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

    let linker = Arc::new(linker);
    Ok(linker)
}
