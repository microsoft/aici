use anyhow::{anyhow, ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use wasmtime;

pub struct WasmCtx {
    store: wasmtime::Store<()>,
    instance: wasmtime::Instance,
    memory: wasmtime::Memory,
}

pub struct GlobalInfo {
    pub vocab_size: u32,
}

pub struct GvmInfo {
    handle: WasmGvm,
    logit_ptr: WasmPtr,
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
        linker: Arc<wasmtime::Linker<()>>,
        globals: Arc<RwLock<GlobalInfo>>,
    ) -> Result<Self> {
        let engine = module.engine();

        let mut store = wasmtime::Store::new(engine, ());
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
        let wasm = &mut self.wasm;

        let init_fn = wasm
            .instance
            .get_typed_func::<(), ()>(&mut wasm.store, "gvm_init")?;
        init_fn.call(&mut wasm.store, ())?;

        Ok(())
    }

    pub fn run_main(&mut self) -> Result<()> {
        self.run_init()?;

        let wasm = &mut self.wasm;
        let main_fn = wasm
            .instance
            .get_typed_func::<(i32, i32), i32>(&mut wasm.store, "main")?;
        main_fn.call(&mut wasm.store, (0, 0))?;

        Ok(())
    }

    pub fn add_op(&mut self, dst_slice: &'static mut [u8], op: GvmOp) -> bool {
        self.ops.push(IdxOp { dst_slice, op });
        self.ops.len() == 1
    }

    fn setup_logit_bias(&mut self, id: Id, handle: WasmGvm) -> Result<u32> {
        let wasm = &mut self.wasm;
        let inst = wasm.instance;

        let gvm_get_logit_bias_buffer = inst.get_typed_func::<(WasmGvm, u32), WasmPtr>(
            &mut wasm.store,
            "gvm_get_logit_bias_buffer",
        )?;

        let vocab_size = { self.globals.read().unwrap().vocab_size };

        let logit_ptr = gvm_get_logit_bias_buffer.call(&mut wasm.store, (handle, vocab_size))?;

        let ginfo = GvmInfo { handle, logit_ptr };
        self.gvm_handles.insert(id, ginfo);

        Ok(logit_ptr)
    }

    pub fn exec(&mut self) -> Result<()> {
        let inst = self.wasm.instance;
        let ops = std::mem::replace(&mut self.ops, Vec::new());

        for opidx in ops {
            let logit_ptr;
            match &opidx.op {
                GvmOp::Prompt {
                    id,
                    prompt,
                    module_arg,
                    ..
                } => {
                    ensure!(module_arg == "", "module_arg not supported yet");

                    self.run_init()?;

                    let gvm_create =
                        inst.get_typed_func::<(), WasmGvm>(&mut self.wasm.store, "gvm_create")?;
                    let handle = gvm_create.call(&mut self.wasm.store, ())?;
                    logit_ptr = self.setup_logit_bias(*id, handle)?;

                    let gvm_get_prompt_buffer = inst.get_typed_func::<(WasmGvm, u32), WasmPtr>(
                        &mut self.wasm.store,
                        "gvm_get_prompt_buffer",
                    )?;
                    let prompt_ptr = gvm_get_prompt_buffer.call(
                        &mut self.wasm.store,
                        (handle, prompt.len().try_into().unwrap()),
                    )?;

                    self.wasm.write_mem(&prompt, prompt_ptr)?;

                    let gvm_process_prompt = inst.get_typed_func::<WasmGvm, ()>(
                        &mut self.wasm.store,
                        "gvm_process_prompt",
                    )?;
                    gvm_process_prompt.call(&mut self.wasm.store, handle)?;
                }

                GvmOp::Gen { id, gen, clone_id } => {
                    match clone_id {
                        None => {}
                        Some(cid) => {
                            let gvm_clone = inst.get_typed_func::<WasmGvm, WasmGvm>(
                                &mut self.wasm.store,
                                "gvm_clone",
                            )?;
                            let handles = &mut self.gvm_handles;
                            let parent = handles
                                .get(&cid)
                                .ok_or(anyhow!("invalid clone_id {} (inner)", cid))?;
                            let child = gvm_clone.call(&mut self.wasm.store, parent.handle)?;
                            self.setup_logit_bias(*id, child)?;
                        }
                    }

                    let handles = &self.gvm_handles;
                    let ginfo = handles.get(&id).ok_or(anyhow!("invalid Gen id {}", id))?;

                    logit_ptr = ginfo.logit_ptr;

                    let gvm_append_token = inst.get_typed_func::<(WasmGvm, Token), ()>(
                        &mut self.wasm.store,
                        "gvm_append_token",
                    )?;
                    gvm_append_token.call(&mut self.wasm.store, (ginfo.handle, *gen))?;
                }
            };

            self.wasm.read_mem(logit_ptr, opidx.dst_slice)?;
        }

        Ok(())
    }
}
