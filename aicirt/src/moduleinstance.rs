use aici_abi::toktree::TokTrie;
use aici_abi::{InitPromptArg, ProcessResult, TokenId};
use aici_tokenizers::Tokenizer;
use anyhow::{anyhow, ensure, Result};
use log::warn;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use wasmtime;

use crate::hostimpl::{setup_linker, AiciLimits, GlobalInfo, ModuleData, ModuleInstId};
use crate::shm::Shm;
use crate::worker::ExecOp;

#[derive(Clone)]
pub struct WasmContext {
    pub engine: wasmtime::Engine,
    pub linker: Arc<wasmtime::Linker<ModuleData>>,
    pub globals: GlobalInfo,
    pub limits: AiciLimits,
}

impl WasmContext {
    pub fn deserialize_module(&self, path: PathBuf) -> Result<wasmtime::Module> {
        unsafe { wasmtime::Module::deserialize_file(&self.engine, path) }
    }

    pub fn new(limits: AiciLimits, tokenizer: Tokenizer) -> Result<Self> {
        let mut cfg = wasmtime::Config::default();
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

        let engine = wasmtime::Engine::new(&cfg)?;
        let linker = setup_linker(&engine)?;

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
            engine,
            linker,
            globals,
            limits,
        })
    }
}

pub struct ModuleInstance {
    store: wasmtime::Store<ModuleData>,
    memory: wasmtime::Memory,
    instance: wasmtime::Instance,
    handle: WasmAici,
    had_error: bool,
    #[allow(dead_code)]
    limits: AiciLimits,
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
        if r.is_err() {
            self.had_error = true;
        }
        ctx.flush_logs(name);
        r
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
        id: ModuleInstId,
        ctx: WasmContext,
        module: wasmtime::Module,
        module_arg: Arc<String>,
    ) -> Result<Self> {
        let engine = module.engine();

        let mut store = wasmtime::Store::new(
            engine,
            ModuleData::new(
                id,
                &ctx.limits,
                &module,
                module_arg,
                &ctx.linker,
                ctx.globals,
            ),
        );
        store.limiter(|state| &mut state.store_limits);

        let instance = ctx.linker.instantiate(&mut store, &module)?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or(anyhow!("memory missing"))?;
        store.data_mut().instance = Some(instance);
        store.data_mut().memory = Some(memory);

        Ok(ModuleInstance {
            handle: 0,
            store,
            memory,
            instance,
            had_error: false,
            limits: ctx.limits,
        })
    }

    pub fn set_id(&mut self, id: ModuleInstId) {
        self.store.data_mut().id = id;
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

    pub fn exec(&mut self, op: ExecOp, shm: &Shm) -> Value {
        let mut json_type = "ok";
        let mut suffix = "".to_string();
        let t0 = Instant::now();

        self.store.data_mut().set_exec_data(op, shm);
        let res = self.call_func::<WasmAici, ()>("aici_process", self.handle);

        let res = match res {
            Ok(()) => {
                let bytes = &self.store.data().process_result;
                if bytes.len() == 0 {
                    Err(anyhow!("aici_host_return_process_result not called"))
                } else {
                    serde_json::from_slice::<ProcessResult>(bytes).map_err(|e| e.into())
                }
            }
            Err(e) => Err(e),
        };

        let res = match res {
            Ok(ProcessResult::SampleWithBias) => Ok(()),
            Ok(r) => Err(anyhow!("unhandled {r:?}")),
            Err(e) => Err(e),
        };

        match res {
            Ok(_) => {}
            Err(e) => {
                json_type = "error";
                suffix = format!("\nError: {:?}", e);
                warn!("exec error:{}", suffix);
            }
        };

        let logs = self.store.data_mut().string_log();
        let ff_tokens = self.store.data().ff_tokens.clone();
        self.store.data_mut().ff_tokens.clear();
        json!({
            "type": json_type,
            "millis": t0.elapsed().as_millis() as u64,
            "logs": logs + &suffix,
            "ff_tokens": ff_tokens,
        })
    }

    pub fn tokenize(&mut self, s: &str) -> Result<Vec<u32>> {
        self.store.data_mut().tokenize(s)
    }

    pub fn setup(&mut self, prompt: Vec<TokenId>) -> Result<()> {
        self.run_init()?;

        self.handle = self.call_func::<(), WasmAici>("aici_create", ())?;

        self.store
            .data_mut()
            .set_process_arg(serde_json::to_vec(&InitPromptArg { prompt })?);
        self.call_func::<WasmAici, ()>("aici_init_prompt", self.handle)?;

        Ok(())
    }
}
