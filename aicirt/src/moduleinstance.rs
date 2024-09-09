use crate::{
    api::ModuleInstId,
    hostimpl::{AiciLimits, GlobalInfo, ModuleData},
    setup_component_linker,
    worker::{GroupHandle, RtMidProcessArg},
    TimerSet, UserError,
};
use aici_abi::toktrie::TokTrie;
use aicirt::{
    api::{InferenceCapabilities, SequenceResult},
    bindings::{self, exports::aici::abi::controller::*, InitPromptResult},
    bintokens::ByteTokenizer,
};
use anyhow::Result;
use std::{path::PathBuf, sync::Arc, time::Instant};
use wasmtime;

#[derive(Clone)]
pub struct WasmContext {
    pub engine: wasmtime::Engine,
    pub component_linker: Arc<wasmtime::component::Linker<ModuleData>>,
    pub globals: GlobalInfo,
    pub limits: AiciLimits,
    pub timers: TimerSet,
}

impl WasmContext {
    pub fn deserialize_component(&self, path: PathBuf) -> Result<wasmtime::component::Component> {
        // TODO: Use type safety to ensure that the input is derived from `Component::serialize` or
        // `Engine::precompile_component`.
        unsafe { wasmtime::component::Component::deserialize_file(&self.engine, path) }
    }

    pub fn new(
        inference_caps: InferenceCapabilities,
        limits: AiciLimits,
        tokenizer: ByteTokenizer,
    ) -> Result<Self> {
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
            .wasm_component_model(true)
            .wasm_bulk_memory(true)
            .wasm_multi_value(true)
            .wasm_memory64(false)
            .strategy(wasmtime::Strategy::Auto)
            .cranelift_nan_canonicalization(false)
            .parallel_compilation(true);

        // we use fork()
        cfg.macos_use_mach_ports(false);

        // disable stuff we don't need
        cfg.wasm_backtrace_details(wasmtime::WasmBacktraceDetails::Disable);

        // compilation in Speed mode seems to be ~10% slower but the generated code is 20-30% faster
        cfg.cranelift_opt_level(wasmtime::OptLevel::Speed);

        let engine = wasmtime::Engine::new(&cfg)?;
        // let linker = setup_linker(&engine)?;
        let component_linker = setup_component_linker(&engine)?;

        let tokens = tokenizer.token_bytes();
        let trie = TokTrie::from(&tokenizer.tokrx_info(), &tokens);
        trie.check_against(&tokens);
        let bytes = trie.serialize();
        // validate
        let trie2 = TokTrie::from_bytes(&bytes);
        assert!(trie.info().to_bin() == trie2.info().to_bin());
        trie2.check_against(&tokens);

        // let tok = tokenizers::Tokenizer::from_bytes(tokenizer.hf_bytes).unwrap();
        // let tokens = tok.encode("I am something", false).unwrap();
        // println!("tokens: {:?}", tokens);

        let globals = GlobalInfo {
            tokrx_info: tokenizer.tokrx_info(),
            tok_trie: Arc::new(trie2),
            trie_bytes: Arc::new(bytes),
            hf_tokenizer: Arc::new(tokenizer.hf_tokenizer),
            inference_caps,
        };

        Ok(Self {
            engine,
            component_linker,
            globals,
            limits,
            timers: TimerSet::new(),
        })
    }
}

pub struct ModuleInstance {
    store: wasmtime::Store<ModuleData>,
    aici: bindings::Aici,
    runner: bindings::Runner,
    #[allow(dead_code)]
    limits: AiciLimits,
}

impl ModuleInstance {
    pub fn new(
        id: ModuleInstId,
        ctx: WasmContext,
        component: wasmtime::component::Component,
        module_arg: String,
        group_channel: GroupHandle,
    ) -> Result<Self> {
        let mut store = wasmtime::Store::new(
            &ctx.engine.clone(),
            ModuleData::new(id, &ctx.limits, ctx.globals, group_channel),
        );
        store.limiter(|state| &mut state.store_limits);

        let aici = bindings::Aici::instantiate(&mut store, &component, &ctx.component_linker)?;
        let runner = aici
            .aici_abi_controller()
            .runner()
            .call_constructor(&mut store, &module_arg)?;

        Ok(ModuleInstance {
            store,
            aici,
            runner,
            limits: ctx.limits,
        })
    }

    pub fn set_id(&mut self, id: ModuleInstId) {
        self.store.data_mut().id = id;
    }

    pub fn group_channel(&self) -> &GroupHandle {
        &self.store.data().group_channel
    }

    fn do_mid_process(&mut self, arg: RtMidProcessArg) -> Result<MidProcessResult> {
        self.aici.aici_abi_controller().runner().call_mid_process(
            &mut self.store,
            self.runner,
            &arg.op,
        )
    }

    fn seq_result<T>(&mut self, lbl: &str, t0: Instant, res: Result<T>) -> SequenceResult<T> {
        // 10us accuracy for Spectre mitigation
        let micros = (t0.elapsed().as_micros() as u64 / 10) * 10;
        let logs = self.store.data_mut().string_log();
        let storage = std::mem::take(&mut self.store.data_mut().storage_log);
        match res {
            Ok(r) => SequenceResult {
                error: String::new(),
                logs,
                storage,
                micros,
                result: Some(r),
            },

            Err(e) => {
                let error = format!("Error ({lbl}): {}", UserError::maybe_stacktrace(&e));
                let logs = logs + "\n" + &error;
                log::warn!("exec: {error}");
                SequenceResult {
                    error,
                    logs,
                    storage,
                    micros,
                    result: None,
                }
            }
        }
    }

    pub fn mid_process(&mut self, op: RtMidProcessArg) -> SequenceResult<MidProcessResult> {
        let t0 = Instant::now();
        let res = self.do_mid_process(op);
        // log::info!("mid_process: {:?}", t0.elapsed());
        self.seq_result("mid", t0, res)
    }

    pub fn tokenize(&mut self, s: &str) -> Result<Vec<u32>> {
        self.store.data_mut().tokenize_bytes_greedy(s)
    }

    fn setup_inner(&mut self, prompt: Vec<TokenId>) -> Result<InitPromptResult> {
        let res = self.aici.aici_abi_controller().runner().call_init_prompt(
            &mut self.store,
            self.runner,
            &InitPromptArg { prompt },
        )?;
        Ok(res.into())
    }

    pub fn setup(&mut self, prompt: Vec<TokenId>) -> SequenceResult<InitPromptResult> {
        let t0 = Instant::now();
        match self.setup_inner(prompt) {
            Err(err) => self.seq_result("setup", t0, Err(err)),
            Ok(res) => self.seq_result("setup", t0, Ok(res)),
        }
    }
}
