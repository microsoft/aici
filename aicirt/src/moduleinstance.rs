use crate::{
    api::ModuleInstId,
    hostimpl::{
        setup_linker, AiciLimits, GlobalInfo, ModuleData, LOGIT_BIAS_ALLOW, LOGIT_BIAS_DISALLOW,
    },
    shm::Shm,
    worker::{GroupHandle, RtMidProcessArg},
    TimerSet, UserError,
};
use aici_abi::{
    toktree::TokTrie, InitPromptArg, MidProcessResult, PostProcessArg, PostProcessResult,
    PreProcessArg, PreProcessResult, TokenId,
};
use aicirt::{
    api::{AiciMidProcessResultInner, AiciPostProcessResultInner, SequenceResult},
    bail_user,
    bintokens::ByteTokenizer,
    user_error,
};
use anyhow::{anyhow, bail, ensure, Result};
use serde::Deserialize;
use std::{path::PathBuf, sync::Arc, time::Instant};
use wasmtime;

#[derive(Clone)]
pub struct WasmContext {
    pub engine: wasmtime::Engine,
    pub linker: Arc<wasmtime::Linker<ModuleData>>,
    pub globals: GlobalInfo,
    pub limits: AiciLimits,
    pub timers: TimerSet,
}

impl WasmContext {
    pub fn deserialize_module(&self, path: PathBuf) -> Result<wasmtime::Module> {
        unsafe { wasmtime::Module::deserialize_file(&self.engine, path) }
    }

    pub fn new(limits: AiciLimits, tokenizer: ByteTokenizer) -> Result<Self> {
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

        // we use fork()
        cfg.macos_use_mach_ports(false);

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
            trie_bytes: Arc::new(bytes),
            hf_tokenizer: Arc::new(tokenizer.hf_tokenizer),
        };

        Ok(Self {
            engine,
            linker,
            globals,
            limits,
            timers: TimerSet::new(),
        })
    }
}

pub struct ModuleInstance {
    store: wasmtime::Store<ModuleData>,
    memory: wasmtime::Memory,
    instance: wasmtime::Instance,
    pending_pre_result: Option<PreProcessResult>,
    handle: WasmAici,
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
        if self.store.data().had_error {
            bail_user!("Previous WASM Error");
        }
        let f = self
            .instance
            .get_typed_func::<Params, Results>(&mut self.store, name)?;
        let r = f.call(&mut self.store, params);
        let ctx = self.store.data_mut();
        ctx.flush_logs(name);
        match r {
            Ok(r) => Ok(r),
            Err(e) => {
                ctx.had_error = true;
                if let Some(e) = e.downcast_ref::<UserError>() {
                    Err(user_error!("{}\n{}", ctx.string_log(), e))
                } else if let Some(bt) = e.downcast_ref::<wasmtime::WasmBacktrace>() {
                    Err(user_error!(
                        "{}\n{}\n\n{}",
                        ctx.string_log(),
                        bt,
                        e.root_cause()
                    ))
                } else {
                    Err(anyhow!("{:?}\n\n{}", e, ctx.string_log()))
                }
            }
        }
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
        module_arg: String,
        group_channel: GroupHandle,
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
                group_channel,
            ),
        );
        store.limiter(|state| &mut state.store_limits);

        let instance = ctx.linker.instantiate(&mut store, &module)?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow!("memory missing"))?;
        store.data_mut().instance = Some(instance);
        store.data_mut().memory = Some(memory);

        Ok(ModuleInstance {
            handle: 0,
            store,
            memory,
            instance,
            limits: ctx.limits,
            pending_pre_result: None,
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
        if self
            .instance
            .get_export(&mut self.store, "aici_main")
            .is_some()
        {
            self.call_func::<u32, ()>("aici_main", self.handle)?;
        } else {
            let _ = self.call_func::<(i32, i32), i32>("main", (0, 0))?;
        }
        //println!("{}\n", self.store.data_mut().string_log());
        println!("time: {:?}", t0.elapsed());
        Ok(())
    }

    pub fn group_channel(&self) -> &GroupHandle {
        &self.store.data().group_channel
    }

    fn proc_result<T: for<'a> Deserialize<'a>>(&self) -> Result<T> {
        let bytes = &self.store.data().process_result;
        if bytes.len() == 0 {
            Err(anyhow!("aici_host_return_process_result not called"))
        } else {
            serde_json::from_slice::<T>(bytes).map_err(|e| e.into())
        }
    }

    fn do_pre_process(&mut self, rtarg: PreProcessArg) -> Result<PreProcessResult> {
        let mut ff_tokens = Vec::new();
        let mut cnt = 0;
        loop {
            let mut res = self.do_pre_process_inner(&rtarg)?;

            if res.ff_tokens.len() > 0 {
                ensure!(res.num_forks == 1, "can't fork when returning ff_tokens");
                ensure!(
                    res.suspend == false,
                    "can't suspend when returning ff_tokens"
                );
                ff_tokens.extend_from_slice(&res.ff_tokens);
                let r_post = self.do_post_process(PostProcessArg {
                    tokens: res.ff_tokens.clone(),
                    backtrack: 0,
                })?;
                if r_post.stop {
                    res.num_forks = 0;
                    return Ok(res); // we're stopping - no point returning ff_tokens
                } else {
                    cnt += 1;
                    if cnt > 10 {
                        bail!("too many ff_tokens rounds from pre_process")
                    }
                    continue;
                }
            }

            res.ff_tokens = ff_tokens;
            return Ok(res);
        }
    }

    fn do_pre_process_inner(&mut self, rtarg: &PreProcessArg) -> Result<PreProcessResult> {
        self.store.data_mut().set_pre_process_data(&rtarg);
        self.call_func::<WasmAici, ()>("aici_pre_process", self.handle)?;
        let res: PreProcessResult = self.proc_result()?;
        Ok(res)
    }

    fn do_mid_process(
        &mut self,
        op: RtMidProcessArg,
        shm: &Shm,
    ) -> Result<Option<AiciMidProcessResultInner>> {
        self.store.data_mut().set_mid_process_data(op, shm);
        self.call_func::<WasmAici, ()>("aici_mid_process", self.handle)?;
        match self.proc_result()? {
            MidProcessResult::SampleWithBias { .. } => Ok(None),
            MidProcessResult::Stop { .. } => {
                let eos = self.store.data().globals.tokrx_info.tok_eos;
                self.store
                    .data_mut()
                    .logit_ptr
                    .iter_mut()
                    .for_each(|v| *v = LOGIT_BIAS_DISALLOW);
                self.store.data_mut().logit_ptr[eos as usize] = LOGIT_BIAS_ALLOW;
                Ok(None)
            }
            MidProcessResult::Splice {
                mut backtrack,
                ff_tokens,
            } => {
                let vocab_size = self.store.data().logit_ptr.len();
                if let Some((idx, val)) = ff_tokens.iter().enumerate().find_map(|(idx, t)| {
                    if *t as usize >= vocab_size {
                        Some((idx, *t))
                    } else {
                        None
                    }
                }) {
                    bail!("ff_token out of range ({val} >= {vocab_size} at {idx})")
                } else {
                    log::debug!("backtrack: {backtrack}, ff_tokens:{ff_tokens:?}");
                    if backtrack == 0 {
                        if ff_tokens.len() == 0 {
                            bail!("empty Splice (both backtrack == 0 and ff_tokens == [])")
                        }
                        // first token will be sampled; next tokens will be passed via "ff_tokens"
                        let t0 = ff_tokens[0];
                        self.store.data_mut().logit_ptr[t0 as usize] = LOGIT_BIAS_ALLOW;
                    } else {
                        // we don't really care about biases, as we're going to backtrack this token anyways
                        // but just in case, allow all
                        self.store
                            .data_mut()
                            .logit_ptr
                            .iter_mut()
                            .for_each(|v| *v = LOGIT_BIAS_ALLOW);
                        // don't remove anything from ff_tokens - they all need to be appended after backtracking

                        // backtrack needs to include also the next token to be generated
                        backtrack += 1;
                    }
                    Ok(Some(AiciMidProcessResultInner {
                        ff_tokens,
                        backtrack,
                    }))
                }
            }
        }
    }

    fn do_post_process(&mut self, rtarg: PostProcessArg) -> Result<AiciPostProcessResultInner> {
        self.store.data_mut().set_post_process_data(rtarg);
        self.call_func::<WasmAici, ()>("aici_post_process", self.handle)?;
        let res: PostProcessResult = self.proc_result()?;
        Ok(AiciPostProcessResultInner { stop: res.stop })
    }

    fn seq_result<T>(
        &mut self,
        lbl: &str,
        t0: Instant,
        res: Result<Option<T>>,
    ) -> SequenceResult<T> {
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
                result: r,
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

    pub fn pre_process(&mut self, op: PreProcessArg) -> SequenceResult<PreProcessResult> {
        let t0 = Instant::now();
        if let Some(pp) = self.pending_pre_result.clone() {
            return self.seq_result("pre-cached", t0, Ok(Some(pp)));
        }
        match self.do_pre_process(op) {
            Err(e) => self.seq_result("pre0", t0, Err(e)),
            Ok(pp) => {
                if pp.ff_tokens.len() > 0 {
                    self.pending_pre_result = Some(pp.clone());
                }
                self.seq_result("pre", t0, Ok(Some(pp)))
            }
        }
    }

    pub fn mid_process(
        &mut self,
        op: RtMidProcessArg,
        shm: &Shm,
    ) -> SequenceResult<AiciMidProcessResultInner> {
        let t0 = Instant::now();
        self.pending_pre_result = None;
        let res = self.do_mid_process(op, shm);
        // log::info!("mid_process: {:?}", t0.elapsed());
        self.seq_result("mid", t0, res)
    }

    pub fn post_process(
        &mut self,
        op: PostProcessArg,
    ) -> SequenceResult<AiciPostProcessResultInner> {
        let t0 = Instant::now();
        self.pending_pre_result = None;
        let res = self.do_post_process(op);
        match res {
            Err(e) => {
                let mut r = self.seq_result("post", t0, Err(e));
                r.result = Some(AiciPostProcessResultInner { stop: true });
                r
            }
            Ok(res) => self.seq_result("post", t0, Ok(Some(res))),
        }
    }

    pub fn tokenize(&mut self, s: &str) -> Result<Vec<u32>> {
        self.store.data_mut().tokenize(s)
    }

    fn setup_inner(&mut self, prompt: Vec<TokenId>) -> Result<()> {
        self.run_init()?;

        self.handle = self.call_func::<(), WasmAici>("aici_create", ())?;

        self.store
            .data_mut()
            .set_process_arg(serde_json::to_vec(&InitPromptArg { prompt })?);
        self.call_func::<WasmAici, ()>("aici_init_prompt", self.handle)?;

        Ok(())
    }

    pub fn setup(&mut self, prompt: Vec<TokenId>) -> SequenceResult<PreProcessResult> {
        let t0 = Instant::now();
        match self.setup_inner(prompt) {
            Err(err) => self.seq_result("setup", t0, Err(err)),
            Ok(()) => match self.do_pre_process(PreProcessArg {}) {
                Err(e) => self.seq_result("setup-pre", t0, Err(e)),
                Ok(pp) if pp.suspend || pp.num_forks == 0 => self.seq_result(
                    "setup-pre",
                    t0,
                    Err(user_error!("setup-pre asked for suspend")),
                ),
                Ok(mut pp) => {
                    // force a single fork; we expect another fork request when pre() runs in its own turn
                    pp.num_forks = 1;
                    // don't set self.pending_pre_result -> we assume the results are always handled
                    self.seq_result("setup-pre", t0, Ok(Some(pp)))
                }
            },
        }
    }
}
