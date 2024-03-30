use crate::{
    api::ModuleInstId,
    hostimpl::{AiciLimits, GlobalInfo, ModuleData, LOGIT_BIAS_ALLOW, LOGIT_BIAS_DISALLOW},
    setup_component_linker,
    shm::Shm,
    worker::{GroupHandle, RtMidProcessArg},
    TimerSet, UserError,
};
use aici_abi::toktree::TokTrie;
use aicirt::{
    api::{AiciMidProcessResultInner, AiciPostProcessResultInner, SequenceResult},
    bintokens::ByteTokenizer,
    bindings::{self, exports::aici::abi::controller::*, InitPromptResult, PreProcessArg},
    user_error,
};
use anyhow::{bail, ensure, Result};
use serde::Deserialize;
use std::{path::Path, sync::Arc, time::Instant};
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
    pub fn deserialize_component(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<wasmtime::component::Component> {
        // TODO: Use type safety to ensure that the input is derived from `Component::serialize` or
        // `Engine::precompile_component`.
        unsafe { wasmtime::component::Component::deserialize_file(&self.engine, path) }
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
    pending_pre_result: Option<PreProcessResult>,
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

        let component_instance = ctx.component_linker.instantiate(&mut store, &component)?;
        let aici = bindings::Aici::new(&mut store, &component_instance)?;
        let runner = aici.aici_abi_controller().runner().call_constructor(&mut store, &module_arg)?;

        Ok(ModuleInstance {
            store,
            aici,
            runner,
            limits: ctx.limits,
            pending_pre_result: None,
        })
    }

    pub fn set_id(&mut self, id: ModuleInstId) {
        self.store.data_mut().id = id;
    }

    pub fn group_channel(&self) -> &GroupHandle {
        &self.store.data().group_channel
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

    fn do_pre_process_inner(
        &mut self,
        &PreProcessArg(): &PreProcessArg,
    ) -> Result<PreProcessResult> {
        self.aici
            .aici_abi_controller()
            .runner()
            .call_pre_process(&mut self.store, self.runner)
    }

    fn do_mid_process(&mut self, op: RtMidProcessArg, shm: &Shm) -> Result<Option<AiciMidProcessResultInner>> {
        let vocab_size = self.store.data().globals.tokrx_info.vocab_size as usize;
        assert!(op.logit_size == vocab_size * 4);
        let logit_ptr = shm.slice_at_byte_offset(op.logit_offset, vocab_size);
        logit_ptr
            .iter_mut()
            .for_each(|x| *x = LOGIT_BIAS_DISALLOW);

        match self.aici.aici_abi_controller().runner().call_mid_process(
            &mut self.store,
            self.runner,
            &op.op,
        )? {
            MidProcessResult::SampleWithBias(SampleWithBias { allowed_tokens }) => {
                for idx in 0..vocab_size {
                    let mask = allowed_tokens.data[idx / 32];
                    let bit = 1 << (idx % 32);
                    if mask & bit != 0 {
                        logit_ptr[idx] = LOGIT_BIAS_ALLOW;
                    }
                }
                Ok(None)
            },
            MidProcessResult::Stop { .. } => {
                let eos = self.store.data().globals.tokrx_info.tok_eos;
                logit_ptr
                    .iter_mut()
                    .for_each(|v| *v = LOGIT_BIAS_DISALLOW);
                logit_ptr[eos as usize] = LOGIT_BIAS_ALLOW;
                Ok(None)
            }
            MidProcessResult::Splice(Splice {
                mut backtrack,
                ff_tokens,
            }) => {
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
                        logit_ptr[t0 as usize] = LOGIT_BIAS_ALLOW;
                    } else {
                        // we don't really care about biases, as we'lre going to backtrack this token anyways
                        // but just in case, allow all
                        logit_ptr
                            .iter_mut()
                            .for_each(|v| *v = LOGIT_BIAS_DISALLOW);
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

    fn do_post_process(&mut self, rtarg: PostProcessArg) -> Result<PostProcessResult> {
        self.aici.aici_abi_controller().runner().call_post_process(
            &mut self.store,
            self.runner,
            &rtarg,
        )
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
            Ok(res) => self.seq_result(
                "post",
                t0,
                Ok(Some(AiciPostProcessResultInner { stop: res.stop })),
            ),
        }
    }

    pub fn tokenize(&mut self, s: &str) -> Result<Vec<u32>> {
        self.store.data_mut().tokenize(s)
    }

    fn setup_inner(&mut self, prompt: Vec<TokenId>) -> Result<InitPromptResult> {
        let res = self.aici.aici_abi_controller().runner().call_init_prompt(
            &mut self.store,
            self.runner,
            &InitPromptArg { prompt },
        )?;
        Ok(res.into())
    }

    pub fn setup(&mut self, prompt: Vec<TokenId>) -> SequenceResult<PreProcessResult> {
        let t0 = Instant::now();
        match self.setup_inner(prompt) {
            Err(err) => self.seq_result("setup", t0, Err(err)),
            Ok(InitPromptResult()) => match self.do_pre_process(PreProcessArg()) {
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
