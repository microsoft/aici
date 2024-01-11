use std::sync::Mutex;

use aici_abi::{
    aici_stop,
    recognizer::{AnythingGoes, StackRecognizer},
    svob::SimpleVob,
    toktree::{Recognizer, SpecialToken, TokTrie},
    AiciCtrl, InitPromptArg, InitPromptResult, MidProcessArg, MidProcessResult, PostProcessArg,
    PostProcessResult, PreProcessArg, PreProcessResult, TokenId, VariableStorage,
};
use rquickjs::{
    class::Trace, function::IntoArgs, ArrayBuffer, Context, Ctx, FromJs, Function, IntoAtom,
    IntoJs, Module, Object, Result, Runtime, TypedArray, Value,
};

struct ModuleState {
    trie: TokTrie,
    vars: VariableStorage,
}

unsafe impl Send for ModuleState {}

lazy_static::lazy_static! {
    static ref GLOBAL_STATE: Mutex<ModuleState> = Mutex::new(ModuleState {
        trie: TokTrie::from_host(),
        vars: VariableStorage::new(),
    });
}

trait CtxExt<'js> {
    fn error_value_to_string(&self, v: Value<'js>) -> String;
    fn error_to_string(&self, e: rquickjs::Error) -> String;
    fn unwrap_js<T>(&self, result: Result<T>) -> T;
    fn eval2<V: FromJs<'js>, S: Into<Vec<u8>>>(&self, source: S) -> V;
}

trait ObjectExt<'js> {
    fn get2<K: IntoAtom<'js>, V: FromJs<'js>>(&self, k: K) -> V;
}

trait FunctionExt<'js> {
    fn call2<A, R>(&self, args: A) -> R
    where
        A: IntoArgs<'js>,
        R: FromJs<'js>;
}

impl<'js> FunctionExt<'js> for Function<'js> {
    fn call2<A, R>(&self, args: A) -> R
    where
        A: IntoArgs<'js>,
        R: FromJs<'js>,
    {
        let r = self.ctx().unwrap_js(self.call(args));
        r
    }
}

impl<'js> ObjectExt<'js> for Object<'js> {
    fn get2<K: IntoAtom<'js>, V: FromJs<'js>>(&self, k: K) -> V {
        self.ctx().unwrap_js(self.get(k))
    }
}

impl<'js> CtxExt<'js> for Ctx<'js> {
    fn error_value_to_string(&self, v: Value<'js>) -> String {
        match v.as_exception() {
            Some(e) if e.message().is_some() => format!(
                "Exception: {}\n{}",
                e.message().unwrap(),
                e.stack().unwrap_or(String::new())
            ),
            _ => format!("{v:?}"),
        }
    }

    fn error_to_string(&self, e: rquickjs::Error) -> String {
        match e {
            rquickjs::Error::Exception => self.error_value_to_string(self.catch()),
            _ => format!("{e}"),
        }
    }

    fn unwrap_js<T>(&self, result: Result<T>) -> T {
        match result {
            Ok(r) => r,
            Err(e) => {
                println!("{}", self.error_to_string(e));
                aici_stop();
            }
        }
    }

    fn eval2<V: FromJs<'js>, S: Into<Vec<u8>>>(&self, source: S) -> V {
        self.unwrap_js(self.eval(source))
    }
}

/// A class which will be exported from the module.
#[rquickjs::class]
#[derive(Clone)]
pub struct TokenSet {
    inner: SimpleVob,
}

impl<'js> Trace<'js> for TokenSet {
    fn trace<'a>(&self, _tracer: rquickjs::class::Tracer<'a, 'js>) {
        // do nothing
    }
}

#[rquickjs::methods]
#[allow(non_snake_case)]
impl TokenSet {
    #[qjs(constructor)]
    pub fn new() -> TokenSet {
        // TODO set the size
        let mut inner = SimpleVob::new();
        inner.resize(32000);
        TokenSet { inner }
    }

    #[qjs(get)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    pub fn add(&mut self, tok: u32) {
        self.inner.allow_token(tok);
    }

    pub fn delete(&mut self, tok: u32) {
        self.inner.disallow_token(tok);
    }

    pub fn has(&self, tok: u32) -> bool {
        self.inner.is_allowed(tok)
    }

    pub fn clear(&mut self) {
        self.inner.set_all(false);
    }

    pub fn setAll(&mut self, val: bool) {
        self.inner.set_all(val);
    }
}

impl Default for TokenSet {
    fn default() -> Self {
        Self::new()
    }
}

#[rquickjs::class]
pub struct Constraint {
    inner: Box<dyn PyConstraint>,
}

impl Trace<'_> for Constraint {
    fn trace<'a>(&self, _tracer: rquickjs::class::Tracer<'a, '_>) {
        // do nothing
    }
}

impl Constraint {
    fn new(inner: Box<dyn PyConstraint>) -> Self {
        Self { inner }
    }
}

#[rquickjs::methods]
#[allow(non_snake_case)]
impl Constraint {
    #[qjs(constructor)]
    pub fn ctor() -> Self {
        Self::new(Box::new(StackRecognizer::from(AnythingGoes {})))
    }

    pub fn eosAllowed(&mut self) -> bool {
        self.inner.eos_allowed()
    }

    pub fn eosForced(&mut self) -> bool {
        self.inner.eos_forced()
    }

    pub fn tokenAllowed(&mut self, t: TokenId) -> bool {
        self.inner.token_allowed(t)
    }

    pub fn appendToken(&mut self, t: TokenId) {
        self.inner.append_token(t)
    }

    pub fn allowTokens(&mut self, ts: &mut TokenSet) {
        self.inner.allow_tokens(&mut ts.inner);
    }
}

struct Buffer(Vec<u8>);

impl<'js> FromJs<'js> for Buffer {
    fn from_js(ctx: &Ctx<'js>, v: Value<'js>) -> Result<Self> {
        match TypedArray::<'js, u8>::from_value(v.clone()) {
            Ok(r) => Ok(Buffer(r.as_bytes().unwrap().to_vec())),
            Err(_) => match ArrayBuffer::from_value(v.clone()) {
                Some(r) => Ok(Buffer(r.as_bytes().unwrap().to_vec())),
                None => match String::from_js(ctx, v) {
                    Ok(r) => Ok(Buffer(r.into_bytes())),
                    Err(e) => Err(e),
                },
            },
        }
    }
}

impl<'js> IntoJs<'js> for Buffer {
    fn into_js(self, ctx: &Ctx<'js>) -> Result<Value<'js>> {
        TypedArray::<'js, u8>::new(ctx.clone(), self.0).into_js(ctx)
    }
}

#[rquickjs::module]
#[allow(non_snake_case)]
mod aici_mod {
    use crate::{Buffer, CtxExt};

    pub use super::{Constraint, TokenSet};

    use super::GLOBAL_STATE;
    use aici_abi::{
        aici_stop, cfg::CfgParser, rx::RecRx, substring::SubStrMatcher, toktree::SpecialToken,
        TokenId,
    };
    use rquickjs::{Ctx, Exception, Result, Value};

    #[rquickjs::function]
    pub fn selfSeqId() -> u32 {
        aici_abi::self_seq_id().0
    }

    #[rquickjs::function]
    pub fn tokenize(text: Buffer) -> Vec<TokenId> {
        aici_abi::tokenize_bytes(&text.0)
    }

    #[rquickjs::function]
    pub fn panic<'js>(ctx: Ctx<'js>, err: Value<'js>) {
        println!("panic:\n{}", ctx.error_value_to_string(err));
        aici_stop();
    }

    #[rquickjs::function]
    pub fn detokenize(tokens: Vec<TokenId>) -> Buffer {
        let trie = &mut GLOBAL_STATE.lock().unwrap().trie;
        let bytes = tokens
            .iter()
            .flat_map(|t| trie.token(*t).to_vec())
            .collect();
        Buffer(bytes)
    }

    #[rquickjs::function]
    pub fn getVar(name: String) -> Option<Buffer> {
        let name = name.as_str();
        let v = GLOBAL_STATE.lock().unwrap().vars.get(name);
        v.map(Buffer)
    }

    #[rquickjs::function]
    pub fn setVar(name: String, value: Buffer) {
        let name = name.as_str();
        let vars = &GLOBAL_STATE.lock().unwrap().vars;
        vars.set(name, value.0);
    }

    #[rquickjs::function]
    pub fn appendVar(name: String, value: Buffer) {
        let name = name.as_str();
        let vars = &GLOBAL_STATE.lock().unwrap().vars;
        vars.append(name, value.0);
    }

    #[rquickjs::function]
    pub fn bufferToString(ctx: Ctx<'_>, value: Buffer) -> Value<'_> {
        rquickjs::String::from_str(ctx, &String::from_utf8_lossy(&value.0))
            .unwrap()
            .into()
    }

    #[rquickjs::function]
    pub fn bufferRepr(value: Buffer) -> String {
        match String::from_utf8(value.0) {
            Ok(s) => format!("b{:?}", s),
            Err(err) => {
                let mut res = Vec::new();
                let mut i = 0;
                let buf = err.as_bytes();
                while i < buf.len() {
                    match buf[i] {
                        b'\\' | b'"' => {
                            res.push(b'\\');
                            res.push(buf[i]);
                        }
                        (32..=127) => res.push(buf[i]),
                        _ => {
                            let mut ok = false;
                            for len in 2..=4 {
                                match String::from_utf8(buf[i..i + len].to_vec()) {
                                    Ok(s) => {
                                        res.extend_from_slice(s.as_bytes());
                                        i += len - 1;
                                        ok = true;
                                        break;
                                    }
                                    Err(_) => {}
                                }
                            }
                            if !ok {
                                res.push(b'\\');
                                res.push(b'x');
                                res.push(b'0' + (buf[i] >> 4));
                                res.push(b'0' + (buf[i] & 0xf));
                            }
                        }
                    }
                    i += 1;
                }
                format!("b\"{}\"", String::from_utf8_lossy(&res))
            }
        }
    }

    #[rquickjs::function]
    pub fn stringToBuffer(value: Buffer) -> Buffer {
        // Buffer has an implicit conversion from a string, so we can make this just identity
        value
    }

    #[rquickjs::function]
    pub fn eosToken() -> TokenId {
        let trie = &GLOBAL_STATE.lock().unwrap().trie;
        trie.special_token(SpecialToken::EndOfSentence)
    }

    #[rquickjs::function]
    pub fn regexConstraint(regex: String) -> Constraint {
        println!("regex constraint: {:?}", regex);
        let rx = RecRx::from_rx(regex.as_str()).to_stack_recognizer();
        Constraint::new(Box::new(rx))
    }

    #[rquickjs::function]
    pub fn cfgConstraint<'js>(ctx: Ctx<'js>, cfg: String) -> Result<Constraint> {
        match CfgParser::from_yacc(cfg.as_str()) {
            Ok(cfg) => Ok(Constraint::new(Box::new(cfg))),
            Err(e) => Err(Exception::throw_type(&ctx, &format!("{}", e))),
        }
    }

    #[rquickjs::function]
    pub fn substrConstraint(templ: String, end_str: String) -> Constraint {
        let rx = SubStrMatcher::new(templ.as_str(), end_str.as_str()).to_stack_recognizer();
        Constraint::new(Box::new(rx))
    }
}

fn main() {}

trait PyConstraint {
    fn eos_allowed(&mut self) -> bool;
    fn eos_forced(&mut self) -> bool;
    fn token_allowed(&mut self, t: TokenId) -> bool;
    fn append_token(&mut self, t: TokenId);
    fn allow_tokens(&mut self, logits: &mut SimpleVob);
}

impl<T: Recognizer> PyConstraint for T {
    fn eos_allowed(&mut self) -> bool {
        self.special_allowed(SpecialToken::EndOfSentence)
    }

    fn eos_forced(&mut self) -> bool {
        self.special_allowed(SpecialToken::EndOfSentence)
            && (0..=255).all(|b| !self.byte_allowed(b))
    }

    fn token_allowed(&mut self, t: TokenId) -> bool {
        let trie = &mut GLOBAL_STATE.lock().unwrap().trie;
        trie.token_allowed(self, t)
    }

    fn append_token(&mut self, t: TokenId) {
        let trie = &mut GLOBAL_STATE.lock().unwrap().trie;
        trie.append_token(self, t)
    }

    fn allow_tokens(&mut self, logits: &mut SimpleVob) {
        let trie = &mut GLOBAL_STATE.lock().unwrap().trie;
        trie.compute_bias(self, logits)
    }
}

pub struct Runner {
    context: Context,
}

fn _print(msg: String) {
    println!("{msg}");
}

impl Runner {
    pub fn new(arg: Vec<u8>) -> Self {
        let source = String::from_utf8(arg).unwrap();

        let rt = Runtime::new().unwrap();
        let s = Self {
            context: Context::full(&rt).unwrap(),
        };

        let aici_js = include_str!("../ts/aici.js");

        s.with_cb("_new", |ctx| {
            let global = ctx.globals();
            let cons = Object::new(ctx.clone()).unwrap();
            let f = Function::new(ctx.clone(), _print).unwrap();
            cons.set("_print", f).unwrap();
            global.set("console", cons).unwrap();

            Module::declare_def::<js_aici_mod, _>(ctx.clone(), "_aici").unwrap();

            let _ = ctx.unwrap_js(ctx.clone().compile("aici", aici_js));
            let _ = ctx.unwrap_js(ctx.clone().compile("main", source));
        });

        s
    }

    pub fn with_cb<F, R>(&self, lbl: &str, f: F) -> R
    where
        F: FnOnce(Ctx) -> R,
    {
        let logging = false;
        if logging {
            println!("running {}", lbl);
        }
        let res = self.context.with(f);
        let mut job_cnt = 0;
        loop {
            match self.context.runtime().execute_pending_job() {
                Err(e) => e.0.with(|ctx| {
                    println!(
                        "exception in deferred job:\n{}",
                        ctx.error_to_string(rquickjs::Error::Exception)
                    );
                    aici_stop();
                }),
                Ok(false) => break,
                Ok(true) => {
                    // self.context.with(|ctx| {
                    //     println!("job {} done; {:?}", job_cnt, ctx.catch());
                    // });
                    job_cnt += 1;
                }
            }
        }
        if logging {
            println!("done {lbl}; +{job_cnt} jobs");
        }
        res
    }
}

/*
export interface AiciCallbacks {
  init_prompt(prompt: Token[]): void;
  pre_process(): PreProcessResult;
  mid_process(fork_group: SeqId[]): MidProcessResult;
  post_process(backtrack: number, tokens: Token[]): PostProcessResult;
}
*/

impl AiciCtrl for Runner {
    fn init_prompt(&mut self, arg: InitPromptArg) -> InitPromptResult {
        self.with_cb("init_prompt", |ctx| {
            let cb: Function = ctx.eval2("globalThis._aici_cb.init_prompt");
            let _: Value = cb.call2((&arg.prompt,));
            InitPromptResult::default()
        })
    }

    fn pre_process(&mut self, _arg: PreProcessArg) -> PreProcessResult {
        self.with_cb("pre_process", |ctx| {
            let cb: Function = ctx.eval2("globalThis._aici_cb.pre_process");
            let r: Object = cb.call2(());
            PreProcessResult {
                attention_masks: r.get2("_n_attention_masks"),
                suspend: r.get2("_n_suspended"),
                ff_tokens: r.get2("_n_ff_tokens"),
            }
        })
    }

    fn mid_process(&mut self, arg: MidProcessArg) -> MidProcessResult {
        loop {
            let res = self.with_cb("mid_process", |ctx| {
                let cb: Function = ctx.eval2("globalThis._aici_cb.mid_process");
                let fg: Vec<u32> = arg.fork_group.iter().map(|v| v.0.clone()).collect();

                let r: Object = cb.call2((&fg,));
                let skip_me: bool = r.get2("_n_skip_me");
                let stop: bool = r.get2("_n_stop");
                if skip_me {
                    None
                } else if stop {
                    Some(MidProcessResult::Stop)
                } else {
                    let backtrack: u32 = r.get2("_n_backtrack");
                    let ff_tokens: Vec<TokenId> = r.get2("_n_ff_tokens");

                    if backtrack > 0 || ff_tokens.len() > 0 {
                        Some(MidProcessResult::Splice {
                            backtrack,
                            ff_tokens,
                        })
                    } else {
                        // TODO perf - clone on TokenSet
                        let logit_bias: TokenSet = r.get2("_n_logit_bias");
                        aici_abi::return_logit_bias(&logit_bias.inner);
                        Some(MidProcessResult::SampleWithBias {
                            allowed_tokens: SimpleVob::new(),
                        })
                    }
                }
            });
            match res {
                Some(r) => return r,
                None => {}
            }
        }
    }

    fn post_process(&mut self, arg: PostProcessArg) -> PostProcessResult {
        self.with_cb("post_process", |ctx| {
            let cb: Function = ctx.eval2("globalThis._aici_cb.post_process");
            let r: Object = cb.call2((arg.backtrack, &arg.tokens));
            let stop: bool = r.get2("_n_stop_seq");
            PostProcessResult { stop }
        })
    }
}

fn runner_from_env() -> Runner {
    Runner::new(aici_abi::arg_bytes())
}

aici_abi::aici_expose_all!(Runner, runner_from_env());
