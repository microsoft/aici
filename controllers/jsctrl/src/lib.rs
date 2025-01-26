use aici_abi::{
    aici_stop, export,
    recognizer::{AnythingGoes, StackRecognizer},
    tokenizer,
    toktrie::{Recognizer, SpecialToken, TokTrie},
    AiciCtrl, ExportedProgram, Guest, InitPromptArg, InitPromptResult, MidProcessArg,
    MidProcessResult, Program, SimpleVob, TokenId,
};
use rquickjs::{
    class::Trace, function::IntoArgs, ArrayBuffer, Context, Ctx, FromJs, Function, IntoAtom,
    IntoJs, Module, Object, Result, Runtime, TypedArray, Value,
};
use std::sync::Mutex;

struct ModuleState {
    trie: TokTrie,
    mid_process_result: Option<MidProcessResult>,
}

unsafe impl Send for ModuleState {}

lazy_static::lazy_static! {
    static ref GLOBAL_STATE: Mutex<ModuleState> = Mutex::new(ModuleState {
        trie: TokTrie::from_bytes(&tokenizer::token_trie_bytes()),
        mid_process_result: None,
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
        let trie = &mut GLOBAL_STATE.lock().unwrap().trie;
        TokenSet {
            inner: trie.alloc_token_set(),
        }
    }

    #[qjs(get)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    pub fn toString(&self) -> String {
        let trie = &mut GLOBAL_STATE.lock().unwrap().trie;
        trie.token_set_dbg(&self.inner)
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

    pub fn numSet(&self) -> usize {
        self.inner.num_set()
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
    use crate::{Buffer, CtxExt, ObjectExt};

    pub use super::{Constraint, TokenSet};

    use super::GLOBAL_STATE;

    use aici_abi::{
        aici_stop, cfg::CfgParser, runtime, rx::RecRx, substring::SubStrMatcher, tokenizer,
        toktrie::SpecialToken, Branch, MidProcessResult, SeqId, Splice, TokenId,
    };
    use rquickjs::{Ctx, Exception, Object, Result, Value};

    #[rquickjs::function]
    pub fn selfSeqId() -> SeqId {
        runtime::sequence_id()
    }

    #[rquickjs::function]
    pub fn tokenize(text: Buffer) -> Vec<TokenId> {
        tokenizer::tokenize_bytes(&text.0)
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
    pub fn tokenRepr(token: TokenId) -> String {
        let trie = &mut GLOBAL_STATE.lock().unwrap().trie;
        trie.token_dbg(token)
    }

    #[rquickjs::function]
    pub fn getVar(name: String) -> Option<Buffer> {
        let name = name.as_str();
        let v = aici_abi::runtime_storage::get(name);
        v.map(Buffer)
    }

    #[rquickjs::function]
    pub fn setVar(name: String, value: Buffer) {
        let name = name.as_str();
        aici_abi::runtime_storage::set(name, &value.0);
    }

    #[rquickjs::function]
    pub fn appendVar(name: String, value: Buffer) {
        let name = name.as_str();
        aici_abi::runtime_storage::append(name, &value.0);
    }

    #[rquickjs::function]
    pub fn _midProcessReturn(obj: Object<'_>) {
        let branches: Vec<Object> = obj.get2("branches");
        let res = MidProcessResult {
            branches: branches
                .into_iter()
                .map(|b| {
                    let sample_mask: Option<TokenSet> = b.get2("sampleMask");
                    let splices: Vec<Object> = b.get2("splices");
                    Branch {
                        sample_mask: sample_mask.map(|ts| ts.inner.into()),
                        temperature: None,
                        splices: splices
                            .into_iter()
                            .map(|s| Splice {
                                when_sampled: s.get2("whenSampled"),
                                ff_tokens: s.get2("ffTokens"),
                                backtrack: s.get2("backtrack"),
                            })
                            .collect(),
                    }
                })
                .collect(),
        };

        let mut st = GLOBAL_STATE.lock().unwrap();
        st.mid_process_result = Some(res);
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
    pub fn regexConstraint<'js>(ctx: Ctx<'js>, regex: String) -> Result<Constraint> {
        println!("regex constraint: {:?}", regex);
        let rx = RecRx::from_rx(regex.as_str(), None)
            .map_err(|e| Exception::throw_type(&ctx, &format!("{}", e)))?
            .to_stack_recognizer();
        Ok(Constraint::new(Box::new(rx)))
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
        trie.append_token(self, t).unwrap()
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
    pub fn new(source: String) -> Self {
        let rt = Runtime::new().unwrap();
        let s = Self {
            context: Context::full(&rt).unwrap(),
        };

        let aici_js = include_str!("../ts/dist/aici.js");

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

impl AiciCtrl for Runner {
    fn init_prompt(&mut self, arg: InitPromptArg) -> InitPromptResult {
        self.with_cb("init_prompt", |ctx| {
            let cb: Function = ctx.eval2("globalThis._aici_cb.init_prompt");
            let _: Value = cb.call2((&arg.prompt,));
            InitPromptResult::from_arg(arg)
        })
    }

    fn mid_process(&mut self, arg: MidProcessArg) -> MidProcessResult {
        self.with_cb("mid_process", |ctx| {
            let cb: Function = ctx.eval2("globalThis._aici_cb.mid_process");
            let fg: Vec<u32> = arg.fork_group.iter().map(|v| *v as u32).collect();
            let _: Value = cb.call2((arg.backtrack, &arg.tokens, &fg));
            ()
        });
        let st = &mut GLOBAL_STATE.lock().unwrap();
        if st.mid_process_result.is_some() {
            st.mid_process_result.take().unwrap()
        } else {
            panic!("mid_process did not return a result")
        }
    }
}

impl Program for Runner {
    fn new(arg: String) -> Self {
        Runner::new(arg)
    }
}

impl Guest for Runner {
    type Runner = ExportedProgram<Runner>;
}

export!(Runner);
