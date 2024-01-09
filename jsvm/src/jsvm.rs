use std::sync::Mutex;

use aici_abi::{
    svob::SimpleVob,
    toktree::{Recognizer, SpecialToken, TokTrie},
    TokenId, VariableStorage,
};
use rquickjs::{
    class::Trace,
    function::{IntoJsFunc, ParamRequirement, Params},
    qjs, Context, Ctx, Function, IntoJs, Module, Object, Runtime, Value,
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

// TODO do something nicer for objects and arrays
fn val_to_str(v: &Value) -> String {
    let mut len = std::mem::MaybeUninit::uninit();
    let ptr =
        unsafe { qjs::JS_ToCStringLen(v.ctx().as_raw().as_ptr(), len.as_mut_ptr(), v.as_raw()) };
    if ptr.is_null() {
        return "???".to_string();
    }
    let len = unsafe { len.assume_init() };
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(ptr as _, len as _) };
    let result = String::from_utf8_lossy(bytes).to_string();
    unsafe { qjs::JS_FreeCString(v.ctx().as_raw().as_ptr(), ptr) };
    result
}

struct ConsoleLog;
impl<'js> IntoJsFunc<'js, ()> for ConsoleLog {
    fn param_requirements() -> ParamRequirement {
        ParamRequirement::any()
    }

    fn call<'a>(&self, params: Params<'a, 'js>) -> rquickjs::Result<Value<'js>> {
        for idx in 0..params.len() {
            let v = params.arg(idx).unwrap();
            print!("{}", val_to_str(&v));
            if idx < params.len() - 1 {
                print!(" ");
            }
        }
        println!("");
        ().into_js(params.ctx())
    }
}

fn error_to_string(ctx: &Ctx, e: rquickjs::Error) -> String {
    match e {
        rquickjs::Error::Exception => {
            let v = ctx.catch();
            match v.as_exception() {
                Some(e) if e.message().is_some() => format!(
                    "Exception: {}\n{}",
                    e.message().unwrap(),
                    e.stack().unwrap_or(String::new())
                ),
                _ => format!("{v:?}"),
            }
        }
        _ => format!("{e}"),
    }
}

fn unwrap_js<T>(ctx: &Ctx, result: Result<T, rquickjs::Error>) -> T {
    match result {
        Ok(r) => r,
        Err(e) => {
            println!("{}", error_to_string(ctx, e));
            std::process::exit(1)
        }
    }
}

/// A class which will be exported from the module.
#[rquickjs::class]
pub struct TokenSet {
    inner: SimpleVob,
}

impl<'js> Trace<'js> for TokenSet {
    fn trace<'a>(&self, _tracer: rquickjs::class::Tracer<'a, 'js>) {
        // do nothing
    }
}

#[rquickjs::methods]
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

    pub fn set_all(&mut self, val: bool) {
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
impl Constraint {
    pub fn eos_allowed(&mut self) -> bool {
        self.inner.eos_allowed()
    }

    pub fn eos_forced(&mut self) -> bool {
        self.inner.eos_forced()
    }

    pub fn token_allowed(&mut self, t: TokenId) -> bool {
        self.inner.token_allowed(t)
    }

    pub fn append_token(&mut self, t: TokenId) {
        self.inner.append_token(t)
    }

    pub fn allow_tokens(&mut self, ts: &mut TokenSet) {
        self.inner.allow_tokens(&mut ts.inner);
    }
}

#[rquickjs::module]
mod aici_mod {
    pub use super::{Constraint, TokenSet};

    use super::GLOBAL_STATE;
    use aici_abi::{
        cfg::CfgParser, rx::RecRx, substring::SubStrMatcher, toktree::SpecialToken, TokenId,
    };
    use rquickjs::{Ctx, Exception, Result};

    type StrOrBuffer = String; // TODO

    #[rquickjs::function]
    pub fn self_seq_id() -> u32 {
        aici_abi::self_seq_id().0
    }

    #[rquickjs::function]
    pub fn tokenize(text: StrOrBuffer) -> Vec<TokenId> {
        aici_abi::tokenize_bytes(&text.as_bytes())
    }

    #[rquickjs::function]
    fn detokenize(tokens: Vec<TokenId>) -> Vec<u8> {
        let trie = &mut GLOBAL_STATE.lock().unwrap().trie;
        let bytes = tokens
            .iter()
            .flat_map(|t| trie.token(*t).to_vec())
            .collect();
        bytes
    }

    #[rquickjs::function]
    fn get_var(name: String) -> Option<Vec<u8>> {
        let name = name.as_str();
        let v = GLOBAL_STATE.lock().unwrap().vars.get(name);
        v
    }

    #[rquickjs::function]
    fn set_var(name: String, value: StrOrBuffer) {
        let name = name.as_str();
        let vars = &GLOBAL_STATE.lock().unwrap().vars;
        vars.set(name, (&value.as_bytes()).to_vec());
    }

    #[rquickjs::function]
    fn append_var(name: String, value: StrOrBuffer) {
        let name = name.as_str();
        let vars = &GLOBAL_STATE.lock().unwrap().vars;
        vars.append(name, (&value.as_bytes()).to_vec());
    }

    #[rquickjs::function]
    fn eos_token() -> TokenId {
        let trie = &GLOBAL_STATE.lock().unwrap().trie;
        trie.special_token(SpecialToken::EndOfSentence)
    }

    #[rquickjs::function]
    #[qjs(rename = "RegexConstraint")]
    fn regex_constraint(regex: String) -> Constraint {
        let rx = RecRx::from_rx(regex.as_str()).to_stack_recognizer();
        Constraint::new(Box::new(rx))
    }

    #[rquickjs::function]
    #[qjs(rename = "CfgConstraint")]
    fn cfg_constraint<'js>(ctx: Ctx<'js>, cfg: String) -> Result<Constraint> {
        match CfgParser::from_yacc(cfg.as_str()) {
            Ok(cfg) => Ok(Constraint::new(Box::new(cfg))),
            Err(e) => Err(Exception::throw_type(&ctx, &format!("{}", e))),
        }
    }

    #[rquickjs::function]
    #[qjs(rename = "SubStrConstraint")]
    fn substr_constraint(templ: String, end_str: String) -> Constraint {
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
    interpreter: Context,
}

impl Runner {
    pub fn new(arg: Vec<u8>) -> Self {
        let source = String::from_utf8(arg).unwrap();

        let rt = Runtime::new().unwrap();
        let ctx = Context::full(&rt).unwrap();

        let print = r#"
import * as _aici from '_aici'
console.info('Hello', {})
const t = _aici.TokenSet()
console.info('Hello', t, t.length)
export function foo() {
    console.log("foo")
}
"#;

        let aici_js = include_str!("../ts/aici.js");

        ctx.with(|ctx| {
            let global = ctx.globals();
            let cons = Object::new(ctx.clone()).unwrap();
            let f = Function::new(ctx.clone(), ConsoleLog).unwrap();
            cons.set("log", f.clone()).unwrap();
            cons.set("info", f.clone()).unwrap();
            cons.set("warn", f.clone()).unwrap();
            cons.set("error", f.clone()).unwrap();
            cons.set("debug", f.clone()).unwrap();
            global.set("console", cons).unwrap();

            Module::declare_def::<js_aici_mod, _>(ctx.clone(), "_aici").unwrap();

            let _ = unwrap_js(&ctx, ctx.clone().compile("main", print));
            // let v: Value = unwrap_js(&ctx, ctx.clone().eval("globalThis.foo"));
            let _ = unwrap_js(&ctx, ctx.clone().compile("aici", aici_js));
            let _ = unwrap_js(&ctx, ctx.clone().compile("main", source));
        });

        Self { interpreter: ctx }
    }
}
