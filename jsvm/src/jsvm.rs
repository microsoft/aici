use std::sync::Mutex;

use aici_abi::{svob::SimpleVob, toktree::TokTrie, VariableStorage};
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

#[rquickjs::module()]
mod aici_mod {
    pub use super::TokenSet;

    /// You can also export functions.
    #[rquickjs::function]
    pub fn foo() -> u32 {
        1 + 1
    }
}

fn main() {
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

let print2 = r#"
import * as main from 'main';
main.foo();
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
        let v: Value = unwrap_js(&ctx, ctx.clone().eval("globalThis.foo"));
        println!("v={:?}", v);

        println!("import script module");
        let _ = unwrap_js(&ctx, ctx.clone().compile("aici.js", aici_js));
    });
}
