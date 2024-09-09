use aici_abi::{
    aici_stop, export, tokenizer,
    toktrie::{Recognizer, SpecialToken, TokTrie},
    AiciCtrl, Branch, ExportedProgram, Guest, InitPromptArg, InitPromptResult, MidProcessArg,
    MidProcessResult, Program, SimpleVob, Splice, TokenId,
};
use anyhow::Result;
use lazy_static::lazy_static;
use num_traits::ToPrimitive;
use rustpython_vm::{
    builtins::*, compiler::parser::ast::bigint::BigInt, AsObject, PyObjectRef, PyRef, PyResult,
    VirtualMachine,
};
use std::{
    ops::Deref,
    sync::{Arc, Mutex},
    vec,
};

struct ModuleState {
    cb_obj: Option<PyObjectRef>,
    trie: TokTrie,
}

unsafe impl Send for ModuleState {}

lazy_static! {
    static ref GLOBAL_STATE: Mutex<ModuleState> = Mutex::new(ModuleState {
        cb_obj: None,
        trie: TokTrie::from_bytes(&tokenizer::token_trie_bytes()),
        // tokens: vec![],
        // bytes: vec![],
    });
}

fn get_cb_obj() -> PyObjectRef {
    let obj = GLOBAL_STATE.lock().unwrap().cb_obj.clone();
    if obj.is_none() {
        panic!("_aici.register() not called!");
    }
    obj.unwrap()
}

#[rustpython_derive::pymodule]
mod _aici {
    use crate::{ConstraintWrapper, PyConstraint, VmExt, GLOBAL_STATE};
    use aici_abi::{
        cfg::CfgParser,
        dlex::{self, DynamicLexerRec},
        recognizer::{AnythingGoes, StackRecognizer},
        runtime,
        rx::RecRx,
        substring::SubStrMatcher,
        tokenizer,
        toktrie::SpecialToken,
        SimpleVob, TokenId,
    };
    use once_cell::sync::Lazy;
    use rustpython_derive::pyclass;
    use rustpython_vm::{
        atomic_func,
        builtins::{PyStrRef, PyTypeRef},
        function::{ArgStrOrBytesLike, FuncArgs},
        protocol::PySequenceMethods,
        types::{AsSequence, Constructor, Representable},
        Py, PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine,
    };
    use std::{
        fmt::Debug,
        sync::{Arc, Mutex},
    };

    #[pyfunction]
    fn register(obj: PyObjectRef, _vm: &VirtualMachine) -> PyResult<()> {
        GLOBAL_STATE.lock().unwrap().cb_obj = Some(obj);
        Ok(())
    }

    #[pyfunction]
    fn self_seq_id() -> runtime::SeqId {
        runtime::sequence_id()
    }

    #[pyfunction]
    fn is_server_side() -> bool {
        true
    }

    #[pyfunction]
    fn tokenize(text: ArgStrOrBytesLike, vm: &VirtualMachine) -> PyResult {
        let tokens = tokenizer::tokenize_bytes(&text.borrow_bytes());
        Ok(vm.new_int_list(&tokens).into())
    }

    #[pyfunction]
    fn detokenize(tokens: PyObjectRef, vm: &VirtualMachine) -> Vec<u8> {
        let tokens = vm.to_list(tokens, |v| vm.to_i32(v) as u32);
        let trie = &mut GLOBAL_STATE.lock().unwrap().trie;
        let bytes = tokens
            .iter()
            .flat_map(|t| trie.token(*t).to_vec())
            .collect();
        bytes
    }

    #[pyfunction]
    fn tokens_repr(tokens: PyObjectRef, vm: &VirtualMachine) -> String {
        let tokens = vm.to_u32_list(tokens);
        let trie = &mut GLOBAL_STATE.lock().unwrap().trie;
        trie.tokens_dbg(&tokens)
    }

    #[pyfunction]
    fn token_repr(token: u32) -> String {
        let trie = &mut GLOBAL_STATE.lock().unwrap().trie;
        trie.token_dbg(token)
    }

    #[pyfunction]
    fn get_var(name: PyStrRef, _vm: &VirtualMachine) -> Option<Vec<u8>> {
        let name = name.as_str();
        let v = aici_abi::runtime_storage::get(name);
        v
    }

    #[pyfunction]
    fn set_var(name: PyStrRef, value: ArgStrOrBytesLike, _vm: &VirtualMachine) {
        let name = name.as_str();
        aici_abi::runtime_storage::set(name, &value.borrow_bytes());
    }

    #[pyfunction]
    fn append_var(name: PyStrRef, value: ArgStrOrBytesLike, _vm: &VirtualMachine) {
        let name = name.as_str();
        aici_abi::runtime_storage::append(name, &value.borrow_bytes());
    }

    #[pyfunction]
    fn eos_token() -> TokenId {
        let trie = &GLOBAL_STATE.lock().unwrap().trie;
        trie.special_token(SpecialToken::EndOfSentence)
    }

    #[pyfunction]
    fn get_config(name: PyStrRef) -> PyResult<i32> {
        let name = name.as_str();
        let v = runtime::get_config(name);
        Ok(v)
    }

    #[pyattr]
    #[pyclass(name)]
    #[derive(PyPayload)]
    pub struct Constraint(pub Mutex<Box<dyn PyConstraint>>);

    impl Debug for Constraint {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Constraint").finish()
        }
    }

    #[pyclass(flags(BASETYPE), with(Constructor))]
    impl Constraint {
        fn new(obj: impl PyConstraint + 'static) -> Self {
            Constraint(Mutex::new(Box::new(obj)))
        }

        #[pymethod]
        fn eos_allowed(&self) -> bool {
            let mut s = self.0.lock().unwrap();
            s.eos_allowed()
        }

        #[pymethod]
        fn eos_forced(&self) -> bool {
            let mut s = self.0.lock().unwrap();
            s.eos_forced()
        }

        #[pymethod]
        fn token_allowed(&self, t: TokenId) -> bool {
            let mut s = self.0.lock().unwrap();
            s.token_allowed(t)
        }

        #[pymethod]
        fn append_token(&self, t: TokenId) {
            let mut s = self.0.lock().unwrap();
            s.append_token(t)
        }

        #[pymethod]
        fn allow_tokens(&self, ts: PyRef<TokenSet>) {
            let mut s = self.0.lock().unwrap();
            let mut ts = ts.0.lock().unwrap();
            s.allow_tokens(&mut *ts);
        }
    }

    #[pyattr]
    #[pyclass(name)]
    #[derive(PyPayload)]
    pub struct DynamicLexer(pub Arc<Mutex<DynamicLexerRec>>);

    impl Debug for DynamicLexer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("DynamicLexer").finish()
        }
    }

    #[pyclass(with(Constructor))]
    impl DynamicLexer {
        #[pymethod]
        fn add(&self, word: PyStrRef) {
            let mut lexer = self.0.lock().unwrap();
            lexer.recognizer_mut().add(word.as_str().as_bytes());
        }

        #[pymethod]
        fn constraint(&self) -> PyResult<Constraint> {
            Ok(Constraint::new(ConstraintWrapper(self.0.clone())))
        }
    }

    impl Constructor for DynamicLexer {
        type Args = (Option<PyStrRef>,);
        fn py_new(cls: PyTypeRef, arg: Self::Args, vm: &VirtualMachine) -> PyResult {
            let id_chars = match arg.0 {
                Some(id_chars) => id_chars.as_str().chars().collect(),
                None => vec![],
            };
            let lexer = dlex::DynamicLexer::new(&id_chars).to_stack_recognizer();
            DynamicLexer(Arc::new(Mutex::new(lexer)))
                .into_ref_with_type(vm, cls)
                .map(Into::into)
        }
    }

    #[pyfunction(name = "RegexConstraint")]
    fn regex_constraint(regex: PyStrRef, vm: &VirtualMachine) -> PyResult<Constraint> {
        let rx = RecRx::from_rx(regex.as_str(), None)
            .map_err(|e| vm.new_runtime_error(format!("{}", e)))?
            .to_stack_recognizer();
        Ok(Constraint::new(rx))
    }

    #[pyfunction(name = "CfgConstraint")]
    fn cfg_constraint(cfg: PyStrRef, vm: &VirtualMachine) -> PyResult<Constraint> {
        match CfgParser::from_yacc(cfg.as_str()) {
            Ok(cfg) => Ok(Constraint(Mutex::new(Box::new(cfg)))),
            Err(e) => Err(vm.new_runtime_error(format!("{}", e))),
        }
    }

    #[pyfunction(name = "SubStrConstraint")]
    fn substr_constraint(templ: PyStrRef, end_str: PyStrRef) -> PyResult<Constraint> {
        let rx = SubStrMatcher::new(templ.as_str(), end_str.as_str()).to_stack_recognizer();
        Ok(Constraint(Mutex::new(Box::new(rx))))
    }

    impl Constructor for Constraint {
        type Args = FuncArgs;
        fn py_new(cls: PyTypeRef, _arg: Self::Args, vm: &VirtualMachine) -> PyResult {
            let anything = StackRecognizer::from(AnythingGoes {});
            Constraint(Mutex::new(Box::new(anything)))
                .into_ref_with_type(vm, cls)
                .map(Into::into)
        }
    }

    #[pyattr]
    #[pyclass(name)]
    #[derive(Debug, PyPayload)]
    pub struct TokenSet(pub Mutex<SimpleVob>);

    #[pyclass(with(Constructor, AsSequence, Representable))]
    impl TokenSet {
        fn len(&self) -> usize {
            self.0.lock().unwrap().len()
        }
        fn get_at(&self, i: isize) -> Option<bool> {
            let inner = self.0.lock().unwrap();
            if i < 0 || i >= inner.len() as isize {
                None
            } else {
                Some(inner.is_allowed(i as u32))
            }
        }
        fn set_at(&self, i: isize, b: bool) -> Option<()> {
            let mut inner = self.0.lock().unwrap();
            if i < 0 || i >= inner.len() as isize {
                None
            } else {
                if b {
                    inner.allow_token(i as u32);
                } else {
                    inner.disallow_token(i as u32);
                }
                Some(())
            }
        }

        #[pymethod]
        fn set_all(&self, v: bool) {
            let mut inner = self.0.lock().unwrap();
            inner.set_all(v)
        }

        #[pymethod]
        fn num_set(&self) -> usize {
            let inner = self.0.lock().unwrap();
            inner.num_set()
        }
    }

    impl Representable for TokenSet {
        fn repr_str(zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
            let inner = zelf.0.lock().unwrap();
            let trie = &GLOBAL_STATE.lock().unwrap().trie;
            Ok(trie.token_set_dbg(&inner))
        }
    }

    impl AsSequence for TokenSet {
        fn as_sequence() -> &'static PySequenceMethods {
            static AS_SEQUENCE: Lazy<PySequenceMethods> = Lazy::new(|| PySequenceMethods {
                length: atomic_func!(|seq, _vm| Ok(TokenSet::sequence_downcast(seq).len())),
                item: atomic_func!(|seq, i, vm| {
                    TokenSet::sequence_downcast(seq)
                        .get_at(i)
                        .map(|x| vm.ctx.new_bool(x).into())
                        .ok_or_else(|| vm.new_index_error("index out of range".to_owned()))
                }),
                ass_item: atomic_func!(|seq, i, value, vm| {
                    if let Some(value) = value {
                        TokenSet::sequence_downcast(seq)
                            .set_at(i, vm.to_bool_strict(value))
                            .ok_or_else(|| vm.new_index_error("index out of range".to_owned()))
                    } else {
                        Err(vm.new_index_error("can't del".to_owned()))
                    }
                }),
                ..PySequenceMethods::NOT_IMPLEMENTED
            });
            &AS_SEQUENCE
        }
    }

    impl Constructor for TokenSet {
        type Args = ();
        fn py_new(cls: PyTypeRef, _arg: Self::Args, vm: &VirtualMachine) -> PyResult {
            let v = GLOBAL_STATE.lock().unwrap().trie.alloc_token_set();
            TokenSet(Mutex::new(v))
                .into_ref_with_type(vm, cls)
                .map(Into::into)
        }
    }
}

fn _main() -> Result<()> {
    let source = std::fs::read_to_string("samples/test.py").unwrap();
    let mut runner = Runner::new(source);

    runner.init_prompt(InitPromptArg { prompt: vec![1] });

    Ok(())
}

pub struct Runner {
    interpreter: rustpython_vm::Interpreter,
}

impl Runner {
    pub fn new(source: String) -> Self {
        let interpreter = rustpython_vm::Interpreter::with_init(Default::default(), |vm| {
            vm.add_native_module(
                "pyaici.server_native".to_owned(),
                Box::new(_aici::make_module),
            );
            vm.add_frozen(rustpython_vm::py_freeze!(dir = "Lib"));

            let code = rustpython_vm::py_compile!(
                file = "../../py/pyaici/server.py",
                module_name = "pyaici.server",
                mode = "exec"
            );
            let empty = rustpython_vm::py_compile!(
                source = "# nothing",
                module_name = "pyaici",
                mode = "exec"
            );
            let frozen_vec = vec![
                (
                    "pyaici",
                    rustpython_vm::frozen::FrozenModule {
                        code: empty,
                        package: true,
                    },
                ),
                (
                    "pyaici.server",
                    rustpython_vm::frozen::FrozenModule {
                        code,
                        package: true,
                    },
                ),
            ];
            vm.add_frozen(frozen_vec.into_iter());
        });
        interpreter.enter(|vm| {
            let scope = vm.new_scope_with_builtins();

            let r = vm
                .compile(
                    &source,
                    rustpython_vm::compiler::Mode::Exec,
                    "<arg>".to_owned(),
                )
                .map_err(|err| vm.new_syntax_error(&err, Some(&source)))
                .and_then(|code_obj| vm.run_code_obj(code_obj, scope));

            match r {
                Ok(_) => {
                    // make sure the callback is registered
                    let _ = get_cb_obj();
                }
                Err(e) => {
                    vm.print_exception(e.clone());
                    panic!("Python Exception: {e:?}");
                }
            }
        });
        Self { interpreter }
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

struct ConstraintWrapper<T: Recognizer>(Arc<Mutex<T>>);
impl<T: Recognizer> PyConstraint for ConstraintWrapper<T> {
    fn eos_allowed(&mut self) -> bool {
        self.0.lock().unwrap().eos_allowed()
    }

    fn eos_forced(&mut self) -> bool {
        self.0.lock().unwrap().eos_forced()
    }

    fn token_allowed(&mut self, t: TokenId) -> bool {
        self.0.lock().unwrap().token_allowed(t)
    }

    fn append_token(&mut self, t: TokenId) {
        self.0.lock().unwrap().append_token(t)
    }

    fn allow_tokens(&mut self, logits: &mut SimpleVob) {
        self.0.lock().unwrap().allow_tokens(logits)
    }
}

trait VmExt {
    fn get_vm(&self) -> &VirtualMachine;

    fn catch_exn<T>(&self, r: PyResult<T>) -> T {
        match r {
            Ok(v) => v,
            Err(e) => {
                let vm = self.get_vm();
                vm.print_exception(e.clone());
                aici_stop();
            }
        }
    }

    fn attr(&self, obj: &PyObjectRef, name: &'static str) -> PyObjectRef {
        self.catch_exn(obj.get_attr(name, self.get_vm()))
    }

    fn bool_attr(&self, obj: &PyObjectRef, name: &'static str) -> bool {
        self.to_bool_strict(self.attr(obj, name))
    }

    fn i32_attr(&self, obj: &PyObjectRef, name: &'static str) -> i32 {
        self.to_i32(self.attr(obj, name))
    }

    fn u32_attr(&self, obj: &PyObjectRef, name: &'static str) -> u32 {
        self.to_u32(self.attr(obj, name))
    }

    fn to_bool_strict(&self, obj: PyObjectRef) -> bool {
        let vm = self.get_vm();
        if obj.is(&vm.ctx.true_value) {
            true
        } else if obj.is(&vm.ctx.false_value) {
            false
        } else {
            panic!("not a bool");
        }
    }

    fn to_i32(&self, obj: PyObjectRef) -> i32 {
        let vm = self.get_vm();
        let v = obj.to_number().int(vm).expect("expecting int");
        self.catch_exn(v)
            .as_bigint()
            .to_i32()
            .expect("expecting i32")
    }

    fn to_u32(&self, obj: PyObjectRef) -> u32 {
        let vm = self.get_vm();
        let v = obj.to_number().int(vm).expect("expecting int");
        self.catch_exn(v)
            .as_bigint()
            .to_u32()
            .expect("expecting u32 (non-negative)")
    }

    fn to_f64(&self, obj: PyObjectRef) -> f64 {
        let vm = self.get_vm();
        let v = obj.to_number().float(vm).expect("expecting float");
        self.catch_exn(v).to_f64()
    }

    fn to_u32_list(&self, obj: PyObjectRef) -> Vec<u32> {
        self.to_list(obj, |v| self.to_u32(v))
    }

    fn to_list<F, R>(&self, obj: PyObjectRef, mut f: F) -> Vec<R>
    where
        F: FnMut(PyObjectRef) -> R,
    {
        let vm = self.get_vm();
        obj.payload_if_exact::<PyList>(vm)
            .unwrap_or_else(|| panic!("expecting list, got {:?}", obj))
            .borrow_vec()
            .iter()
            .map(|x| f(x.clone()))
            .collect::<Vec<_>>()
    }

    fn new_int_list<T: Into<BigInt> + ToPrimitive + Clone>(&self, lst: &Vec<T>) -> PyRef<PyList> {
        let vm = self.get_vm();
        let elts = lst
            .iter()
            .map(|v| vm.ctx.new_int(v.clone()).into())
            .collect();
        vm.ctx.new_list(elts)
    }
}

impl VmExt for VirtualMachine {
    fn get_vm(&self) -> &VirtualMachine {
        self
    }
}

impl AiciCtrl for Runner {
    fn init_prompt(&mut self, arg: InitPromptArg) -> InitPromptResult {
        let obj = get_cb_obj();
        self.interpreter.enter(|vm| {
            let lst = vm.new_int_list(&arg.prompt);
            vm.catch_exn(vm.call_method(obj.deref(), "init_prompt", vec![lst.into()]));
            InitPromptResult::from_arg(arg)
        })
    }

    fn mid_process(&mut self, arg: MidProcessArg) -> MidProcessResult {
        let obj = get_cb_obj();
        self.interpreter.enter(|vm| {
            let fork_group = vm.new_int_list(&arg.fork_group.iter().map(|v| *v as u64).collect());
            let tokens = vm.new_int_list(&arg.tokens);
            let bt = vm.ctx.new_int(arg.backtrack as i32);
            let r = vm.catch_exn(vm.call_method(
                obj.deref(),
                "mid_process",
                vec![bt.into(), tokens.into(), fork_group.into()],
            ));

            let branches = vm.to_list(vm.attr(&r, "branches"), |b| {
                let logit_bias = vm.attr(&b, "sample_mask");
                let sample_mask = if vm.is_none(&logit_bias) {
                    None
                } else {
                    let v = logit_bias
                        .payload_if_exact::<_aici::TokenSet>(vm)
                        .expect("expecting TokenSet as sample_mask");
                    let bias = v.0.lock().unwrap();
                    Some(bias.clone().into())
                };
                let splices = vm.to_list(vm.attr(&b, "splices"), |s| {
                    let backtrack = vm.u32_attr(&s, "backtrack");
                    let ff_tokens = vm.to_u32_list(vm.attr(&s, "ff_tokens"));
                    let when_sampled = vm.to_u32_list(vm.attr(&s, "when_sampled"));
                    Splice {
                        when_sampled,
                        backtrack,
                        ff_tokens,
                    }
                });

                Branch {
                    sample_mask,
                    temperature: None,
                    splices,
                }
            });

            MidProcessResult { branches }
        })
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
