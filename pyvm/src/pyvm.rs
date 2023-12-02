mod rx;

use aici_abi::{
    svob::SimpleVob, toktree::TokTrie, AiciVm, InitPromptArg, MidProcessArg, MidProcessResult,
    PostProcessArg, PostProcessResult, PreProcessArg, PreProcessResult,
};
use anyhow::Result;

use lazy_static::lazy_static;
use num_traits::ToPrimitive;
use rustpython_vm::{
    builtins::*, compiler::parser::ast::bigint::BigInt, AsObject, PyObjectRef, PyRef, PyResult,
    VirtualMachine,
};
use std::{ops::Deref, sync::Mutex, vec};

struct ModuleState {
    cb_obj: Option<PyObjectRef>,
    trie: TokTrie,
}

unsafe impl Send for ModuleState {}

// Define a global Mutex wrapped in a lazy_static
lazy_static! {
    static ref GLOBAL_STATE: Mutex<ModuleState> = Mutex::new(ModuleState {
        cb_obj: None,
        trie: TokTrie::from_host(),
        // vars: VariableStorage::new(),
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
    use std::sync::Mutex;

    use aici_abi::{
        svob::SimpleVob,
        toktree::{Recognizer, SpecialToken},
        TokenId,
    };
    use once_cell::sync::Lazy;
    use rustpython_derive::pyclass;
    use rustpython_vm::{
        atomic_func,
        builtins::{PyStrRef, PyTypeRef},
        function::ArgStrOrBytesLike,
        protocol::PySequenceMethods,
        types::{AsSequence, Constructor},
        PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine,
    };

    use crate::{VmExt, GLOBAL_STATE};

    #[pyfunction]
    fn register(obj: PyObjectRef, _vm: &VirtualMachine) -> PyResult<()> {
        GLOBAL_STATE.lock().unwrap().cb_obj = Some(obj);
        Ok(())
    }

    #[pyfunction]
    fn tokenize(text: ArgStrOrBytesLike, vm: &VirtualMachine) -> PyResult {
        let tokens = aici_abi::tokenize_bytes(&text.borrow_bytes());
        Ok(vm.new_int_list(&tokens).into())
    }

    #[pyattr]
    #[pyclass(name)]
    #[derive(Debug, PyPayload)]
    pub struct RegexConstraint(pub Mutex<crate::rx::RxStackRecognizer>);

    #[pyclass(with(Constructor))]
    impl RegexConstraint {
        #[pymethod]
        fn eos_allowed(&self) -> bool {
            let mut s = self.0.lock().unwrap();
            s.special_allowed(SpecialToken::EndOfSentence)
        }

        #[pymethod]
        fn token_allowed(&self, t: TokenId) -> bool {
            let trie = &mut GLOBAL_STATE.lock().unwrap().trie;
            let mut s = self.0.lock().unwrap();
            trie.token_allowed(&mut *s, t)
        }

        #[pymethod]
        fn append_token(&self, t: TokenId) {
            let trie = &mut GLOBAL_STATE.lock().unwrap().trie;
            let mut s = self.0.lock().unwrap();
            trie.append_token(&mut *s, t)
        }

        #[pymethod]
        fn allow_tokens(&self, ts: PyRef<TokenSet>) {
            let trie = &mut GLOBAL_STATE.lock().unwrap().trie;
            let mut s = self.0.lock().unwrap();
            let mut ts = ts.0.lock().unwrap();
            trie.compute_bias(&mut *s, &mut *ts);
        }
    }

    impl Constructor for RegexConstraint {
        type Args = (PyStrRef,);
        fn py_new(cls: PyTypeRef, arg: Self::Args, vm: &VirtualMachine) -> PyResult {
            let v = crate::rx::RecRx::from_rx(arg.0.as_str());
            RegexConstraint(Mutex::new(v.to_stack_recognizer()))
                .into_ref_with_type(vm, cls)
                .map(Into::into)
        }
    }

    #[pyattr]
    #[pyclass(name)]
    #[derive(Debug, PyPayload)]
    pub struct TokenSet(pub Mutex<SimpleVob>);

    #[pyclass(with(Constructor, AsSequence))]
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
    let mut runner = Runner::new(source.as_bytes().to_vec());

    runner.init_prompt(InitPromptArg {
        prompt: vec![1, 2, 3],
    });
    runner.init_prompt(InitPromptArg {
        prompt: vec![1, 2, 3],
    });

    Ok(())
}

#[no_mangle]
pub extern "C" fn aici_main(p: *mut Runner) {
    let runner = unsafe { &mut *p };
    let _ = runner;
    // runner.init_prompt(InitPromptArg {
    //     prompt: vec![1, 2, 3],
    // });
}

fn main() {
    _main().unwrap();
}

pub struct Runner {
    interpreter: rustpython_vm::Interpreter,
}

impl Runner {
    pub fn new(arg: Vec<u8>) -> Self {
        let source = String::from_utf8(arg).unwrap();
        let interpreter = rustpython_vm::Interpreter::with_init(Default::default(), |vm| {
            vm.add_native_module("_aici".to_owned(), Box::new(_aici::make_module));
            vm.add_frozen(rustpython_vm::py_freeze!(dir = "Lib"));
            vm.add_frozen(rustpython_vm::py_freeze!(dir = "aici-pylib"));
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

trait VmExt {
    fn get_vm(&self) -> &VirtualMachine;

    fn catch_exn<T>(&self, r: PyResult<T>) -> T {
        match r {
            Ok(v) => v,
            Err(e) => {
                let vm = self.get_vm();
                vm.print_exception(e.clone());
                panic!("Python Exception: {e:?}");
            }
        }
    }

    fn attr(&self, obj: &PyObjectRef, name: &'static str) -> PyObjectRef {
        self.catch_exn(obj.get_attr(name, self.get_vm()))
    }

    fn bool_attr(&self, obj: &PyObjectRef, name: &'static str) -> bool {
        self.to_bool_strict(self.attr(obj, name))
    }

    fn int_attr(&self, obj: &PyObjectRef, name: &'static str) -> i32 {
        self.to_i32(self.attr(obj, name))
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

    fn to_f64(&self, obj: PyObjectRef) -> f64 {
        let vm = self.get_vm();
        let v = obj.to_number().float(vm).expect("expecting float");
        self.catch_exn(v).to_f64()
    }

    fn to_list<F, R>(&self, obj: PyObjectRef, mut f: F) -> Vec<R>
    where
        F: FnMut(PyObjectRef) -> R,
    {
        let vm = self.get_vm();
        obj.payload_if_exact::<PyList>(vm)
            .expect("expecting list")
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

impl AiciVm for Runner {
    fn init_prompt(&mut self, arg: InitPromptArg) {
        let obj = get_cb_obj();
        self.interpreter.enter(|vm| {
            let lst = vm.new_int_list(&arg.prompt);
            vm.catch_exn(vm.call_method(obj.deref(), "init_prompt", vec![lst.into()]));
        });
    }

    fn pre_process(&mut self, _arg: PreProcessArg) -> PreProcessResult {
        let obj = get_cb_obj();
        self.interpreter.enter(|vm| {
            let r = vm.catch_exn(vm.call_method(obj.deref(), "pre_process", vec![]));
            let suspend = vm.bool_attr(&r, "suspended");
            let attention_masks = vm.to_list(vm.attr(&r, "attention_masks"), |v| {
                vm.to_list(v, |v| vm.to_f64(v) as f32)
            });
            PreProcessResult {
                attention_masks,
                suspend,
            }
        })
    }

    fn mid_process(&mut self, arg: MidProcessArg) -> MidProcessResult {
        let obj = get_cb_obj();
        self.interpreter.enter(|vm| {
            let fork_group = vm.new_int_list(&arg.fork_group.iter().map(|v| v.0.clone()).collect());
            let r =
                vm.catch_exn(vm.call_method(obj.deref(), "mid_process", vec![fork_group.into()]));
            let stop = vm.bool_attr(&r, "stop");
            if stop {
                MidProcessResult::Stop
            } else {
                let backtrack = vm.int_attr(&r, "backtrack") as u32;
                let ff_tokens = vm.to_list(vm.attr(&r, "ff_tokens"), |v| vm.to_i32(v) as u32);

                if backtrack > 0 || ff_tokens.len() > 0 {
                    MidProcessResult::Splice {
                        backtrack,
                        ff_tokens,
                    }
                } else {
                    let logit_bias = vm.attr(&r, "logit_bias");
                    let v = logit_bias
                        .payload_if_exact::<_aici::TokenSet>(vm)
                        .expect("expecting TokenSet as logit_bias");
                    let bias = v.0.lock().unwrap();
                    aici_abi::return_logit_bias(&bias);
                    MidProcessResult::SampleWithBias {
                        allowed_tokens: SimpleVob::new(),
                    }
                }
            }
        })
    }

    fn post_process(&mut self, arg: PostProcessArg) -> PostProcessResult {
        let obj = get_cb_obj();
        self.interpreter.enter(|vm| {
            let tokens = vm.new_int_list(&arg.tokens);
            let backtrack = vm.ctx.new_int(arg.backtrack as i32);
            let _ignore = vm.catch_exn(vm.call_method(
                obj.deref(),
                "post_process",
                vec![backtrack.into(), tokens.into()],
            ));
            PostProcessResult {}
        })
    }
}

fn runner_from_env() -> Runner {
    Runner::new(aici_abi::arg_bytes())
}

aici_abi::aici_expose_all!(Runner, runner_from_env());
