use aici_abi::{
    AiciVm, InitPromptArg, MidProcessArg, MidProcessResult, PostProcessArg, PostProcessResult,
    PreProcessArg, PreProcessResult,
};
use anyhow::Result;

use lazy_static::lazy_static;
use num_traits::ToPrimitive;
use rustpython_vm::{
    builtins::*, compiler::parser::ast::bigint::BigInt, AsObject, PyObjectRef, PyRef, PyResult,
    VirtualMachine,
};
use std::{ops::Deref, sync::Mutex, vec};

struct VmImpl {
    cb_obj: Option<PyObjectRef>,
}

unsafe impl Send for VmImpl {}

// Define a global Mutex wrapped in a lazy_static
lazy_static! {
    static ref GLOBAL_STATE: Mutex<VmImpl> = Mutex::new(VmImpl { cb_obj: None });
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
    use rustpython_vm::{PyObjectRef, PyResult, VirtualMachine};

    use crate::GLOBAL_STATE;

    #[pyfunction]
    fn register(obj: PyObjectRef, _vm: &VirtualMachine) -> PyResult<()> {
        GLOBAL_STATE.lock().unwrap().cb_obj = Some(obj);
        Ok(())
    }
}

fn _main() -> Result<()> {
    let source = std::fs::read_to_string("test.py").unwrap();
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
    runner.init_prompt(InitPromptArg {
        prompt: vec![1, 2, 3],
    });
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

impl Runner {
    fn catch_exn<T>(&self, r: PyResult<T>) -> T {
        match r {
            Ok(v) => v,
            Err(e) => self.interpreter.enter(|vm| {
                vm.print_exception(e.clone());
                panic!("Python Exception: {e:?}");
            }),
        }
    }

    fn attr(&self, vm: &VirtualMachine, obj: &PyObjectRef, name: &'static str) -> PyObjectRef {
        self.catch_exn(obj.get_attr(name, vm))
    }

    fn bool_attr(&self, vm: &VirtualMachine, obj: &PyObjectRef, name: &'static str) -> bool {
        self.to_bool_strict(vm, self.attr(vm, obj, name))
    }

    fn int_attr(&self, vm: &VirtualMachine, obj: &PyObjectRef, name: &'static str) -> i32 {
        self.to_i32(vm, self.attr(vm, obj, name))
    }

    fn to_bool_strict(&self, vm: &VirtualMachine, obj: PyObjectRef) -> bool {
        if obj.is(&vm.ctx.true_value) {
            true
        } else if obj.is(&vm.ctx.false_value) {
            false
        } else {
            panic!("not a bool");
        }
    }

    fn to_i32(&self, vm: &VirtualMachine, obj: PyObjectRef) -> i32 {
        let v = obj.to_number().int(vm).expect("expecting int");
        self.catch_exn(v)
            .as_bigint()
            .to_i32()
            .expect("expecting i32")
    }

    fn to_f64(&self, vm: &VirtualMachine, obj: PyObjectRef) -> f64 {
        let v = obj.to_number().float(vm).expect("expecting float");
        self.catch_exn(v).to_f64()
    }

    fn to_list<F, R>(&self, vm: &VirtualMachine, obj: PyObjectRef, mut f: F) -> Vec<R>
    where
        F: FnMut(PyObjectRef) -> R,
    {
        obj.payload_if_exact::<PyList>(vm)
            .expect("expecting list")
            .borrow_vec()
            .iter()
            .map(|x| f(x.clone()))
            .collect::<Vec<_>>()
    }
}

impl AiciVm for Runner {
    fn init_prompt(&mut self, arg: InitPromptArg) {
        let obj = get_cb_obj();
        self.interpreter.enter(|vm| {
            let lst = new_int_list(vm, &arg.prompt);
            self.catch_exn(vm.call_method(obj.deref(), "init_prompt", vec![lst.into()]));
        });
    }

    fn pre_process(&mut self, _arg: PreProcessArg) -> PreProcessResult {
        let obj = get_cb_obj();
        self.interpreter.enter(|vm| {
            let r = self.catch_exn(vm.call_method(obj.deref(), "pre_process", vec![]));
            let suspend = self.bool_attr(vm, &r, "suspend");
            let attention_masks = self.to_list(vm, self.attr(vm, &r, "attention_masks"), |v| {
                self.to_list(vm, v, |v| self.to_f64(vm, v) as f32)
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
            let fork_group =
                new_int_list(vm, &arg.fork_group.iter().map(|v| v.0.clone()).collect());
            let r =
                self.catch_exn(vm.call_method(obj.deref(), "pre_process", vec![fork_group.into()]));
            let stop = self.bool_attr(vm, &r, "stop");
            if stop {
                MidProcessResult::Stop
            } else {
                let backtrack = self.int_attr(vm, &r, "backtrack") as u32;
                let ff_tokens = self.to_list(vm, self.attr(vm, &r, "ff_tokens"), |v| {
                    self.to_i32(vm, v) as u32
                });

                if backtrack > 0 || ff_tokens.len() > 0 {
                    MidProcessResult::Splice {
                        backtrack,
                        ff_tokens,
                    }
                } else {
                    // TODO logit_bias
                    MidProcessResult::Stop
                }
            }
        })
    }

    fn post_process(&mut self, _arg: PostProcessArg) -> PostProcessResult {
        PostProcessResult {}
    }
}

fn new_int_list<T: Into<BigInt> + ToPrimitive + Clone>(
    vm: &VirtualMachine,
    lst: &Vec<T>,
) -> PyRef<PyList> {
    let elts = lst
        .iter()
        .map(|v| vm.ctx.new_int(v.clone()).into())
        .collect();
    vm.ctx.new_list(elts)
}

fn runner_from_env() -> Runner {
    Runner::new(aici_abi::arg_bytes())
}

aici_abi::aici_expose_all!(Runner, runner_from_env());
