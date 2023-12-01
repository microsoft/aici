use aici_abi::{wprintln, AiciVm, MidProcessArg, MidProcessResult};
use anyhow::Result;

use lazy_static::lazy_static;
use rustpython_vm::PyObjectRef;
use std::{ops::Deref, sync::Mutex, vec};

struct VmImpl {
    cb_obj: Option<PyObjectRef>,
}

unsafe impl Send for VmImpl {}

// Define a global Mutex wrapped in a lazy_static
lazy_static! {
    static ref GLOBAL_STATE: Mutex<VmImpl> = Mutex::new(VmImpl { cb_obj: None });
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
    let source = r#"
print("Start")
import aici
aici.hello()
print("STOP")
"#;

    println!("STDOUT!");

    rustpython_vm::Interpreter::with_init(Default::default(), |vm| {
        vm.add_native_module("_aici".to_owned(), Box::new(_aici::make_module));
        vm.add_frozen(rustpython_vm::py_freeze!(dir = "Lib"));
        // vm.add_native_modules(rustpython_stdlib::get_module_inits());
        // vm.add_frozen(rustpython_pylib::FROZEN_STDLIB);
    })
    .enter(|vm| {
        let scope = vm.new_scope_with_builtins();

        let r = vm
            .compile(
                &source,
                rustpython_vm::compiler::Mode::Exec,
                "<embedded>".to_owned(),
            )
            .map_err(|err| vm.new_syntax_error(&err, Some(&source)))
            .and_then(|code_obj| vm.run_code_obj(code_obj, scope));

        match r {
            Ok(_) => {
                let obj = GLOBAL_STATE.lock().unwrap().cb_obj.clone();
                if obj.is_none() {
                    eprintln!("No callback registered!");
                    return Ok(());
                }
                let obj = obj.unwrap();
                let i1 = vm.ctx.new_int(1);
                let lst = vm.ctx.new_list(vec![i1.into()]);
                let m = vm.call_method(obj.deref(), "init_prompt", vec![lst.into()]);
                println!("OK! {m:?}");
            }
            Err(e) => {
                eprintln!("Exn! {:?}", e);
                vm.print_exception(e);
            }
        }

        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn aici_main() {
    if let Err(e) = _main() {
        wprintln!("Error: {:?}", e);
    }
}

fn main() {
    _main().unwrap();
}

pub struct Runner {}

impl Runner {
    pub fn new(_arg: Vec<u8>) -> Self {
        Self {}
    }
}

impl AiciVm for Runner {
    fn mid_process(&mut self, _arg: MidProcessArg) -> MidProcessResult {
        MidProcessResult::Stop
    }
}

fn runner_from_env() -> Runner {
    Runner::new(aici_abi::arg_bytes())
}

aici_abi::aici_expose_all!(Runner, runner_from_env());
