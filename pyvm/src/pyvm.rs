use aici_abi::{wprintln, AiciVm, MidProcessArg, MidProcessResult};
use anyhow::Result;
use rustpython_vm as vm;

fn _main() -> Result<()> {
    let source = r#"
print("Start")
import aici
aici.hello()
print("STOP")
"#;

    println!("STDOUT!");

    vm::Interpreter::with_init(Default::default(), |vm| {
        vm.add_frozen(rustpython_vm::py_freeze!(dir = "Lib"));
        // vm.add_native_modules(rustpython_stdlib::get_module_inits());
        // vm.add_frozen(rustpython_pylib::FROZEN_STDLIB);
    })
    .enter(|vm| {
        let scope = vm.new_scope_with_builtins();

        let r = vm
            .compile(&source, vm::compiler::Mode::Exec, "<embedded>".to_owned())
            .map_err(|err| vm.new_syntax_error(&err, Some(&source)))
            .and_then(|code_obj| vm.run_code_obj(code_obj, scope));

        match r {
            Ok(_) => {
                println!("OK!");
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
