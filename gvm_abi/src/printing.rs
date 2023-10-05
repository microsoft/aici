use std::io;

#[allow(dead_code)]
extern "C" {
    fn gvm_host_print(ptr: *const u8, len: u32);
}

#[cfg(not(target_arch = "wasm32"))]
pub type Printer = std::io::Stdout;

#[cfg(target_arch = "wasm32")]
pub struct Printer {}

#[cfg(target_arch = "wasm32")]
impl io::Write for Printer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        unsafe { gvm_host_print(buf.as_ptr(), buf.len() as u32) };
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub fn init_panic() {
    #[cfg(target_arch = "wasm32")]
    std::panic::set_hook(Box::new(|info| {
        let file = info.location().unwrap().file();
        let line = info.location().unwrap().line();
        let col = info.location().unwrap().column();

        let msg = match info.payload().downcast_ref::<&'static str>() {
            Some(s) => *s,
            None => match info.payload().downcast_ref::<String>() {
                Some(s) => &s[..],
                None => "Box<Any>",
            },
        };

        let err_info = format!("Panicked at '{}', {}:{}:{}\n", msg, file, line, col);
        _print(&err_info);
    }))
}

pub fn stdout() -> Printer {
    #[cfg(target_arch = "wasm32")]
    {
        Printer {}
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        io::stdout()
    }
}

pub fn _print(msg: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        let vec: Vec<u8> = msg.into();
        unsafe { gvm_host_print(vec.as_ptr(), vec.len() as u32) };
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::io::Write;
        std::io::stdout().write_all(msg.as_bytes()).unwrap();
    }
}

#[no_mangle]
pub extern "C" fn gvm_init() {
    init_panic();
}
