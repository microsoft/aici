use std::{io, panic};

extern "C" {
    fn gvm_host_print(ptr: *const u8, len: u32);
}

pub struct Printer {}

impl io::Write for Printer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        unsafe { gvm_host_print(buf.as_ptr(), buf.len() as u32) };
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub fn init() {
    panic::set_hook(Box::new(|info| {
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
    Printer {}
}

pub fn _print(msg: &str) {
    let vec: Vec<u8> = msg.into();
    unsafe { gvm_host_print(vec.as_ptr(), vec.len() as u32) };
}

#[no_mangle]
pub extern "C" fn gvm_init() {
    init();
}
