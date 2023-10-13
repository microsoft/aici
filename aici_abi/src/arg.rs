#[allow(dead_code)]
extern "C" {
    fn aici_host_read_arg(ptr: *mut u8, len: u32) -> u32;
}

pub fn arg_bytes() -> Vec<u8> {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        let size = aici_host_read_arg(0 as _, 0);
        let mut buffer = vec![0u8; size as usize];
        aici_host_read_arg(buffer.as_mut_ptr(), size);
        return buffer;
    }

    #[cfg(not(target_arch = "wasm32"))]
    std::fs::read("arg.json").unwrap()
}
