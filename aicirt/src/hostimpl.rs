use aici_abi::bytes::clone_vec_as_bytes;
use anyhow::Result;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::moduleinstance::ModuleData;

fn read_caller_mem(caller: &wasmtime::Caller<'_, ModuleData>, ptr: u32, len: u32) -> Vec<u8> {
    let mem = caller.data().memory.unwrap();
    let ptr = ptr as usize;
    Vec::from(&mem.data(&caller)[ptr..(ptr + len as usize)])
}

fn write_caller_mem(
    caller: &mut wasmtime::Caller<'_, ModuleData>,
    ptr: u32,
    len: u32,
    src: &[u8],
) -> u32 {
    if len > 0 {
        let mem = caller.data().memory.unwrap();
        let min_len = std::cmp::min(len as usize, src.len());
        mem.write(caller, ptr as usize, &src[..min_len]).unwrap();
    }
    src.len() as u32
}

pub fn setup_linker(engine: &wasmtime::Engine) -> Result<Arc<wasmtime::Linker<ModuleData>>> {
    let mut linker = wasmtime::Linker::<ModuleData>::new(engine);
    linker.func_wrap(
        "env",
        "aici_host_print",
        |mut caller: wasmtime::Caller<'_, ModuleData>, ptr: u32, len: u32| {
            let m = read_caller_mem(&caller, ptr, len);
            caller.data_mut().log.extend_from_slice(&m);
        },
    )?;

    // uint32_t aici_host_read_token_trie(uint8_t *dst, uint32_t size);
    linker.func_wrap(
        "env",
        "aici_host_read_token_trie",
        |mut caller: wasmtime::Caller<'_, ModuleData>, ptr: u32, len: u32| {
            let lock = caller.data().globals.clone();
            let info = lock.read().unwrap();
            write_caller_mem(&mut caller, ptr, len, &info.trie_bytes)
        },
    )?;

    // uint32_t aici_host_read_arg(uint8_t *dst, uint32_t size);
    linker.func_wrap(
        "env",
        "aici_host_read_arg",
        |mut caller: wasmtime::Caller<'_, ModuleData>, ptr: u32, len: u32| {
            let arg = caller.data().module_arg.clone();
            write_caller_mem(&mut caller, ptr, len, arg.as_bytes())
        },
    )?;

    // uint32_t aici_host_tokenize(const uint8_t *src, uint32_t src_size, uint32_t *dst, uint32_t dst_size);
    linker.func_wrap(
        "env",
        "aici_host_tokenize",
        |mut caller: wasmtime::Caller<'_, ModuleData>,
         src: u32,
         src_size: u32,
         dst: u32,
         dst_size: u32| {
            if caller.data().tokenizer.is_none() {
                let lock = caller.data().globals.clone();
                let info = lock.read().unwrap();
                let tok = Tokenizer::from_bytes(info.hf_tokenizer_bytes).unwrap();
                caller.data_mut().tokenizer = Some(tok);
            };
            let m = read_caller_mem(&caller, src, src_size);
            let s = String::from_utf8_lossy(&m);
            let tokens = caller.data().tokenizer.as_ref().unwrap().encode(s, false);
            match tokens {
                Err(_) => 0,
                Ok(tokens) => {
                    let bytes = clone_vec_as_bytes(&tokens.get_ids());
                    write_caller_mem(&mut caller, dst, 4 * dst_size, &bytes) / 4
                }
            }
        },
    )?;

    let linker = Arc::new(linker);
    Ok(linker)
}
