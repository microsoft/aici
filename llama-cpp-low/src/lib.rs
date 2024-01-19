#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

extern crate link_cplusplus;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

fn foo() {
    let x: llama_batch;
}
