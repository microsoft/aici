use std::env;
use std::path::PathBuf;

const SUBMODULE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/llama.cpp");

fn main() {
    let ccache = true;
    let cuda = false;

    let submodule_dir = &PathBuf::from(SUBMODULE_DIR);
    let header_path = submodule_dir.join("llama.h");

    if !submodule_dir.join("CMakeLists.txt").exists() {
        eprintln!("did you run 'git submodules init --update' ?");
        std::process::exit(1);
    }

    let mut cmake = cmake::Config::new(&submodule_dir);
    // cmake
    //     .configure_arg("-DLLAMA_STATIC=OFF")
    //     .configure_arg("-DLLAMA_BUILD_EXAMPLES=OFF")
    //     .configure_arg("-DLLAMA_BUILD_SERVER=OFF")
    //     .configure_arg("-DLLAMA_BUILD_TESTS=OFF");

    if ccache {
        cmake
            .configure_arg("-DCMAKE_C_COMPILER_LAUNCHER=ccache")
            .configure_arg("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
            .configure_arg("-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache");
    }

    if cuda {
        cmake.configure_arg("-DLLAMA_CUBLAS=ON");
        println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cupti");
    }

    let dst = cmake.build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=llama");

    let bindings = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
        .parse_callbacks(Box::new(
            bindgen::CargoCallbacks::new().rerun_on_header_files(false),
        ))
        .generate_comments(true)
        .layout_tests(false)
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        //.allowlist_function("ggml_.*")
        //.allowlist_type("ggml_.*")
        .opaque_type("FILE")
        .clang_arg("-xc++")
        .clang_arg("-fparse-all-comments")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
