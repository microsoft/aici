use std::env;
use std::path::PathBuf;

const SUBMODULE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/llama.cpp");

fn main() {
    let ccache = true;
    let cuda = std::env::var("CARGO_FEATURE_CUDA").unwrap_or(String::new());

    let submodule_dir = &PathBuf::from(SUBMODULE_DIR);
    let header_path = submodule_dir.join("llama.h");

    if !submodule_dir.join("CMakeLists.txt").exists() {
        eprintln!("did you run 'git submodule update --init' ?");
        std::process::exit(1);
    }

    let mut cmake = cmake::Config::new(&submodule_dir);
    cmake
        .configure_arg("-DLLAMA_STATIC=OFF")
        .configure_arg("-DLLAMA_BUILD_EXAMPLES=OFF")
        .configure_arg("-DLLAMA_BUILD_SERVER=OFF")
        .configure_arg("-DLLAMA_BUILD_TESTS=OFF");

    if ccache {
        cmake
            .configure_arg("-DCMAKE_C_COMPILER_LAUNCHER=ccache")
            .configure_arg("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
            .configure_arg("-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache");
    }

    if cuda == "1" {
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

    if env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" {
        // This has no effect: println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET={}", line);
        // so don't bother...
        //
        // let pref = b"CMAKE_OSX_DEPLOYMENT_TARGET:STRING=";
        // fs::read(&dst.join("build/CMakeCache.txt"))
        //     .expect("CMakeCache.txt not found")
        //     .split(|&b| b == b'\n')
        //     .filter(|line| line.starts_with(pref))
        //     .for_each(|line| {
        //         let line = std::str::from_utf8(&line[pref.len()..]).unwrap();
        //         println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET={}", line);
        //     });
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }

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

    std::fs::copy(
        submodule_dir.join("ggml-metal.metal"),
        out_path.join("../../../ggml-metal.metal"),
    ).expect("Couldn't copy ggml-metal.metal");
}
