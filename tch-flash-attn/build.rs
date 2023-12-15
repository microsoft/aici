// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the CANDLE_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::env;
use std::path::PathBuf;
use std::str::FromStr;

const PYTHON_PRINT_PYTORCH_DETAILS: &str = r"
import torch
from torch.utils import cpp_extension
print('LIBTORCH_VERSION:', torch.__version__.split('+')[0])
print('LIBTORCH_CXX11:', torch._C._GLIBCXX_USE_CXX11_ABI)
for include_path in cpp_extension.include_paths():
  print('LIBTORCH_INCLUDE:', include_path)
for library_path in cpp_extension.library_paths():
  print('LIBTORCH_LIB:', library_path)
";

const PYTHON_PRINT_INCLUDE_PATH: &str = r"
import sysconfig
print('PYTHON_INCLUDE:', sysconfig.get_path('include'))
";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinkType {
    Dynamic,
    Static,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Os {
    Linux,
    Macos,
    Windows,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SystemInfo {
    os: Os,
    python_interpreter: PathBuf,
    cxx11_abi: String,
    libtorch_include_dirs: Vec<PathBuf>,
    libtorch_lib_dir: PathBuf,
    link_type: LinkType,
}

fn env_var_rerun(name: &str) -> Result<String, env::VarError> {
    println!("cargo:rerun-if-env-changed={name}");
    env::var(name)
}

impl SystemInfo {
    fn new() -> Result<Self> {
        let os = match env::var("CARGO_CFG_TARGET_OS")
            .expect("Unable to get TARGET_OS")
            .as_str()
        {
            "linux" => Os::Linux,
            "windows" => Os::Windows,
            "macos" => Os::Macos,
            os => anyhow::bail!("unsupported TARGET_OS '{os}'"),
        };
        // Locate the currently active Python binary, similar to:
        // https://github.com/PyO3/maturin/blob/243b8ec91d07113f97a6fe74d9b2dcb88086e0eb/src/target.rs#L547
        let python_interpreter = match os {
            Os::Windows => PathBuf::from("python.exe"),
            Os::Linux | Os::Macos => {
                if env::var_os("VIRTUAL_ENV").is_some() {
                    PathBuf::from("python")
                } else {
                    PathBuf::from("python3")
                }
            }
        };
        let mut libtorch_include_dirs = vec![];

        if true {
            let output = std::process::Command::new(&python_interpreter)
                .arg("-c")
                .arg(PYTHON_PRINT_INCLUDE_PATH)
                .output()
                .with_context(|| format!("error running {python_interpreter:?}"))?;
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                if let Some(path) = line.strip_prefix("PYTHON_INCLUDE: ") {
                    libtorch_include_dirs.push(PathBuf::from(path))
                }
            }
        }

        let mut libtorch_lib_dir = None;
        let cxx11_abi = {
            let output = std::process::Command::new(&python_interpreter)
                .arg("-c")
                .arg(PYTHON_PRINT_PYTORCH_DETAILS)
                .output()
                .with_context(|| format!("error running {python_interpreter:?}"))?;
            let mut cxx11_abi = None;
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                match line.strip_prefix("LIBTORCH_CXX11: ") {
                    Some("True") => cxx11_abi = Some("1".to_owned()),
                    Some("False") => cxx11_abi = Some("0".to_owned()),
                    _ => {}
                }
                if let Some(path) = line.strip_prefix("LIBTORCH_INCLUDE: ") {
                    libtorch_include_dirs.push(PathBuf::from(path))
                }
                if let Some(path) = line.strip_prefix("LIBTORCH_LIB: ") {
                    libtorch_lib_dir = Some(PathBuf::from(path))
                }
            }
            match cxx11_abi {
                Some(cxx11_abi) => cxx11_abi,
                None => anyhow::bail!("no cxx11 abi returned by python {output:?}"),
            }
        };

        let libtorch_lib_dir = libtorch_lib_dir.expect("no libtorch lib dir found");
        let link_type = match env_var_rerun("LIBTORCH_STATIC").as_deref() {
            Err(_) | Ok("0") | Ok("false") | Ok("FALSE") => LinkType::Dynamic,
            Ok(_) => LinkType::Static,
        };
        Ok(Self {
            os,
            python_interpreter,
            cxx11_abi,
            libtorch_include_dirs,
            libtorch_lib_dir,
            link_type,
        })
    }
}

const KERNEL_FILES: [&str; 24] = [
    "flash_api.cpp",
    "flash_fwd_split_hdim128_bf16_sm80.cu",
    "flash_fwd_split_hdim160_bf16_sm80.cu",
    "flash_fwd_split_hdim192_bf16_sm80.cu",
    "flash_fwd_split_hdim224_bf16_sm80.cu",
    "flash_fwd_split_hdim256_bf16_sm80.cu",
    "flash_fwd_split_hdim32_bf16_sm80.cu",
    "flash_fwd_split_hdim64_bf16_sm80.cu",
    "flash_fwd_split_hdim96_bf16_sm80.cu",
    "flash_fwd_split_hdim128_fp16_sm80.cu",
    "flash_fwd_split_hdim160_fp16_sm80.cu",
    "flash_fwd_split_hdim192_fp16_sm80.cu",
    "flash_fwd_split_hdim224_fp16_sm80.cu",
    "flash_fwd_split_hdim256_fp16_sm80.cu",
    "flash_fwd_split_hdim32_fp16_sm80.cu",
    "flash_fwd_split_hdim64_fp16_sm80.cu",
    "flash_fwd_split_hdim96_fp16_sm80.cu",
    "vllm/activation_kernels.cu",
    "vllm/cache_kernels.cu",
    "vllm/cuda_utils_kernels.cu",
    "vllm/layernorm_kernels.cu",
    "vllm/pos_encoding_kernels.cu",
    "vllm/attention/attention_kernels.cu",
    "vllm/bindings.cpp",
];

fn main() -> Result<()> {
    if std::env::var("VSCODE_CWD").is_ok() {
        // do not compile from rust-analyzer
        return Ok(());
    }

    let num_cpus = std::env::var("RAYON_NUM_THREADS").map_or_else(
        |_| num_cpus::get_physical(),
        |s| usize::from_str(&s).unwrap(),
    );

    let sysinfo = SystemInfo::new()?;
    println!("sysinfo: {:#?}", sysinfo);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus)
        .build_global()
        .unwrap();

    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("CANDLE_FLASH_ATTN_BUILD_DIR") {
        Err(_) =>
        {
            #[allow(clippy::redundant_clone)]
            out_dir.clone()
        }
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            path.canonicalize().expect(&format!(
                "Directory doesn't exists: {} (the current directory is {})",
                &path.display(),
                std::env::current_dir()?.display()
            ))
        }
    };
    set_cuda_include_dir()?;

    {
        let mut inner = out_dir.clone();
        inner.push("vllm/attention");
        std::fs::create_dir_all(inner).unwrap();
    }

    let ccbin_env = std::env::var("CANDLE_NVCC_CCBIN");
    println!("cargo:rerun-if-env-changed=CANDLE_NVCC_CCBIN");

    let compute_cap = compute_cap()?;

    let out_file = build_dir.join("libflashattention.a");

    let kernel_dir = PathBuf::from("kernels");
    let cu_files: Vec<_> = KERNEL_FILES
        .iter()
        .map(|f| {
            let mut obj_file = out_dir.join(f);
            obj_file.set_extension("o");
            (kernel_dir.join(f), obj_file)
        })
        .collect();

    let out_modified: Result<_, _> = out_file.metadata().and_then(|m| m.modified());
    let mut should_compile = false;
    if out_modified.is_err() {
        should_compile = true;
    }

    for entry in glob::glob("kernels/**/*.*").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => {
                let str_path = path.to_str().unwrap();
                if str_path.ends_with(".cu")
                    || str_path.ends_with(".cuh")
                    || str_path.ends_with(".cpp")
                    || str_path.ends_with(".h")
                {
                    if let Ok(out_modified) = &out_modified {
                        let in_modified = path.metadata().unwrap().modified().unwrap();
                        if in_modified.duration_since(*out_modified).is_ok() {
                            should_compile = true;
                        }
                    } else {
                        should_compile = true;
                    }
                    println!("cargo:rerun-if-changed={str_path}");
                }
            }
            Err(e) => println!("cargo:warning={:?}", e),
        }
    }

    if should_compile {
        cu_files
            .par_iter()
            .map(|(cu_file, obj_file)| {
                // $LANG this is set or not depending if running inside terminal or vscode
                // we set it explicitly to help ccache
                let mut command = std::process::Command::new("ccache");
                command
                    .env("LANG", "en_US.UTF-8") 
                    .arg("nvcc")
                    .arg("-std=c++17")
                    .arg("-O3")
                    .arg("-U__CUDA_NO_HALF_OPERATORS__")
                    .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
                    .arg("-U__CUDA_NO_HALF2_OPERATORS__")
                    .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
                    .arg(format!("--gpu-architecture=sm_{compute_cap}"))
                    .arg("-c")
                    .args(["-o", obj_file.to_str().unwrap()])
                    .args(["--default-stream", "per-thread"])
                    .arg("-Icutlass/include")
                    .arg(format!("-D_GLIBCXX_USE_CXX11_ABI={}", sysinfo.cxx11_abi))
                    .args(sysinfo.libtorch_include_dirs.iter().map(|p| {
                        format!(
                            "-I{}",
                            p.to_str().unwrap()
                        )
                    }))
                    .arg("--expt-relaxed-constexpr")
                    .arg("--expt-extended-lambda")
                    .arg("--use_fast_math")
                    .arg("--verbose")
                    ;
                if let Ok(ccbin_path) = &ccbin_env {
                    command
                        .arg("-allow-unsupported-compiler")
                        .args(["-ccbin", ccbin_path]);
                }
                command.arg(cu_file);
                let output = command
                    .spawn()
                    .context("failed spawning nvcc")?
                    .wait_with_output()?;
                if !output.status.success() {
                    anyhow::bail!(
                        "nvcc error while executing compiling: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                        &command,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    )
                }
                Ok(())
            })
            .collect::<Result<()>>()?;
        let obj_files = cu_files.iter().map(|c| c.1.clone()).collect::<Vec<_>>();
        let mut command = std::process::Command::new("nvcc");
        command
            .arg("--lib")
            .args(["-o", out_file.to_str().unwrap()])
            .args(obj_files);
        let output = command
            .spawn()
            .context("failed spawning nvcc")?
            .wait_with_output()?;
        if !output.status.success() {
            anyhow::bail!(
                "nvcc error while linking: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                &command,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        }
    }
    println!("cargo:rustc-link-search={}", build_dir.display());
    let cudart_path = sysinfo
        .libtorch_lib_dir
        .join("../../nvidia/cuda_runtime/lib")
        .display()
        .to_string();
    println!("cargo:rustc-link-search={cudart_path}");
    println!("cargo:rustc-link-lib=flashattention");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=c10_cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    /* laurent: I tried using the cc cuda integration as below but this lead to ptaxs never
       finishing to run for some reason. Calling nvcc manually worked fine.
    cc::Build::new()
        .cuda(true)
        .include("cutlass/include")
        .flag("--expt-relaxed-constexpr")
        .flag("--default-stream")
        .flag("per-thread")
        .flag(&format!("--gpu-architecture=sm_{compute_cap}"))
        .file("kernels/flash_fwd_hdim32_fp16_sm80.cu")
        .compile("flashattn");
    */
    Ok(())
}

fn set_cuda_include_dir() -> Result<()> {
    // NOTE: copied from cudarc build.rs.
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok)
        .map(Into::<PathBuf>::into);

    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];
    let roots = roots.into_iter().map(Into::<PathBuf>::into);
    let root = env_vars
        .chain(roots)
        .find(|path| path.join("include").join("cuda.h").is_file())
        .context("cannot find include/cuda.h")?;
    println!(
        "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
        root.join("include").display()
    );
    Ok(())
}

#[allow(unused)]
fn compute_cap() -> Result<usize> {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    // Try to parse compute caps from env
    let mut compute_cap = if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap_str}");
        compute_cap_str
            .parse::<usize>()
            .context("Could not parse compute cap")?
    } else {
        // Use nvidia-smi to get the current compute cap
        let out = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=compute_cap")
            .arg("--format=csv")
            .output()
            .context("`nvidia-smi` failed. Ensure that you have CUDA installed and that `nvidia-smi` is in your PATH.")?;
        let out = std::str::from_utf8(&out.stdout).context("stdout is not a utf8 string")?;
        let mut lines = out.lines();
        assert_eq!(
            lines.next().context("missing line in stdout")?,
            "compute_cap"
        );
        let cap = lines
            .next()
            .context("missing line in stdout")?
            .replace('.', "");
        let cap = cap
            .parse::<usize>()
            .with_context(|| format!("cannot parse as int {cap}"))?;
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={cap}");
        cap
    };

    // Grab available GPU codes from nvcc and select the highest one
    let (supported_nvcc_codes, max_nvcc_code) = {
        let out = std::process::Command::new("nvcc")
            .arg("--list-gpu-code")
            .output()
            .expect("`nvcc` failed. Ensure that you have CUDA installed and that `nvcc` is in your PATH.");
        let out = std::str::from_utf8(&out.stdout).unwrap();

        let out = out.lines().collect::<Vec<&str>>();
        let mut codes = Vec::with_capacity(out.len());
        for code in out {
            let code = code.split('_').collect::<Vec<&str>>();
            if !code.is_empty() && code.contains(&"sm") {
                if let Ok(num) = code[1].parse::<usize>() {
                    codes.push(num);
                }
            }
        }
        codes.sort();
        let max_nvcc_code = *codes.last().context("no gpu codes parsed from nvcc")?;
        (codes, max_nvcc_code)
    };

    // Check that nvcc supports the asked compute caps
    if !supported_nvcc_codes.contains(&compute_cap) {
        anyhow::bail!(
            "nvcc cannot target gpu arch {compute_cap}. Available nvcc targets are {supported_nvcc_codes:?}."
        );
    }
    if compute_cap > max_nvcc_code {
        anyhow::bail!(
            "CUDA compute cap {compute_cap} is higher than the highest gpu code from nvcc {max_nvcc_code}"
        );
    }

    Ok(compute_cap)
}
