use clap::Parser;
use rllm::{
    llm::{
        tmodel::{TModel, TchLoaderArgs},
        DType,
    },
    util::parse_with_settings,
};
use tch::Device;

/// Serve LLMs with AICI over HTTP with tch (torch) backend.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct DriverArgs {
    #[clap(flatten)]
    pub args: rllm::server::RllmCliArgs,

    /// Specify which type to use in the model (bf16, f16, f32)
    #[arg(long, default_value = "", help_heading = "Model")]
    pub dtype: String,

    /// Enable nvprof profiling for given engine step (if available)
    #[arg(long, default_value_t = 0, help_heading = "Development")]
    pub profile_step: usize,
}

#[actix_web::main]
async fn main() -> () {
    let args = parse_with_settings::<DriverArgs>();
    let _ = args;

    let (device, dtype) = if tch::Cuda::is_available() {
        (Device::Cuda(0), None)
    } else {
        // At least on AMD 5500m MPS is 3x slower than CPU
        // #[cfg(target_os = "macos")]
        // let r = (Device::Mps, DType::Half);
        // #[cfg(not(target_os = "macos"))]
        let r = (Device::Cpu, Some(DType::Float));
        r
    };

    let dtype = match args.dtype.as_str() {
        "bf16" => Some(DType::BFloat16),
        "f16" => Some(DType::Half),
        "f32" => Some(DType::Float),
        "" => dtype,
        _ => panic!("invalid dtype; try one of bf16, f16, f32"),
    };

    let model_args = TchLoaderArgs { device, dtype, profile_step_no: args.profile_step };
    rllm::server::server_main::<TModel>(args.args, model_args).await;
}
