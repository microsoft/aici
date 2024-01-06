pub mod api;
mod bench;
pub mod bintokens;
pub mod msgchannel;
pub mod semaphore;
pub mod shm;

use anyhow::Result;
pub use bench::*;
use flexi_logger::{DeferredNow, Logger, WriteMode};
use log::Record;

pub enum LogMode {
    Normal,
    Test,
    Deamon,
}

fn daemon_format(
    w: &mut dyn std::io::Write,
    now: &mut DeferredNow,
    record: &Record,
) -> Result<(), std::io::Error> {
    write!(
        w,
        "[{}] {} {}",
        now.format("%Y-%m-%d %H:%M:%S%.3f"),
        record.level(),
        &record.args()
    )
}

pub fn init_log(mode: LogMode) -> Result<()> {
    let logger = match mode {
        LogMode::Normal => Logger::try_with_env_or_str("warn")?.log_to_stdout(),
        LogMode::Test => {
            Logger::try_with_env_or_str("debug")?.write_mode(WriteMode::SupportCapture)
        }
        LogMode::Deamon => Logger::try_with_env_or_str("info")?
            .format(daemon_format)
            .log_to_stdout(),
    };

    logger.start()?;
    Ok(())
}

pub fn setup_log() {
    init_log(LogMode::Normal).expect("Failed to initialize log")
}

/// An error thrown from the WASM runtime - should not generate additional stacktraces
/// from where it's caught.
#[derive(Debug)]
pub struct WasmError {
    pub msg: String,
}

impl WasmError {
    pub fn new(msg: String) -> Self {
        Self {
            msg
        }
    }

    pub fn anyhow(msg: String) -> anyhow::Error {
        anyhow::anyhow!(Self::new(msg))
    }

    pub fn is_self(e: &anyhow::Error) -> bool {
        e.downcast_ref::<Self>().is_some()
    }

    pub fn maybe_stacktrace(e: &anyhow::Error) -> String {
        if let Some(e) = e.downcast_ref::<Self>() {
            format!("{}", e)
        } else {
            format!("{:?}", e)
        }
    }

    pub fn prefix(&self, prefix: &str) -> Self {
        Self {
            msg: format!("{}\n{}", prefix, self.msg),
        }
    }
}

impl std::fmt::Display for WasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl std::error::Error for WasmError {}

pub fn get_unix_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
