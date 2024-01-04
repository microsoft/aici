pub mod api;
pub mod msgchannel;
pub mod semaphore;
pub mod shm;
pub mod bintokens;
mod bench;

use anyhow::Result;
use flexi_logger::{DeferredNow, Logger, WriteMode};
use log::Record;
pub use bench::*;

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
