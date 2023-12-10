pub mod api;
pub mod msgchannel;
pub mod semaphore;
pub mod shm;

use anyhow::Result;
use flexi_logger::{Logger, WriteMode};

fn init_log(is_test: bool) -> Result<()> {
    let logger = if is_test {
        Logger::try_with_env_or_str("debug")?.write_mode(WriteMode::SupportCapture)
    } else {
        Logger::try_with_env_or_str("warn")?.log_to_stdout()
    };
    logger.start()?;
    Ok(())
}

pub fn setup_log() {
    init_log(false).expect("Failed to initialize log")
}

pub fn setup_log_for_test() {
    init_log(true).expect("Failed to initialize log")
}
