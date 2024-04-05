use std::fmt::Write;

use anyhow::Result;
use flexi_logger::style;
use flexi_logger::{DeferredNow, Logger, WriteMode};
use log::Record;

pub enum LogMode {
    Normal,
    Test,
    Daemon,
}

struct LimitedWrite {
    limit: usize,
    dst: Vec<u8>,
}

impl Write for LimitedWrite {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        if self.dst.len() > self.limit {
            return Err(std::fmt::Error);
        }
        if self.dst.len() + s.len() < self.limit {
            self.dst.extend_from_slice(s.as_bytes());
            Ok(())
        } else {
            let remaining = self.limit - self.dst.len();
            self.dst.extend_from_slice(&s.as_bytes()[..remaining]);
            self.dst.extend_from_slice(b" (...)");
            Err(std::fmt::Error)
        }
    }
}

fn args_to_str(limit: usize, args: &std::fmt::Arguments) -> String {
    // let capacity = args.estimated_capacity();
    let mut output = LimitedWrite {
        limit,
        dst: Vec::with_capacity(128),
    };
    if output.write_fmt(*args).is_err() {
        assert!(output.dst.len() > limit);
    }
    match String::from_utf8(output.dst) {
        Ok(s) => s,
        Err(err) => String::from_utf8_lossy(err.as_bytes()).to_string(),
    }
}

fn truncated_format(
    w: &mut dyn std::io::Write,
    _now: &mut DeferredNow,
    record: &Record,
) -> Result<(), std::io::Error> {
    let level = record.level();
    write!(
        w,
        "{} [{}] {}",
        style(level).paint(level.to_string()),
        record.module_path().unwrap_or("<unnamed>"),
        style(level).paint(args_to_str(1000, record.args()))
    )
}

fn daemon_format(
    w: &mut dyn std::io::Write,
    now: &mut DeferredNow,
    record: &Record,
) -> Result<(), std::io::Error> {
    write!(
        w,
        "{} {} [{}] {}",
        now.format("%Y-%m-%d %H:%M:%S%.3f"),
        record.level(),
        record.module_path().unwrap_or("<unnamed>"),
        args_to_str(5000, record.args())
    )
}

pub fn init_log(mode: LogMode) -> Result<()> {
    let logger = match mode {
        LogMode::Normal => Logger::try_with_env_or_str("info")?
            .format(truncated_format)
            .log_to_stdout(),
        LogMode::Test => {
            Logger::try_with_env_or_str("debug")?.write_mode(WriteMode::SupportCapture)
        }
        LogMode::Daemon => Logger::try_with_env_or_str("info")?
            .format(daemon_format)
            .log_to_stdout(),
    };

    logger.start()?;
    Ok(())
}

pub fn setup_log() {
    init_log(LogMode::Normal).expect("Failed to initialize log")
}
