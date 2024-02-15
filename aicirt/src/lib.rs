pub mod api;
mod bench;
pub mod bintokens;
pub mod futexshm;
pub mod msgchannel;
pub mod semaphore;
pub mod shm;

#[cfg(target_os = "macos")]
mod macos;

use std::fmt::Write;

use anyhow::Result;
pub use bench::*;
use flexi_logger::style;
use flexi_logger::{DeferredNow, Logger, WriteMode};
use log::Record;
use thread_priority::{
    set_thread_priority_and_policy, thread_native_id, RealtimeThreadSchedulePolicy, ThreadPriority,
    ThreadSchedulePolicy,
};

pub use fxhash::FxHashMap as HashMap;
pub use fxhash::FxHashSet as HashSet;

pub enum LogMode {
    Normal,
    Test,
    Daemon,
}

struct LimitedWrite {
    limit: usize,
    dst: String,
}

impl Write for LimitedWrite {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        if self.dst.len() > self.limit {
            return Err(std::fmt::Error);
        }
        if self.dst.len() + s.len() < self.limit {
            self.dst.push_str(s);
            Ok(())
        } else {
            let remaining = self.limit - self.dst.len();
            self.dst.push_str(&s[..remaining]);
            self.dst.push_str(" (...)");
            Err(std::fmt::Error)
        }
    }
}

fn args_to_str(limit: usize, args: &std::fmt::Arguments) -> String {
    // let capacity = args.estimated_capacity();
    let mut output = LimitedWrite {
        limit,
        dst: String::with_capacity(128),
    };
    if output.write_fmt(*args).is_err() {
        assert!(output.dst.len() > limit);
    }
    output.dst
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

pub fn get_unix_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// An error thrown from the WASM runtime or otherwise originating from user error
/// - should not generate additional stacktraces from where it's caught.
#[derive(Debug)]
pub struct UserError {
    pub msg: String,
}

impl UserError {
    pub fn new(msg: String) -> Self {
        Self { msg }
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
}

impl std::fmt::Display for UserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl std::error::Error for UserError {}

#[macro_export]
macro_rules! user_error {
    ($($tt:tt)*) => {
        $crate::UserError::anyhow(format!($($tt)*))
    };
}

#[macro_export]
macro_rules! bail_user {
    ($($tt:tt)*) => {
        return Err($crate::UserError::anyhow(format!($($tt)*)))
    };
}

#[macro_export]
macro_rules! ensure_user {
    ($cond:expr, $($tt:tt)*) => {
        if !$cond {
            return Err($crate::UserError::anyhow(format!($($tt)*)))
        }
    };
}

pub fn is_hex_string(s: &str) -> bool {
    s.chars().all(|c| c.is_digit(16))
}

pub fn valid_module_or_tag(s: &str) -> bool {
    valid_module_id(s) || valid_tagname(s)
}

pub fn valid_module_id(s: &str) -> bool {
    s.len() == 64 && is_hex_string(s)
}

pub fn valid_tagname(s: &str) -> bool {
    match s.chars().next() {
        Some(c) if c.is_alphabetic() => {
            !valid_module_id(s)
                && s.chars().all(|c| {
                    c == '_' || c == '-' || c == '.' || c.is_digit(10) || c.is_alphabetic()
                })
        }
        _ => false,
    }
}

fn set_priority(pri: ThreadPriority) {
    // this fails on WSL
    let _ = set_thread_priority_and_policy(
        thread_native_id(),
        pri,
        ThreadSchedulePolicy::Realtime(RealtimeThreadSchedulePolicy::Fifo),
    );
}

pub fn set_max_priority() {
    set_priority(ThreadPriority::Max);
}

pub fn set_min_priority() {
    set_priority(ThreadPriority::Min);
}
