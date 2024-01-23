use crate::HashMap;
use aicirt::bintokens::list_tokenizers;
use anyhow::{bail, Result};
use clap::{Args, Command, Parser};
use std::time::Instant;

const SETTINGS: [(&'static str, &'static str, f64); 4] = [
    ("attn_rtol", "relative tolerance for flash attn check", 0.1),
    ("attn_atol", "absolute tolerance for flash attn check", 0.1),
    ("test_maxtol", "max allowed error", 0.5),
    ("test_avgtol", "avg allowed error", 0.2),
];

lazy_static::lazy_static! {
    static ref CHECK_SETTINGS: std::sync::Mutex<HashMap<String, f64>> = std::sync::Mutex::new(
        SETTINGS.iter().map(|(k, _, v)| (k.to_string(), *v)).collect::<HashMap<_,_>>()
    );
}

pub fn all_settings() -> String {
    format!(
        "Settings available via -s or --setting (with their default values):\n{all}\n",
        all = SETTINGS
            .map(|(k, d, v)| format!("  -s {:20} {}", format!("{}={}", k, v), d))
            .join("\n")
    )
}

pub fn set_setting(name: &str, val: f64) -> Result<()> {
    let mut settings = CHECK_SETTINGS.lock().unwrap();
    let name = name.to_string();
    if settings.contains_key(&name) {
        settings.insert(name, val);
        Ok(())
    } else {
        bail!("unknown setting: {name}")
    }
}

pub fn get_setting(name: &str) -> f64 {
    let settings = CHECK_SETTINGS.lock().unwrap();
    if let Some(val) = settings.get(name) {
        *val
    } else {
        panic!("unknown setting: {}", name)
    }
}

fn apply_setting(s: &str) -> Result<()> {
    let parts: Vec<&str> = s.split('=').collect();
    if parts.len() != 2 {
        bail!("expecting name=value");
    }
    let v = parts[1].parse::<f64>()?;
    set_setting(parts[0], v)
}

pub fn apply_settings(settings: &Vec<String>) -> Result<()> {
    for s in settings {
        match apply_setting(s) {
            Ok(_) => {}
            Err(e) => {
                bail!(
                    "{all}\nfailed to set setting {s}: {e}",
                    all = all_settings()
                );
            }
        }
    }
    Ok(())
}

fn logging_info() -> &'static str {
    r#"
There are 5 logging levels: error, warn, info, debug, and trace.
You can assign different log levels to crates:

   info,rllm=trace,aicirt=debug - rllm at trace, aicirt at debug, otherwise info

You can set logging levels with --log or with RUST_LOG environment variable.
"#
}

pub fn parse_with_settings<T>() -> T
where
    T: Parser + Args,
{
    let cli = Command::new("CLI").after_help(format!(
        "\n{}\n{}\n{}",
        all_settings(),
        list_tokenizers(),
        logging_info()
    ));
    let cli = T::augment_args(cli);
    let matches = cli.get_matches();
    T::from_arg_matches(&matches)
        .map_err(|err| err.exit())
        .unwrap()
}

pub fn limit_str(s: &str, max_len: usize) -> String {
    limit_bytes(s.as_bytes(), max_len)
}

pub fn limit_bytes(s: &[u8], max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", String::from_utf8_lossy(&s[0..max_len]))
    } else {
        String::from_utf8_lossy(s).to_string()
    }
}

pub fn pad_to_multiple<T>(v: &mut Vec<T>, multiple: usize)
where
    T: Default + Clone,
{
    let len = v.len();
    let rem = len % multiple;
    if rem > 0 {
        let pad_len = multiple - rem;
        v.extend(std::iter::repeat(T::default()).take(pad_len));
    }
}

pub struct TimerGuard {
    name: &'static str,
    start: Instant,
}

impl TimerGuard {
    fn new(name: &'static str) -> TimerGuard {
        TimerGuard {
            name,
            start: Instant::now(),
        }
    }
}

impl std::ops::Drop for TimerGuard {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        log::info!("TIMER {}: {:?}", self.name, duration);
    }
}

pub fn timer(name: &'static str) -> TimerGuard {
    TimerGuard::new(name)
}

pub fn log_time(name: &'static str, start: Instant) {
    let duration = start.elapsed();
    log::info!("{}: {:?}", name, duration);
}
