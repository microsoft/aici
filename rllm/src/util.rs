use std::collections::HashMap;

use crate::Tensor;
use anyhow::{bail, Result};
use tch::{kind::Element, Device, IndexOp as _};
#[cfg(feature = "cuda")]
use tch_flash_attn::{
    cuda_empty_cache, cuda_get_stats_allocated_bytes, cuda_reset_peak_memory_stats,
};

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
    SETTINGS
        .map(|(k, d, v)| format!("{}: {} (default={})", k, d, v))
        .join("\n")
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
                    "all settings:\n{all}\nfailed to set setting {s}: {e}",
                    all = all_settings()
                );
            }
        }
    }
    Ok(())
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

pub fn check_all_close_attn(t1: &Tensor, t2: &Tensor) {
    assert!(t1.size() == t2.size());

    let rtol = get_setting("attn_rtol");
    let atol = get_setting("attn_atol");

    let diff = (t1 - t2).abs() - (rtol * t2.abs() + atol);

    let max_over = diff.max().double_value(&[]);
    if max_over > 0.0 {
        panic!("not close (rel) {max_over:.5}");
    }
}

pub fn check_all_close(t1: &Tensor, t2: &Tensor, max_diff: f64) {
    assert!(t1.size() == t2.size());

    let diff = (t1 - t2).abs();

    let df = diff.max().double_value(&[]);
    if df > max_diff {
        print!("A: {t1:?}\n{t1}\n");
        print!("B: {t2:?}\n{t2}\n");
        print!("D: {diff:?}\n{diff}\n");
        let avg = diff.mean(Some(crate::DType::Float)).double_value(&[]);
        panic!("not close {df:.5} (mean={avg:.5})");
    }
}

pub fn to_vec1<T: Element>(t: &Tensor) -> Vec<T> {
    let sz = t.size1().unwrap();
    let mut dst = vec![T::ZERO; sz as usize];
    t.to_kind(T::KIND).copy_data::<T>(&mut dst, sz as usize);
    dst
}

pub fn to_vec2<T: Element>(t: &Tensor) -> Vec<Vec<T>> {
    let (d0, d2) = t.size2().unwrap();
    (0..d0)
        .map(|i| {
            let mut dst = vec![T::ZERO; d2 as usize];
            t.i((i, ..))
                .to_kind(T::KIND)
                .copy_data::<T>(&mut dst, d2 as usize);
            dst
        })
        .collect::<Vec<_>>()
}

pub fn to_vec3<T: Element>(t: &Tensor) -> Vec<Vec<Vec<T>>> {
    let (d0, d1, d2) = t.size3().unwrap();
    (0..d0)
        .map(|i| {
            (0..d1)
                .map(|j| {
                    let mut dst = vec![T::ZERO; d2 as usize];
                    t.i((i, j, ..))
                        .to_kind(T::KIND)
                        .copy_data::<T>(&mut dst, d2 as usize);
                    dst
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

pub fn reset_mem_stats(device: Device) {
    #[cfg(feature = "cuda")]
    match device {
        Device::Cuda(n) => {
            cuda_empty_cache();
            cuda_reset_peak_memory_stats(n);
        }
        _ => {}
    }
}

pub fn log_mem_stats(lbl: &str, device: Device) {
    #[cfg(feature = "cuda")]
    match device {
        Device::Cuda(n) => {
            let stats = cuda_get_stats_allocated_bytes(n);
            log::info!("cuda mem: {lbl} {stats}");
        }
        _ => {}
    }
}
