use anyhow::Result;
use candle::{DType, Tensor};

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

pub fn max_diff(t1: &Tensor, t2: &Tensor) -> Result<f64> {
    let mut diff = t1.sub(t2)?.abs()?;
    while diff.dims().len() > 0 {
        diff = diff.max(0)?;
    }
    let max: f64 = diff.to_dtype(DType::F64)?.to_vec0()?;
    Ok(max)
}

pub fn check_all_close(t1: &Tensor, t2: &Tensor) {
    let df = max_diff(t1, t2).unwrap();
    if df > 1e-2 {
        print!("A: {t1:?}\n{t1}\n");
        print!("B: {t2:?}\n{t2}\n");
        let d = t1.sub(t2).unwrap().abs().unwrap();
        print!("D: {d:?}\n{d}\n");

        panic!("not close {df:.5}");
    }
}
