use crate::Tensor;
use anyhow::Result;
use tch::kind::Element;

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
    let diff = (t1 - t2).abs();
    let max = diff.max().double_value(&[]);
    Ok(max)
}

pub fn check_all_close(t1: &Tensor, t2: &Tensor, max_diff_: f64) {
    assert!(t1.size() == t2.size());
    let df = max_diff(t1, t2).unwrap();
    if df > max_diff_ {
        print!("A: {t1:?}\n{t1}\n");
        print!("B: {t2:?}\n{t2}\n");
        let d = (t1 - t2).abs();
        print!("D: {d:?}\n{d}\n");

        panic!("not close {df:.5}");
    }
}

pub fn to_vec1<T: Element>(t: &Tensor) -> Vec<T> {
    let sz = t.size1().unwrap();
    let mut dst = vec![T::ZERO; sz as usize];
    t.to_kind(T::KIND)
        .copy_data::<T>(&mut dst, sz as usize);
    dst
}
