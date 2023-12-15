use crate::Tensor;
use tch::{kind::Element, IndexOp as _};

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

pub fn check_all_close_rel(t1: &Tensor, t2: &Tensor, _max_diff: f64) {
    assert!(t1.size() == t2.size());

    let atol: f64 = 0.50;
    let rtol: f64 = 0.20;

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
