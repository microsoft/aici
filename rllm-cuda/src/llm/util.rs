use super::DType;
use rllm::util::get_setting;
use tch::{kind::Element, Device, IndexOp as _, Tensor};

#[cfg(feature = "cuda")]
use tch_cuda::{
    cuda_empty_cache, cuda_get_device_properties, cuda_get_stats_allocated_bytes,
    cuda_reset_peak_memory_stats,
};

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
        let avg = diff.mean(Some(DType::Float)).double_value(&[]);
        panic!("not close {df:.5} (mean={avg:.5})");
    }
}

pub fn to_vec1<T: Element>(t: &Tensor) -> Vec<T> {
    let sz = t.size1().unwrap();
    let mut dst = vec![T::ZERO; sz as usize];
    t.to_kind(T::KIND).copy_data::<T>(&mut dst, sz as usize);
    dst
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(n) => {
            cuda_empty_cache();
            cuda_reset_peak_memory_stats(n);
        }
        _ => {}
    }
}

pub fn log_mem_stats(lbl: &str, device: Device) {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(n) => {
            let stats = cuda_get_stats_allocated_bytes(n);
            log::info!("cuda mem: {lbl} {stats}");
        }
        _ => {
            let _ = lbl;
        }
    }
}

pub fn gpu_peak_allocated_bytes(device: Device) -> usize {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(n) => {
            let stats = cuda_get_stats_allocated_bytes(n);
            stats.peak as usize
        }
        _ => 0,
    }
}

pub fn gpu_memory_size(device: Device) -> usize {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(n) => {
            let stats = cuda_get_device_properties(n);
            stats.total_memory as usize
        }
        _ => 0,
    }
}

pub fn synchronize(device: Device) {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(n) => tch::Cuda::synchronize(n as i64),
        _ => {}
    }
}

#[allow(dead_code)]
pub fn scalar_tensor<T>(v: T, d: Device) -> Tensor
where
    T: Element,
{
    Tensor::from_slice(&[v]).to(d).reshape(&[])
}
