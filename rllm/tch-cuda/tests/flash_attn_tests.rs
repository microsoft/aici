use anyhow::Result;
use tch::{kind::Element, Device, IndexOp, Kind, Tensor};

fn to_vec3<T: Element>(t: &Tensor) -> Vec<Vec<Vec<T>>> {
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

fn to_vec3_round(t: &Tensor, digits: i32) -> Vec<Vec<Vec<f32>>> {
    let b = 10f32.powi(digits);
    let t = to_vec3::<f32>(t);
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    t
}

#[test]
fn flash_attn_varlen() -> Result<()> {
    let device = Device::Cuda(0);
    let q = Tensor::arange(48, (Kind::BFloat16, device)).reshape(&[3, 2, 8]);
    let k = &q / 40.;
    let v = &q / 50.;
    let q = &q / 30.;

    let seqlens_q = Tensor::from_slice(&[0i32, 2i32]).to_device(device);
    let seqlens_k = Tensor::from_slice(&[0i32, 2i32]).to_device(device);

    let ys = {
        let q = q.transpose(0, 1);
        let k = k.transpose(0, 1);
        let v = v.transpose(0, 1);
        tch_cuda::flash_attn_varlen(&q, &k, &v, &seqlens_q, &seqlens_k, 32, 32, 0.5, false)
            .transpose(0, 1)
    };
    let ys = ys.to_kind(Kind::Float);

    assert_eq!(ys.size(), &[3, 2, 8]);
    assert_eq!(
        to_vec3_round(&ys, 4),
        &[
            [
                [0.084, 0.1035, 0.124, 0.1436, 0.1641, 0.1836, 0.2031, 0.2236],
                [0.0923, 0.1118, 0.1318, 0.1523, 0.1719, 0.1924, 0.2119, 0.2324]
            ],
            [
                [0.4199, 0.4395, 0.459, 0.4805, 0.5, 0.5195, 0.543, 0.5625],
                [0.4277, 0.4473, 0.4668, 0.4883, 0.5078, 0.5273, 0.5508, 0.5703]
            ],
            [
                [0.7539, 0.7734, 0.793, 0.8125, 0.832, 0.8516, 0.875, 0.8945],
                [0.7617, 0.7813, 0.8008, 0.8203, 0.8398, 0.8594, 0.8828, 0.9023]
            ]
        ]
    );
    Ok(())
}
