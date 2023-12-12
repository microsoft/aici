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
                        .to_dtype(T::KIND, false, false)
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
    let q = Tensor::arange(48, (Kind::Half, device)).reshape(&[3, 2, 8]);
    let k = &q / 40.;
    let v = &q / 50.;
    let q = &q / 30.;

    let seqlens_q = Tensor::from_slice(&[0i32, 2i32]).to_device(device);
    let seqlens_k = Tensor::from_slice(&[0i32, 2i32]).to_device(device);

    let ys = {
        let q = q.transpose(0, 1);
        let k = k.transpose(0, 1);
        let v = v.transpose(0, 1);
        tch_flash_attn::flash_attn_varlen(&q, &k, &v, &seqlens_q, &seqlens_k, 32, 32, 0.5, false)
            .transpose(0, 1)
    };
    let ys = ys.to_dtype(Kind::Float, false, true);

    assert_eq!(ys.size(), &[3, 2, 8]);
    assert_eq!(
        to_vec3_round(&ys, 4),
        &[
            [
                [0.0837, 0.1038, 0.1237, 0.1437, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1521, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.875, 0.895],
                [0.7617, 0.7817, 0.8018, 0.8218, 0.8418, 0.8618, 0.8818, 0.9019]
            ]
        ]
    );
    Ok(())
}
