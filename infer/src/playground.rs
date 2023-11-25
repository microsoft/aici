use anyhow::Result;
use candle::{DType, Device, IndexOp, Shape, Tensor};

use crate::to_offsets;

struct XorShiftRng {
    state: u32,
}

impl XorShiftRng {
    fn new() -> Self {
        Self { state: 12345 }
    }

    fn next(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        (x & 0xffffffff) as u32
    }

    fn urandom(&mut self) -> f32 {
        self.next() as f32 / 0xffffffffu32 as f32
    }

    fn srandom(&mut self) -> f32 {
        self.urandom() * 2.0 - 1.0
    }

    fn rand_tensor<S: Into<Shape>>(&mut self, shape: S, device: &Device) -> Tensor {
        let shape: Shape = shape.into();
        let n = shape.elem_count();
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(self.srandom());
        }
        Tensor::new(data.as_slice(), device)
            .unwrap()
            .reshape(shape)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }
}

fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_seqlen: &[usize],
    kv_seqlen: &[usize],
    _p: f32,
    softmax_scale: f32,
) -> Result<Tensor> {
    // flash-attn expects (seq_len, nheads, head_dim)
    let q = q.transpose(0, 1)?.contiguous()?;
    let k = k.transpose(0, 1)?.contiguous()?;
    let v = v.transpose(0, 1)?.contiguous()?;
    let cuda = Device::new_cuda(0)?;
    let device = &cuda;
    let causal = true;
    let r = candle_flash_attn::flash_attn_varlen(
        &q,
        &k,
        &v,
        &to_offsets(q_seqlen, device),
        &to_offsets(kv_seqlen, device),
        *q_seqlen.iter().max().unwrap(),
        *kv_seqlen.iter().max().unwrap(),
        softmax_scale,
        causal,
    )?
    .transpose(0, 1)?;

    Ok(r)
}

pub fn all_close(t1: &Tensor, t2: &Tensor) -> Result<bool> {
    let mut diff = t1.sub(t2)?.abs()?;
    while diff.dims().len() > 0 {
        diff = diff.max(0)?;
    }
    let max: f64 = diff.to_dtype(DType::F64)?.to_vec0()?;
    Ok(max < 1e-3)
}

pub fn check_all_close(t1: &Tensor, t2: &Tensor) {
    let cl = all_close(t1, t2).unwrap();
    if !cl {
        print!("A: {t1:?}\n{t1}\n");
        print!("B: {t2:?}\n{t2}\n");
        let d = t1.sub(t2).unwrap().abs().unwrap();
        print!("D: {d:?}\n{d}\n");
        panic!("not close");
    }
}

#[allow(dead_code)]
pub fn playground_1() {
    let mut xor = XorShiftRng::new();
    let device = Device::new_cuda(0).unwrap();

    let slen = 5;
    let pref = 2;
    let head_dim = 8;
    let n_heads = 1;
    let query = xor.rand_tensor(&[n_heads, slen, head_dim], &device);
    let key = xor.rand_tensor(&[n_heads, slen, head_dim], &device);
    let value = xor.rand_tensor(&[n_heads, slen, head_dim], &device);

    let q2 = query.i((.., pref.., ..)).unwrap();

    let q1 = query.i((.., 0..pref, ..)).unwrap();
    let k1 = key.i((.., 0..pref, ..)).unwrap();
    let v1 = value.i((.., 0..pref, ..)).unwrap();

    let out = flash_attn(&query, &key, &value, &[slen], &[slen], 0.0, 1.0).unwrap();
    println!("out\n{out}");
    let o1 = flash_attn(&q1, &k1, &v1, &[pref], &[pref], 0.0, 1.0).unwrap();
    println!("o1\n{o1}");
    let o2 = flash_attn(&q2, &key, &value, &[slen - pref], &[slen], 0.0, 1.0).unwrap();
    println!("o2\n{o2}");
    println!("{:?}", out.dims());
    println!("{:?}", o1.dims());
    println!("{:?}", o2.dims());
    println!("pref");
    check_all_close(&out.i((.., 0..pref, ..)).unwrap(), &o1);
    println!("suff");
    check_all_close(&out.i((.., pref.., ..)).unwrap(), &o2);
}

