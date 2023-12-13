use crate::{to_offsets, util::check_all_close};
use crate::{DType, Device, IndexOp, Tensor};
use anyhow::Result;

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

    fn rand_tensor(&mut self, shape: &[i64], device: Device) -> Tensor {
        let mut n = 1;
        for k in shape {
            n *= *k as usize;
        }
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(self.srandom());
        }
        Tensor::from_slice(data.as_slice())
            .to(device)
            .reshape(shape)
            .to_kind(DType::BFloat16)
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
    println!("Q: {q}");
    println!("K: {k}");
    println!("V: {v}");
    println!("q_seqlen: {q_seqlen:?}");
    println!("kv_seqlen: {kv_seqlen:?}");

    // flash-attn expects (seq_len, nheads, head_dim)
    let q = q.contiguous();
    let k = k.contiguous();
    let v = v.contiguous();
    let cuda = Device::Cuda(0);
    let device = cuda;
    let causal = true;
    let r = tch_flash_attn::flash_attn_varlen(
        &q,
        &k,
        &v,
        &to_offsets(q_seqlen, device).1,
        &to_offsets(kv_seqlen, device).1,
        *q_seqlen.iter().max().unwrap(),
        *kv_seqlen.iter().max().unwrap(),
        softmax_scale,
        causal,
    );

    let r2 = crate::attn::naive_attn(
        &q,
        &k,
        &v,
        &to_offsets(q_seqlen, device).1,
        &to_offsets(kv_seqlen, device).1,
        *q_seqlen.iter().max().unwrap(),
        *kv_seqlen.iter().max().unwrap(),
        softmax_scale,
        causal,
    )?;

    println!("R: {r}");
    println!("R2: {r2}");

    check_all_close(&r, &r2, 0.05);

    Ok(r)
}

#[allow(dead_code)]
pub fn playground_1() {
    let mut xor = XorShiftRng::new();
    let device = Device::Cuda(0);

    let slen = 5;
    let pref = 2;
    let head_dim = 16;
    let n_heads = 8;
    let query = xor.rand_tensor(&[slen, n_heads, head_dim], device);
    let key = xor.rand_tensor(&[slen, n_heads, head_dim], device);
    let value = xor.rand_tensor(&[slen, n_heads, head_dim], device);

    let q2 = query.i((pref.., .., ..));

    // let q1 = query.i((0..pref, .., ..)).unwrap();
    // let k1 = key.i((0..pref, .., ..)).unwrap();
    // let v1 = value.i((0..pref, .., ..)).unwrap();

    // let out = flash_attn(&query, &key, &value, &[slen], &[slen], 0.0, 1.0).unwrap();
    // println!("out\n{out}");
    // let o1 = flash_attn(&q1, &k1, &v1, &[pref], &[pref], 0.0, 1.0).unwrap();
    // println!("o1\n{o1}");
    let _o2 = flash_attn(
        &q2,
        &key,
        &value,
        &[(slen - pref) as usize],
        &[slen as usize],
        0.0,
        1.0,
    );
    // println!("o2\n{o2}");
    // println!("{:?}", out.dims());
    // println!("{:?}", o1.dims());
    // println!("{:?}", o2.dims());
    // println!("pref");
    // check_all_close(&out.i((.., 0..pref, ..)).unwrap(), &o1);
    // println!("suff");
    // check_all_close(&out.i((.., pref.., ..)).unwrap(), &o2);
}
