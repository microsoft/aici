use tch::Tensor;
use torch_sys::C_tensor;

unsafe fn ptr_to_string(ptr: *mut libc::c_char) -> Option<String> {
    if !ptr.is_null() {
        let str = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
        libc::free(ptr as *mut libc::c_void);
        Some(str)
    } else {
        None
    }
}

extern "C" {
    fn mha_varlen_fwd_C(
        q: *const C_tensor, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        k: *const C_tensor, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
        v: *const C_tensor, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
        cu_seqlens_q: *const C_tensor, // b+1
        cu_seqlens_k: *const C_tensor, // b+1
        seqused_k: *const C_tensor, // b. If given, only this many elements of each batch element's keys are used. (opt)
        max_seqlen_q: i32,
        max_seqlen_k: i32,
        p_dropout: f32,
        softmax_scale: f32,
        zero_tensors: bool,
        is_causal: bool,
        window_size_left: i32,
        window_size_right: i32,
        outp: *mut *mut C_tensor, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    ) -> *mut libc::c_char;
}

/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Tensor {
    let mut outputs = vec![std::ptr::null_mut(); 1];
    let err = unsafe {
        ptr_to_string(mha_varlen_fwd_C(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            seqlens_q.as_ptr(),
            seqlens_k.as_ptr(),
            std::ptr::null(),
            max_seqlen_q as i32,
            max_seqlen_k as i32,
            0.0,
            softmax_scale,
            false,
            causal,
            -1,
            -1,
            outputs.as_mut_ptr(),
        ))
    };
    match err {
        None => unsafe { Tensor::from_ptr(outputs[0]) },
        Some(err) => panic!("flash_attn_varlen: {}", err),
    }
}
