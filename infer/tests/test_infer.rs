use rllm::{config::SamplingParams, LoaderArgs, RllmEngine};

struct Ctx {
    infer: RllmEngine,
    sampling_params: SamplingParams,
}

impl Ctx {
    fn new() -> Self {
        let mut sampling_params = SamplingParams::default();
        sampling_params.max_tokens = 10;
        Self {
            infer: RllmEngine::load(LoaderArgs::default()).unwrap(),
            sampling_params,
        }
    }
}

fn expect(ctx: &mut Ctx, prompt: &str, expected: &str) {
    let gen = ctx
        .infer
        .generate(prompt, ctx.sampling_params.clone())
        .unwrap();
    if gen != expected {
        panic!("expected: {:?}, got: {:?}", expected, gen);
    }
}

// this is for Llama-2-7b-hf
const QUERIES: &'static [(&'static str, &'static str)] = &[
    (
        "Color green is",
        "the color of nature, the color of the earth",
    ),
    (
        "Tarski's fixed-point theorem was proven by",
        "Alfred Tarski in 1936.",
    ),
    (
        "Tarski's fixed-point theorem is about",
        "the existence of a fixed point in a certain relation",
    ),
];

#[test]
fn test_infer1() {
    let mut ctx = Ctx::new();

    let stats0 = ctx.infer.get_stats();

    // make sure we get deterministic results
    expect(&mut ctx, QUERIES[0].0, QUERIES[0].1);
    expect(&mut ctx, QUERIES[0].0, QUERIES[0].1);
    expect(&mut ctx, QUERIES[0].0, QUERIES[0].1);

    for idx in 0..QUERIES.len() {
        expect(&mut ctx, QUERIES[idx].0, QUERIES[idx].1);
    }

    ctx.infer.alt = 1;
    for idx in 0..QUERIES.len() {
        expect(&mut ctx, QUERIES[idx].0, QUERIES[idx].1);
    }

    let stats = ctx.infer.get_stats();
    assert!(stats0.free_gpu_blocks == stats.free_gpu_blocks);
    assert!(stats0.free_cpu_blocks == stats.free_cpu_blocks);
}
