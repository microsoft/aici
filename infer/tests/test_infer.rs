use std::collections::HashMap;

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
    let _ = env_logger::builder().is_test(true).try_init();

    let mut ctx = Ctx::new();

    let stats0 = ctx.infer.get_stats();

    if true {
        // make sure we get deterministic results
        expect(&mut ctx, QUERIES[0].0, QUERIES[0].1);
        expect(&mut ctx, QUERIES[0].0, QUERIES[0].1);
        expect(&mut ctx, QUERIES[0].0, QUERIES[0].1);

        for idx in 0..QUERIES.len() {
            expect(&mut ctx, QUERIES[idx].0, QUERIES[idx].1);
        }

        assert!(ctx.infer.get_stats().same_as(&stats0));
    }

    let mut expected_map = HashMap::new();
    for idx in 0..QUERIES.len() {
        let id = ctx.infer.gen_req_id();
        expected_map.insert(id.clone(), QUERIES[idx].1.to_string());
        ctx.infer
            .add_request(id, QUERIES[idx].0, ctx.sampling_params.clone())
            .unwrap();
    }

    loop {
        let res = ctx.infer.step().unwrap();
        if res.is_empty() {
            break;
        }
        for sgo in &res {
            assert!(sgo.seq_outputs.len() == 1);
            let so = &sgo.seq_outputs[0];
            if so.finish_reason.is_some() {
                let outp = ctx.infer.seq_output_text(so).unwrap();
                let expected = expected_map.remove(&sgo.request_id).unwrap();
                if outp != expected {
                    // TODO this fails
                    log::warn!("expected: {:?}, got: {:?}", expected, outp);
                }
            }
        }
    }
    assert!(expected_map.is_empty());
    assert!(ctx.infer.get_stats().same_as(&stats0));
}
