use rllm::{LlamaInfer, LoaderArgs, LogitsProcessor};

struct Ctx {
    infer: LlamaInfer,
    logits_processor: LogitsProcessor,
}

impl Ctx {
    fn new() -> Self {
        Self {
            infer: LlamaInfer::load(LoaderArgs::default()).unwrap(),
            logits_processor: LogitsProcessor::new(42, Some(0.0), None),
        }
    }
}

fn expect(ctx: &mut Ctx, prompt: &str, expected: &str) {
    let gen = ctx
        .infer
        .generate(prompt, 10, &mut ctx.logits_processor)
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

    // make sure we get deterministic results
    for _ in 0..2 {
        expect(&mut ctx, QUERIES[0].0, QUERIES[0].1);
    }

    for idx in 0..QUERIES.len() {
        expect(&mut ctx, QUERIES[idx].0, QUERIES[idx].1);
    }

    ctx.infer.alt = 1;
    for idx in 0..QUERIES.len() {
        expect(&mut ctx, QUERIES[idx].0, QUERIES[idx].1);
    }
}
