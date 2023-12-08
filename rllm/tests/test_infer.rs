use std::collections::HashMap;

use rllm::{config::SamplingParams, LoaderArgs, RllmEngine};

struct Ctx {
    engine: RllmEngine,
    sampling_params: SamplingParams,
}

impl Ctx {
    fn new() -> Self {
        let mut sampling_params = SamplingParams::default();
        sampling_params.max_tokens = 10;
        Self {
            engine: RllmEngine::load(LoaderArgs::default()).unwrap(),
            sampling_params,
        }
    }
}

fn expect(ctx: &mut Ctx, prompt: &str, expected: &str) {
    let gen = ctx
        .engine
        .generate(prompt, ctx.sampling_params.clone())
        .unwrap();
    if gen != expected {
        panic!("expected: {:?}, got: {:?}", expected, gen);
    }
}

/*
This is more examples of strongly pre-determined (top logit is at least 5% higher than the second for all tokens) 
results for Llama-2-7b-hf:

FOUND "April 2022 The Moon is a relatively large, terrestrial," "planetary-mass object in the Solar System"
FOUND "about one-quarter of Earth's. It is the largest moon" "in the Solar System relative to the size of"
FOUND "amounts of carbon dioxide and other gaseous molecules.[207] Water" "vapor is the most abundant greenhouse gas"
FOUND "from the Sun—and the equinoxes, when Earth's rotational axis" "is perpendicular to the Sun's ray"
FOUND "in 3.48 billion-year-old sandstone in Western Australia,[66] biogenic" "graphite in 3.7 billion-year"
FOUND "km (5 mi) at the poles to 17 km" "(11 mi) at the equator."
FOUND "mi) at the poles to 17 km" "(11 mi) at the equator."
FOUND "ocean-floor sediments, the oceanic crust is predominantly basaltic, while" "the continental crust is predominantly gran"
FOUND "over millions of years," "the earth has been shaped by the forces of"
FOUND "sea level averages 101.325 kPa (14.696 psi),[207]" "and the highest recorded sea level was 10"
FOUND "seasonal change in climate, with summer in the Northern" "Hemisphere and winter in the Southern Hemis"
FOUND "species.[243] Extreme weather, such as tropical cyclones (including" "hurricanes and typhoons),"
FOUND "to 100 million years to form.[37] Estimates" "of the age of the Earth range from 4"
FOUND "dry atmosphere is composed of 78.084% nitrogen," "20.95% oxygen,"
FOUND "6] North American Plate Eurasian Plate Antarctic Plate Indo-Australian" "Plate Pacific Plate African Plate South American"
FOUND "that has ever been habitable for life. Earth's" "atmosphere is a thin layer of gases that surr"
*/

const QUERIES: &'static [(&'static str, &'static str)] = &[
    (
        "can be reconciled by",
        "the fact that the two are not the same.",
    ),
    (
        "leakage of hydrogen into space contributes",
        "to the formation of the hydrogen envelope.",
    ),
    (
        "perihelion occurs around 3 January,",
        "and aphelion around 4 July.",
    ),
];

#[test]
fn test_infer1() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut ctx = Ctx::new();

    let stats0 = ctx.engine.get_stats();

    if true {
        // make sure we get deterministic results
        expect(&mut ctx, QUERIES[0].0, QUERIES[0].1);
        expect(&mut ctx, QUERIES[0].0, QUERIES[0].1);
        expect(&mut ctx, QUERIES[0].0, QUERIES[0].1);

        for idx in 0..QUERIES.len() {
            expect(&mut ctx, QUERIES[idx].0, QUERIES[idx].1);
        }

        assert!(ctx.engine.get_stats().same_as(&stats0));
    }

    let mut expected_map = HashMap::new();
    for idx in 0..QUERIES.len() {
        let id = ctx.engine.gen_req_id();
        expected_map.insert(id.clone(), QUERIES[idx].1.to_string());
        ctx.engine
            .add_request(id, QUERIES[idx].0, ctx.sampling_params.clone())
            .unwrap();
    }

    loop {
        let res = ctx.engine.step().unwrap();
        if res.is_empty() {
            break;
        }
        for sgo in &res {
            assert!(sgo.seq_outputs.len() == 1);
            let so = &sgo.seq_outputs[0];
            if so.finish_reason.is_some() {
                let outp = ctx.engine.seq_output_text(so).unwrap();
                let expected = expected_map.remove(&sgo.request_id).unwrap();
                if outp != expected {
                    // TODO this fails
                    panic!("expected: {:?}, got: {:?}", expected, outp);
                }
            }
        }
    }
    assert!(expected_map.is_empty());
    assert!(ctx.engine.get_stats().same_as(&stats0));
}
