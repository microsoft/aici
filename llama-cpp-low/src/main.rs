use llama_cpp_low::*;

fn main() {
    let mparams = ModelParams::default();
    let mut cparams = ContextParams::default();
    cparams.n_ctx = 2048;
    let mut model = Model::from_file("tmp/llama-2-7b-chat.Q5_K_M.gguf", mparams, cparams);
    let mut batch = Batch::new(512);
    for (idx, tok) in model
        .tokenize("Hello, my name is".as_bytes(), true, true)
        .iter()
        .enumerate()
    {
        batch.add_token(*tok, idx, 0, false)
    }

    let mut logit_idx = batch.len() - 1;
    let mut pos = batch.len();

    batch.enable_logits(logit_idx);

    for _ in 0..10 {
        model.decode(&mut batch).unwrap();
        let logits = model.get_logits(logit_idx);
        let top_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        println!(
            "top_idx: {:?} {:?}",
            top_idx,
            String::from_utf8_lossy(&model.token_to_bytes(top_idx as u32))
        );

        logit_idx = 0;
        batch.clear();
        batch.add_token(top_idx as u32, pos, 0, true);
        pos += 1;
    }
}
