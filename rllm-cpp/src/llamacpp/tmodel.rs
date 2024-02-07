use rllm::{
    config::{ModelMeta, RllmConfig}, seq::SchedulingPhase, AiciBias, HashMap, LoaderArgs, LogitsProcessor, ModelExec, SchedulerOutputs, TensorOps
};
use aicirt::{with_timer, TimerRef};
use anyhow::Result;
use llama_cpp_low as cpp;
use rand::distributions::Distribution as _;
use std::{sync::Arc, time::Instant};

use super::{
    blocks::CppBlockSpaceManager,
    loader::{load_model_config, load_rllm_engine},
    seqid::CppSequenceManager,
    Tensor,
};

pub struct TModel {
    pub(super) model: cpp::Model,
    seq_mgr: Arc<CppSequenceManager>,
    batch: cpp::Batch,
    seq_id_to_idx: HashMap<usize, usize>,
    t0: Instant,
    step_no: usize,
}

pub struct CppLoaderArgs {
    pub n_gpu_layers: Option<usize>,
    pub(crate) cached_model: Option<cpp::Model>,
}

impl CppLoaderArgs {
    pub fn new(n_gpu_layers: Option<usize>) -> Self {
        Self {
            n_gpu_layers,
            cached_model: None,
        }
    }
}

impl ModelExec for TModel {
    type Tensor = Tensor;
    type BlockSpaceManager = CppBlockSpaceManager;
    type AiciBias = CppAiciBias;
    type ModelConfig = ();
    type ModelLoaderArgs = CppLoaderArgs;
    type SequenceManager = CppSequenceManager;

    fn run(
        &mut self,
        _vocab_size: usize,
        tim: &TimerRef,
        step_no: usize,
        sched_out: &mut SchedulerOutputs,
    ) -> Result<()> {
        self.step_no = step_no;
        self.batch.clear();
        self.seq_id_to_idx.clear();

        for sg in sched_out.next_seq_groups.iter_mut() {
            for seq in sg.seqs.iter_mut() {
                if seq.sched_phase != SchedulingPhase::Running {
                    continue;
                }

                let seq_len = seq.get_len();
                let k_len = seq_len;
                log::trace!("fwd seq: {seq:?}");
                let mut q_len = seq.get_len() - seq.num_kv_computed;
                if q_len == 0 {
                    // just re-compute the last token
                    q_len = 1;
                }
                sg.usage.gen_tokens += 1;
                sg.usage.prompt_tokens += q_len;

                let off = k_len - q_len;
                for idx in off..off + q_len {
                    let logits = idx + 1 == off + q_len;
                    if logits {
                        self.seq_id_to_idx
                            .insert(seq.seq_id.to_num(), self.batch.len());
                    }
                    self.seq_mgr.with_cpp(seq.seq_id, |cpp| {
                        cpp.assert_model(&self.model);
                        self.batch.add_token(seq.get_token(idx), idx, &cpp, logits);
                    });
                }

                seq.sync_computed_kv();
            }
        }

        log::trace!("batch_info #{}; {:?}", self.step_no, self.batch);

        self.t0 = Instant::now();

        with_timer!(tim, { self.model.decode(&mut self.batch)? });

        Ok(())
    }

    fn get_logits(&self, seq_id: usize) -> Tensor {
        let l = self.model.get_logits(self.seq_id_to_idx[&seq_id]);
        Tensor::from_slice(l)
    }

    fn finalize_run(&mut self) -> Result<()> {
        let dur = self.t0.elapsed().as_micros() as f64 / 1000.0;

        let ntok = self.batch.len();

        log::info!(
            "model forward: step #{} {:.2}ms; {} tok(s); {:.1}tps",
            self.step_no,
            dur,
            ntok,
            ntok as f64 / (dur / 1000.0),
        );

        Ok(())
    }

    fn empty_bias(&self, vocab_size: usize) -> Self::AiciBias {
        CppAiciBias {
            vocab_size,
            bias: None,
        }
    }

    fn new_bias(
        &self,
        slice: &'static [f32],
        num_seqs: usize,
        vocab_size: usize,
    ) -> Self::AiciBias {
        let tensor = {
            assert!(slice.len() == num_seqs * vocab_size);
            Tensor::from_slice(slice)
        };
        CppAiciBias {
            vocab_size,
            bias: Some(tensor),
        }
    }

    fn sample(&self, state: &mut LogitsProcessor, logits: &Tensor) -> Result<u32> {
        let next_token = match state.temperature {
            None => self.sample_argmax(&logits),
            Some(temperature) => {
                let mut prs: Vec<f32> = logits.to_vec1();
                let temp = (1.0 / temperature) as f32;
                for idx in 0..prs.len() {
                    prs[idx] *= temp;
                }
                let top_p = state.top_p;
                if top_p <= 0.0 || top_p >= 1.0 {
                    self.sample_multinomial(state, &prs)?
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(state, &mut prs, top_p as f32)?
                }
            }
        };
        Ok(next_token)
    }

    fn load_model_config(
        args: &LoaderArgs,
        model_args: &mut Self::ModelLoaderArgs,
    ) -> Result<(ModelMeta, Self::ModelConfig)> {
        let meta = load_model_config(args, model_args)?;
        Ok((meta, ()))
    }

    fn verify_args(_args: &RllmConfig<Self>) -> Result<()> {
        Ok(())
    }

    fn load_rllm_engine(
        args: LoaderArgs,
        model_args: Self::ModelLoaderArgs,
    ) -> Result<rllm::RllmEngine<Self>> {
        load_rllm_engine(args, model_args)
    }

    fn sequence_manager(&self) -> Arc<Self::SequenceManager> {
        self.seq_mgr.clone()
    }
}

impl TModel {
    pub fn new(config: Arc<RllmConfig<Self>>, model: cpp::Model) -> Self {
        let batch = cpp::Batch::new(config.scheduler.max_num_batched_tokens);
        let seq_mgr = Arc::new(CppSequenceManager::new(model.clone()));
        Self {
            model,
            batch,
            seq_id_to_idx: HashMap::default(),
            step_no: 0,
            seq_mgr,
            t0: Instant::now(),
        }
    }

    fn sample_argmax(&self, logits: &Tensor) -> u32 {
        let data = logits.as_slice();
        let mut top = data[0];
        let mut top_idx = 0;
        for (i, x) in data.iter().enumerate() {
            if *x > top {
                top = *x;
                top_idx = i;
            }
        }
        top_idx as u32
    }

    fn sample_multinomial(&self, state: &mut LogitsProcessor, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(prs)?;
        let next_token = distr.sample(&mut state.rng) as u32;
        Ok(next_token)
    }

    fn sample_topp(
        &self,
        state: &mut LogitsProcessor,
        prs: &mut Vec<f32>,
        top_p: f32,
    ) -> Result<u32> {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability top_p. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].partial_cmp(&prs[i]).unwrap());

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        // Sample with clamped probabilities.
        self.sample_multinomial(state, prs)
    }
}

pub struct CppAiciBias {
    pub vocab_size: usize,
    pub bias: Option<Tensor>,
}

impl AiciBias<Tensor> for CppAiciBias {
    fn apply(&self, logits: &mut Tensor, seq_id: usize) {
        let bias = self.bias.as_ref().unwrap();
        let sp = seq_id * self.vocab_size;
        let logits = logits.as_mut_slice();
        let bias = bias.as_slice();
        for i in 0..self.vocab_size {
            logits[i] += bias[sp + i];
        }
    }
}
