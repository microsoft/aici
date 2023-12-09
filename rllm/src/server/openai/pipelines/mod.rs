use std::{
    collections::{HashMap, HashSet},
    env,
    path::PathBuf,
    sync::Arc,
};

use actix_web::web::Bytes;
use candle_core::{DType, Device, Tensor};
use tokenizers::Encoding;
use tokio::sync::mpsc::Sender;

use crate::{
    openai::sampling_params::SamplingType,
    paged_attention::{
        cache_engine::CacheConfig,
        input_metadata::InputMetadata,
        scheduler::SchedulerConfig,
        sequence::{SequenceData, SequenceGroupMetadata, SequenceGroupOutput},
    },
};

use self::sampling_metadata::SamplingMetadata;

use super::{
    conversation::Conversation,
    models::ConfigLike,
    responses::{APIError, ChatChoice, ChatCompletionUsageResponse},
    sampling_params::{EarlyStoppingCondition, SamplingParams},
    streaming::SenderError,
    PipelineConfig, TokenizerWrapper,
};

pub mod llama;
pub mod llm_engine;
pub mod mistral;
pub mod sampling_metadata;

const PAD_SLOT_ID: usize = usize::MAX;

/// A module pipeline that encompasses the inference pass, tokenizer, and conversation.
pub trait ModulePipeline<'s>: Send + Sync {
    fn forward(
        &mut self,
        xs: &Encoding,
        sampling: SamplingParams,
        device: Device,
        streamer: Option<Sender<Result<Bytes, SenderError>>>,
    ) -> Result<(Option<Vec<ChatChoice>>, ChatCompletionUsageResponse), APIError>;

    fn name(&self) -> &str;

    fn tokenizer(&self) -> &dyn TokenizerWrapper<'s, String>;

    fn get_conversation(&mut self) -> &mut dyn Conversation;

    fn get_scheduler_config(&self) -> &SchedulerConfig;

    fn get_model_config(&self) -> Box<dyn ConfigLike>;

    fn profile_run(&mut self) -> Result<(), APIError> {
        let vocab_size = self.get_model_config().get_vocab_size();
        let sampling_params = SamplingParams::new(
            1,
            None,
            0.,
            0.,
            1.,
            1.,
            0.99,
            (vocab_size - 1).try_into().unwrap(),
            false,
            1.,
            EarlyStoppingCondition::UnlikelyBetterCandidates,
            None,
            Vec::new(),
            false,
            16,
            None,
            None,
            true,
        )?;
        let max_num_batched_tokens = self.get_scheduler_config().max_num_batched_tokens;
        let max_num_seqs = self.get_scheduler_config().max_num_seqs;

        let seqs = Vec::new();
        for group_id in 0..max_num_seqs {
            let seq_len = max_num_batched_tokens / max_num_seqs
                + (group_id < max_num_batched_tokens % max_num_seqs) as usize;
            let seq_data = SequenceData {
                prompt_token_ids: vec![0].repeat(seq_len),
                output_token_ids: vec![],
            };
            let seq = SequenceGroupMetadata {
                request_id: group_id.to_string(),
                is_prompt: true,
                seq_data: HashMap::from([(group_id, Arc::new(seq_data))]),
                sampling_params,
                block_tables: None,
            };
            seqs.push(seq);
        }

        self.execute_model(seqs, None);
        Ok(())
    }

    fn _get_block_size(&self) -> &Option<usize>;

    fn set_block_size(&mut self, size: usize);

    // Calls the module pipeline model executer
    fn execute_model(
        &mut self,
        seq_group_metadatas: Vec<SequenceGroupMetadata>,
        kv_cache: Option<Arc<Vec<(Tensor, Tensor)>>>,
    ) -> Result<(), APIError> {
        //https://github.com/vllm-project/vllm/blob/24f60a54f42076e0bfa49fde113756bf4e95f9ef/vllm/worker/model_runner.py#L259
        let (input_tokens, input_positions, input_metadata) =
            if seq_group_metadatas.first().unwrap().is_prompt {
                self.prepare_decode(seq_group_metadatas)?
            } else {
                self.prepare_decode(seq_group_metadatas)?
            };
        let sampling_metadata =
            self.prepare_sample(seq_group_metadatas, input_metadata.prompt_lens);

        todo!()
    }

    fn prepare_sample(
        &self,
        seq_group_metadatas: Vec<SequenceGroupMetadata>,
        prompt_lens: Vec<usize>,
    ) -> Result<SamplingMetadata, APIError> {
        let mut seq_groups = Vec::new();
        let mut selected_token_indices = Vec::new();
        let mut selected_token_start_idx = 0;
        let mut categorized_sample_indices = HashMap::from([
            (SamplingType::BEAM, vec![]),
            (SamplingType::GREEDY, vec![]),
            (SamplingType::RANDOM, vec![]),
        ]);
        let mut categorized_sample_indices_start_idx = 0;

        let max_prompt_len = if !prompt_lens.is_empty() {
            *prompt_lens.iter().max().unwrap()
        } else {
            1
        };

        for (i, seq_group_metadata) in seq_group_metadatas.iter().enumerate() {
            let seq_ids = seq_group_metadata.seq_data.keys();
            let sampling_params = &seq_group_metadata.sampling_params;
            seq_groups.push((
                seq_ids.copied().collect::<Vec<_>>(),
                sampling_params.clone(),
            ));

            if seq_group_metadata.is_prompt {
                assert_eq!(seq_ids.len(), 1);
                let prompt_len = prompt_lens.get(i).unwrap();
                categorized_sample_indices
                    .get_mut(&sampling_params.sampling_type())
                    .unwrap()
                    .push(categorized_sample_indices_start_idx);
                categorized_sample_indices_start_idx += 1;

                selected_token_start_idx += max_prompt_len;
            } else {
                let num_seqs = seq_ids.len();
                selected_token_indices
                    .extend(selected_token_start_idx..selected_token_start_idx + num_seqs);
                selected_token_start_idx += num_seqs;

                categorized_sample_indices
                    .get_mut(&sampling_params.sampling_type())
                    .unwrap()
                    .extend(
                        categorized_sample_indices_start_idx
                            ..categorized_sample_indices_start_idx + num_seqs,
                    );
                categorized_sample_indices_start_idx += num_seqs;
            }
        }

        let selected_token_indices = Tensor::new(
            selected_token_indices
                .iter()
                .map(|x| *x as f64)
                .collect::<Vec<_>>(),
            &Device::new_cuda(0).map_err(APIError::from)?,
        )
        .map_err(APIError::from)?
        .to_dtype(DType::I64)
        .map_err(APIError::from)?;

        let mut categorized_sample_indices_new = HashMap::new();
        for (t, seq_ids) in categorized_sample_indices {
            categorized_sample_indices_new.insert(
                t,
                Tensor::new(
                    seq_ids.iter().map(|x| *x as f64).collect::<Vec<_>>(),
                    &Device::new_cuda(0).map_err(APIError::from)?,
                )
                .map_err(APIError::from)?
                .to_dtype(DType::I64)
                .map_err(APIError::from)?,
            );
        }

        let seq_data = HashMap::new();
        for seq_group_metadata in seq_group_metadatas {
            seq_data.extend(seq_group_metadata.seq_data);
        }

        Ok(SamplingMetadata {
            seq_groups,
            seq_data,
            prompt_lens,
            selected_token_indices,
            categorized_sample_indices: categorized_sample_indices_new,
        })
    }

    /// input toks, input positions, input metadata
    fn prepare_prompt(
        &self,
        seq_group_metadatas: Vec<SequenceGroupMetadata>,
    ) -> Result<(Tensor, Tensor, InputMetadata), APIError> {
        let mut prompt_lens = Vec::new();
        let mut slot_mapping = Vec::new();

        let mut input_tokens = Vec::new();
        let mut input_positions = Vec::new();

        for seq_group_metadata in seq_group_metadatas {
            assert!(seq_group_metadata.is_prompt);
            let seq_ids = seq_group_metadata.seq_data.keys();
            assert_eq!(seq_ids.len(), 1);
            let seq_id = seq_ids.nth(0).unwrap();

            let seq_data = seq_group_metadata.seq_data.get(seq_id).unwrap();
            let prompt_tokens = &seq_data.prompt_token_ids;
            let prompt_len = prompt_tokens.len();
            prompt_lens.push(prompt_len);

            input_tokens.push(prompt_tokens.clone());
            input_positions.push((0..prompt_len).collect::<Vec<_>>());

            // During profiling, block tables are uninitialized. Use a dummy mapping.
            if let Some(block_table) = seq_group_metadata.block_tables {
                slot_mapping.push(vec![]);
                let block_table = block_table.get(seq_id).unwrap();
                let mut start_idx =
                    if let Some(sliding_window) = self.get_model_config().get_sliding_window() {
                        0.max(prompt_len - sliding_window)
                    } else {
                        0
                    };
                for i in 0..prompt_len {
                    if i < start_idx {
                        slot_mapping.last_mut().unwrap().push(PAD_SLOT_ID);
                        continue;
                    }

                    let block_number = block_table
                        .get(i / self._get_block_size().unwrap())
                        .unwrap();
                    let block_offset = i % self._get_block_size().unwrap();
                    let slot = block_number * self._get_block_size().unwrap() + block_offset;
                    slot_mapping.last_mut().unwrap().push(slot);
                }
            } else {
                slot_mapping.push(vec![PAD_SLOT_ID].repeat(prompt_len));
            }
        }

        let input_tokens = _make_tensor_with_pad(input_tokens, prompt_lens.len(), 0, DType::I64)?;
        let input_positions =
            _make_tensor_with_pad(input_positions, prompt_lens.len(), 0, DType::I64)?;

        Ok((
            input_tokens,
            input_positions,
            InputMetadata::new(
                prompt_lens,
                None,
                None,
                None,
                _make_tensor_with_pad(slot_mapping, prompt_lens.len(), PAD_SLOT_ID, DType::I64)?,
            ),
        ))
    }

    /// input toks, input positions, input metadata
    fn prepare_decode(
        &self,
        seq_group_metadatas: Vec<SequenceGroupMetadata>,
    ) -> Result<(Tensor, Tensor, InputMetadata), APIError> {
        let mut slot_mapping = Vec::new();

        let mut input_tokens = Vec::new();
        let mut input_positions = Vec::new();

        let mut context_lens = Vec::new();
        let mut block_tables = Vec::new();

        for sequence_group_metadata in seq_group_metadatas {
            assert!(!sequence_group_metadata.is_prompt);

            let seq_data = &sequence_group_metadata.seq_data;
            for seq_id in seq_data.keys() {
                let seq_data = seq_data.get(seq_id).unwrap();
                let generation_token = seq_data.get_last_token_id();
                input_tokens.push(vec![*generation_token]);

                let context_len =
                    if let Some(sliding_window) = self.get_model_config().get_sliding_window() {
                        seq_data.prompt_token_ids.len()
                            + seq_data.output_token_ids.len().min(sliding_window)
                    } else {
                        seq_data.prompt_token_ids.len() + seq_data.output_token_ids.len()
                    };
                context_lens.push(context_len);

                let position = context_len - 1;
                input_positions.push(vec![position]);

                let mut block_table = sequence_group_metadata
                    .block_tables
                    .unwrap()
                    .get(seq_id)
                    .unwrap()
                    .clone();
                let block_number = block_table
                    .get(position / self._get_block_size().unwrap())
                    .unwrap();
                let block_offset = position % self._get_block_size().unwrap();
                let slot = block_number * self._get_block_size().unwrap() + block_offset;
                slot_mapping.push(vec![slot]);

                if let Some(sliding_window) = self.get_model_config().get_sliding_window() {
                    let sliding_window_blocks = sliding_window / self._get_block_size().unwrap();
                    block_table = block_table
                        [block_table.len() - sliding_window_blocks..block_table.len()]
                        .to_vec();
                }
                block_tables.push(block_table);
            }
        }

        let input_tokens = _make_tensor_with_pad(input_tokens, 1, 0, DType::I64)?;
        let input_positions = _make_tensor_with_pad(input_positions, 1, 0, DType::I64)?;
        let slot_mapping = _make_tensor_with_pad(slot_mapping, 1, PAD_SLOT_ID, DType::I64)?;

        let max_context_len = context_lens.len();
        let context_lens = Tensor::new(
            context_lens.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            &Device::new_cuda(0).map_err(APIError::from)?,
        )
        .map_err(APIError::from)?
        .to_dtype(DType::I64)
        .map_err(APIError::from)?;

        let max_block_table_len = block_tables.iter().map(|t| t.len()).max().unwrap();
        let block_tables = _make_tensor_with_pad(block_tables, max_block_table_len, 0, DType::I64)?;

        Ok((
            input_tokens,
            input_positions,
            InputMetadata::new(
                vec![],
                Some(max_context_len),
                Some(block_tables),
                Some(context_lens),
                slot_mapping,
            ),
        ))
    }
}

fn _make_tensor_with_pad(
    x: Vec<Vec<usize>>,
    max_len: usize,
    pad: usize,
    dtype: DType,
) -> Result<Tensor, APIError> {
    let mut padded_x = Vec::new();
    for mut x_i in x {
        assert!(x_i.len() <= max_len);
        x_i.extend(vec![pad].repeat(max_len - x_i.len()));
        let x_i = x_i.iter().map(|x| *x as f64).collect::<Vec<_>>();
        padded_x.push(
            Tensor::new(x_i, &Device::new_cuda(0).map_err(APIError::from)?)
                .map_err(APIError::from)?,
        );
    }
    Tensor::cat(&padded_x[..], 0).map_err(APIError::from)
}

pub(crate) fn read_env_var(var: String) -> Result<String, APIError> {
    env::var(var).map_err(APIError::from)
}

pub trait ModelPaths {
    fn get_weight_filenames(&self) -> &Vec<PathBuf>;
    fn get_config_filename(&self) -> &PathBuf;
    fn get_tokenizer_filename(&self) -> &PathBuf;
}

pub trait ModelLoader<'a> {
    fn download_model(
        &self,
        model_id: String,
        revision: Option<String>,
        hf_token: Option<String>,
    ) -> Result<Box<dyn ModelPaths>, APIError>;

    fn load_model(
        &self,
        paths: Box<dyn ModelPaths>,
        dtype: DType,
        device: Device,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> Result<(Box<dyn ModulePipeline<'a>>, PipelineConfig), APIError>;
}
