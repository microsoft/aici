use std::{collections::HashMap, sync::Arc};

use candle_core::Tensor;

use crate::{
    openai::sampling_params::{SamplingParams, SamplingType},
    paged_attention::sequence::SequenceData,
};

pub struct SamplingMetadata {
    pub seq_groups: Vec<(Vec<usize>, SamplingParams)>,
    pub seq_data: HashMap<usize, Arc<SequenceData>>,
    pub prompt_lens: Vec<usize>,
    pub selected_token_indices: Tensor,
    pub categorized_sample_indices: HashMap<SamplingType, Tensor>,
}
