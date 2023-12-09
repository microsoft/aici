pub mod llama;
pub mod mistral;

pub trait ConfigLike {
    fn get_num_kv_heads(&self) -> usize;
    fn get_hidden_size(&self) -> usize;
    fn get_num_hidden_layers(&self) -> usize;
    fn get_num_attention_heads(&self) -> usize;
    fn get_vocab_size(&self) -> usize;
    fn get_sliding_window(&self) -> Option<usize>;
}
