//
// This interface needs to be implemented by the WASM binary
//

// Tokens are assumed to be at most 32 bit.
// Typical models range 30k (LLAMA) to 100k (GPT4) tokens.
typedef uint32_t token_t;

// Called first, after instantiating WASM module.
void aici_init(void);

// Called once per module, to get an AICI for a specific query
Aici *aici_create(void);

// These two are called after aici_create() on the fresh AICI.
// They should return the buffers that the WASM code has to allocated and keep around
// until relevant aici_free().

// Return buffer where the prompt will be written. `size` is number of tokens in the prompt.
token_t *aici_get_prompt_buffer(Aici *aici, uint32_t size);

// Return the buffer where the WASM code will write logit biases after
// aici_process_prompt() and aici_append_token().
// Size of number of biases (which equals size of the vocabulary).
float *aici_get_logit_bias_buffer(Aici *aici, uint32_t size);

// This called once, when the AICI should process the prompt in its buffer.
// It should set the values in logit bias buffer.
void aici_process_prompt(Aici *aici);
// The logical type (if WASM would allow such things) of this function is:
// float[vocab_size] aici_process_prompt(Aici *aici, token_t[] prompt);

// This is called after a token is sampled.
// It should set the values in logit bias buffer.
void aici_append_token(Aici *aici, token_t tok);
// The logical type (if WASM would allow such things) of this function is:
// float[vocab_size] aici_append_token(Aici *aici, token_t tok);

//
// This interface is available to the WASM binary
//

// Log a string.
void aici_host_print(const uint8_t *ptr, uint32_t size);

// Read binary representation of TokTrie.
// Always returns the size of the trie, will write up to `size` bytes to `dst`.
uint32_t aici_host_read_token_trie(uint8_t *dst, uint32_t size);

// Similar, for argument passed by the user (typically JSON).
uint32_t aici_host_read_arg(uint8_t *dst, uint32_t size);
