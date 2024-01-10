declare module "_aici" {
  type Buffer = Uint8Array;

  /**
   * Return token indices for a given string (or byte sequence).
   */
  function tokenize(text: string | Buffer): number[];

  /**
   * Return byte (~string) representation of a given list of token indices.
   */
  function detokenize(tokens: number[]): Buffer;

  /**
   * Return identifier of the current sequence.
   * Most useful with fork_group parameter in mid_process() callback.
   * Best use aici.fork() instead.
   */
  function self_seq_id(): number;

  /**
   * Print out a message of the error and stop the program.
   */
  function panic(error: any): never;

  /**
   * Get the value of a shared variable.
   */
  function get_var(name: string): Buffer | null;

  /**
   * Set the value of a shared variable.
   */
  function set_var(name: string, value: string | Buffer): void;

  /**
   * Append to the value of a shared variable.
   */
  function append_var(name: string, value: string | Buffer): void;

  /**
   * Index of the end of sequence token.
   */
  function eos_token(): number;

  /**
   * Represents a set of tokens.
   * The value is true at indices corresponding to tokens in the set.
   */
  class TokenSet {
    /**
     * Create an empty set (with .length set to the total number of tokens).
     */
    constructor();

    add(t: number): void;
    delete(t: number): void;
    has(t: number): boolean;
    clear(): void;

    /**
     * Number of all tokens (not only in the set).
     */
    length: number;

    /**
     * Include or exclude all tokens from the set.
     */
    set_all(value: boolean): void;
  }

  /**
   * Initialize a constraint that allows any token.
   */
  class Constraint {
    constructor();

    /**
     * Check if the constraint allows the generation to end at the current point.
     */
    eos_allowed(): boolean;

    /**
     * Check if the constraint forces the generation to end at the current point.
     */
    eos_forced(): boolean;

    /**
     * Check if token `t` is allowed by the constraint.
     */
    token_allowed(t: number): boolean;

    /**
     * Update the internal state of the constraint to reflect that token `t` was appended.
     */
    append_token(t: number): void;

    /**
     * Set ts[] to True at all tokens that are allowed by the constraint.
     */
    allow_tokens(ts: TokenSet): void;
  }

  /**
   * A constraint that allows only tokens that match the regex.
   * The regex is implicitly anchored at the start and end of the generation.
   */
  function regex_constraint(pattern: string): Constraint;

  /**
   * A constraint that allows only tokens that match the specified yacc-like grammar.
   */
  function cfg_constraint(yacc_grammar: string): Constraint;

  /**
   * A constraint that allows only word-substrings of given string.
   */
  function substr_constraint(template: string, stop_at: string): Constraint;
}
