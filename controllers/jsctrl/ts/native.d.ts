// Top-level symbols

type Token = number;
type Buffer = Uint8Array;

/**
 * Force the exact tokens to be generated; usage: await $`Some text`
 */
declare function $(strings: TemplateStringsArray, ...values: any[]): Promise<void>;

/**
 * Throw an exception if the condition is not met.
 */
declare function assert(cond: boolean, msg?: string): asserts cond;

/**
 * Forces next tokens to be exactly the given text.
 */
declare function fixed(text: string): Promise<void>;

/**
 * Forks the execution into `numForks` branches.
 * @param numForks how many branches
 * @returns a number from 0 to `numForks`-1, indicating the branch
 */
declare function fork(numForks: number): Promise<number>;

/**
 * Suspends execution until all variables are available.
 * @param vars names of variables
 * @returns values of the variables
 */
declare function waitVars(...vars: string[]): Promise<Buffer[]>;

/**
 * Starts the AICI loop. 
 * @param f async function
 */
declare function start(f: () => Promise<void>): void;

/**
 * Specifies options for gen() and genTokens().
 */
interface GenOptions {
  /**
   * Make sure the generated text is one of the options.
   */
  options?: string[];
  /**
   * Make sure the generated text matches given regular expression.
   */
  regex?: string | RegExp;
  /**
   * Make sure the generated text matches given yacc-like grammar.
   */
  yacc?: string;
  /**
   * Make sure the generated text is a substring of the given string.
   */
  substring?: string;
  /**
   * Used together with `substring` - treat the substring as ending the substring
   * (typically '"' or similar).
   */
  substringEnd?: string;
  /**
   * Store result of the generation (as bytes) into a shared variable.
   */
  storeVar?: string;
  /**
   * Stop generation when the string is generated (the result includes the string and any following bytes (from the same token)).
   */
  stopAt?: string;
  /**
   * Stop generation when the given number of tokens have been generated.
   */
  maxTokens?: number;
}

/**
 * Generate a string that matches given constraints.
 * If the tokens do not map cleanly into strings, it will contain Unicode replacement characters.
 */
declare function gen(options: GenOptions): Promise<string>;

/**
 * Generate a list of tokens that matches given constraints.
 */
declare function genTokens(options: GenOptions): Promise<Token[]>;

// Extensions of JavaScript built-in types

interface String {
  /**
   * UTF-8 encode the current string.
   */
  toBuffer(): Uint8Array;
}

interface StringConstructor {
  /**
   * Create a string from UTF-8 buffer (with replacement character for invalid sequences)
   */
  fromBuffer(buffer: Uint8Array): string;
}

interface Uint8Array {
  /**
   * UTF-8 decode the current buffer.
   */
  decode(): string;
}

/** [MDN Reference](https://developer.mozilla.org/docs/Web/API/console) */
interface Console {
  /** [MDN Reference](https://developer.mozilla.org/docs/Web/API/console/debug) */
  debug(...data: any[]): void;
  /** [MDN Reference](https://developer.mozilla.org/docs/Web/API/console/error) */
  error(...data: any[]): void;
  /** [MDN Reference](https://developer.mozilla.org/docs/Web/API/console/info) */
  info(...data: any[]): void;
  /** [MDN Reference](https://developer.mozilla.org/docs/Web/API/console/log) */
  log(...data: any[]): void;
  /** [MDN Reference](https://developer.mozilla.org/docs/Web/API/console/trace) */
  trace(...data: any[]): void;
  /** [MDN Reference](https://developer.mozilla.org/docs/Web/API/console/warn) */
  warn(...data: any[]): void;
}

declare var console: Console;

// native module
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
   * Return debug string representation of a given token index
   */
  function tokenRepr(token: number): string;

  /**
   * Return debug string representation of a given token string
   */
  function tokensRepr(tokens: number[]): string;

  /**
   * Return identifier of the current sequence.
   * Most useful with fork_group parameter in mid_process() callback.
   * Best use aici.fork() instead.
   */
  function selfSeqId(): number;

  /**
   * Print out a message of the error and stop the program.
   */
  function panic(error: any): never;

  /**
   * Get the value of a shared variable.
   */
  function getVar(name: string): Buffer | null;

  /**
   * Set the value of a shared variable.
   */
  function setVar(name: string, value: string | Buffer): void;

  /**
   * Append to the value of a shared variable.
   */
  function appendVar(name: string, value: string | Buffer): void;

  /**
   * Index of the end of sequence token.
   */
  function eosToken(): number;

  /**
   * UTF-8 encode
   */
  function stringToBuffer(s: string): Buffer;

  /**
   * UTF-8 decode (with replacement character for invalid sequences)
   */
  function bufferToString(b: Buffer): string;

  /**
   * Return a string like `b"..."` that represents the given buffer.
   */
  function bufferRepr(b: Buffer): string;

  function _midProcessReturn(midProcessResult: any): void;

  /**
   * Represents a set of tokens.
   * The value is true at indices corresponding to tokens in the set.
   */
  class TokenSet {
    /**
     * Create an empty set (with .length set to the total number of tokens).
     */
    constructor();

    toString(): string;

    add(t: number): void;
    delete(t: number): void;
    has(t: number): boolean;
    clear(): void;

    /**
     * Number of all possible tokens (regardless of whether they are in the set or not).
     */
    length: number;

    /**
     * Include or exclude all tokens from the set.
     */
    setAll(value: boolean): void;

    /**
     * Number of tokens in the set.
     */
    numSet(): number;
  }

  /**
   * Initialize a constraint that allows any token.
   */
  class Constraint {
    constructor();

    /**
     * Check if the constraint allows the generation to end at the current point.
     */
    eosAllowed(): boolean;

    /**
     * Check if the constraint forces the generation to end at the current point.
     */
    eosForced(): boolean;

    /**
     * Check if token `t` is allowed by the constraint.
     */
    tokenAllowed(t: number): boolean;

    /**
     * Update the internal state of the constraint to reflect that token `t` was appended.
     */
    appendToken(t: number): void;

    /**
     * Set ts[] to True at all tokens that are allowed by the constraint.
     */
    allowTokens(ts: TokenSet): void;
  }

  /**
   * A constraint that allows only tokens that match the regex.
   * The regex is implicitly anchored at the start and end of the generation.
   */
  function regexConstraint(pattern: string): Constraint;

  /**
   * A constraint that allows only tokens that match the specified yacc-like grammar.
   */
  function cfgConstraint(yacc_grammar: string): Constraint;

  /**
   * A constraint that allows only word-substrings of given string.
   */
  function substrConstraint(template: string, stop_at: string): Constraint;
}
