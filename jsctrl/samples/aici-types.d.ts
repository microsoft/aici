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
    setAll(value: boolean): void;
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
declare module 'aici' {
/// 
import { TokenSet, tokenize, detokenize, regexConstraint, cfgConstraint, substrConstraint, Constraint, getVar, setVar, appendVar, eosToken, panic } from "_aici";
export { TokenSet, tokenize, detokenize, getVar, setVar, appendVar, eosToken };
export type SeqId = number;
type int = number;
/**
 * Return debug representation of the argument, suitable for printing in the console.
 */
export function inspect(v: any): string;
export function log(...args: any[]): void;
export class AssertionError extends Error {
}
/**
 * Throw an exception if the condition is not met.
 */
export function assert(cond: boolean, msg?: string): asserts cond;
/**
 * Get list of tokens in the current sequence, including the prompt.
 */
export function getTokens(): Token[];
export class MidProcessResult {
    _n_skip_me: boolean;
    _n_stop: boolean;
    _n_logit_bias: TokenSet | null;
    _n_backtrack: number;
    _n_ff_tokens: Token[];
    constructor();
    static stop(): MidProcessResult;
    static skipMe(): MidProcessResult;
    static bias(bias: TokenSet): MidProcessResult;
    static splice(backtrack: number, tokens: Token[]): MidProcessResult;
}
export class PreProcessResult {
    _n_suspended: boolean;
    _n_ff_tokens: Token[];
    _n_attention_masks: number[][];
    constructor();
    static continue_(): PreProcessResult;
    static suspend(): PreProcessResult;
    static fork(numForks: number): PreProcessResult;
    static ffTokens(toks: Token[]): PreProcessResult;
}
export class PostProcessResult {
    _n_stop_seq: boolean;
    constructor(stop_seq?: boolean);
    static continue_(): PostProcessResult;
    static stop(): PostProcessResult;
    static fromTokens(tokens: Token[]): PostProcessResult;
}
export class NextToken {
    finished: boolean;
    currTokens: Token[] | null;
    forkGroup: SeqId[];
    _resolve?: (value: Token[]) => void;
    constructor();
    /**
     * Awaiting this will return generated token (or tokens, if fast-forwarding requested by self.mid_process()).
     * You have only ~1ms to process the results before awaiting a new instance of NextToken() again.
     */
    run(): Promise<Token[]>;
    /**
     * Override to suspend, if the model cannot continue generating tokens
     * now (for example, not all variables are available to compute bias).
     * ~1ms time limit.
     */
    preProcess(): PreProcessResult;
    /**
     * This can be overridden to return a bias, fast-forward tokens, backtrack etc.
     * ~20ms time limit.
     */
    midProcess(): MidProcessResult;
    /**
     * This can be overridden to do something with generated tokens.
     * ~1ms time limit.
     * @param tokens tokens generated in the last step
     */
    postProcess(tokens: Token[]): PostProcessResult;
    _pre_process(): PreProcessResult;
    _mid_process(fork_group: SeqId[]): MidProcessResult;
    _post_process(_backtrack: int, tokens: Token[]): PostProcessResult;
    private reset;
}
/**
 * Forces next tokens to be exactly the given text.
 */
export function fixed(text: string): Promise<void>;
/**
 * Force the exact tokens to be generated; usage: await $`Some text`
 */
export function $(strings: TemplateStringsArray, ...values: any[]): Promise<void>;
/**
 * Forces next tokens to be exactly the given text.
 * If following is given, the text replaces everything that follows the label.
 */
class FixedTokens extends NextToken {
    fixedTokens: Token[];
    following: Label | null;
    constructor(text: string | Buffer, following?: Label | null);
    preProcess(): PreProcessResult;
    midProcess(): MidProcessResult;
}
/**
 * Indicates that the generation should stop.
 */
class StopToken extends NextToken {
    constructor();
    midProcess(): MidProcessResult;
    postProcess(_tokens: Token[]): PostProcessResult;
}
/**
 * Generates a token that satisfies the given constraint.
 * The constraint will be constructed in mid_process() phase, which has slightly longer time limit.
 */
export class ConstrainedToken extends NextToken {
    mkConstraint: () => Constraint;
    _constraint: Constraint | null;
    constructor(mkConstraint: () => Constraint);
    midProcess(): MidProcessResult;
    postProcess(tokens: Token[]): PostProcessResult;
}
export class PreToken extends NextToken {
    midProcess(): MidProcessResult;
}
/**
 * Forks the execution into `numForks` branches.
 * @param numForks how many branches
 * @returns a number from 0 to `numForks`-1, indicating the branch
 */
export function fork(numForks: number): Promise<number>;
/**
 * Suspends execution until all variables are available.
 * @param vars names of variables
 * @returns values of the variables
 */
export function waitVars(...vars: string[]): Promise<Buffer[]>;
/**
 *  Low-level interface for AICI. Use aici.start() to wrap a coroutine.
 */
export interface AiciCallbacks {
    init_prompt(prompt: Token[]): void;
    pre_process(): PreProcessResult;
    mid_process(fork_group: SeqId[]): MidProcessResult;
    post_process(backtrack: number, tokens: Token[]): PostProcessResult;
}
export type CbType = NextToken;
export class AiciAsync implements AiciCallbacks {
    static instance: AiciAsync;
    _tokens: Token[];
    private _pendingCb;
    private _token;
    private midProcessReEntry;
    _nextToken(t: NextToken): void;
    constructor(f: () => Promise<void>);
    step(tokens: Token[]): void;
    init_prompt(prompt: Token[]): void;
    pre_process(): PreProcessResult;
    mid_process(fork_group: SeqId[]): MidProcessResult;
    post_process(backtrack: number, tokens: Token[]): PostProcessResult;
}
/**
 * Starts the AICI loop.
 * @param f async function
 */
export function start(f: () => Promise<void>): AiciAsync;
/**
 * Runs the loop as a test.
 */
export function test(f: () => Promise<void>): AiciAsync;
export class Label {
    ptr: number;
    /**
     * Create a new label the indicates the current position in the sequence.
     * Can be passed as `following=` argument to `FixedTokens()`.
     */
    constructor();
    /**
     * Return tokens generated since the label.
     */
    tokensSince(): Token[];
    /**
     * Return text generated since the label.
     */
    textSince(): string;
    /**
     * Generate given prompt text, replacing all text after the current label.
     */
    fixedAfter(text: string): Promise<void>;
}
export class ChooseConstraint extends Constraint {
    ptr: number;
    options: Token[][];
    constructor(options: string[]);
    eosAllowed(): boolean;
    eosForced(): boolean;
    tokenAllowed(t: Token): boolean;
    appendToken(t: Token): void;
    allowTokens(ts: TokenSet): void;
}
export function genTokens(options: GenOptions): Promise<Token[]>;
export function gen(options: GenOptions): Promise<string>;
export function checkVar(name: string, value: string): void;
export function checkVars(d: Record<string, string>): void;
export const helpers: {
    regex_constraint: typeof regexConstraint;
    cfg_constraint: typeof cfgConstraint;
    substr_constraint: typeof substrConstraint;
    FixedTokens: typeof FixedTokens;
    StopToken: typeof StopToken;
    panic: typeof panic;
};

}
