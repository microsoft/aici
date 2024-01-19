/// <reference path="../native.d.ts" />
import { TokenSet, tokenize, detokenize, regexConstraint, cfgConstraint, substrConstraint, Constraint, getVar, setVar, appendVar, eosToken, panic } from "_aici";
export { TokenSet, tokenize, detokenize, getVar, setVar, appendVar, eosToken };
export type SeqId = number;
type int = number;
export declare function inspect(v: any): string;
export declare function log(...args: any[]): void;
export declare class AssertionError extends Error {
}
/**
 * Throw an exception if the condition is not met.
 */
export declare function assert(cond: boolean, msg?: string): asserts cond;
/**
 * Get list of tokens in the current sequence, including the prompt.
 */
export declare function getTokens(): Token[];
/**
 * Get the length of the prompt in the current sequence.
 */
export declare function getPromptLen(): number;
export declare class MidProcessResult {
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
export declare class PreProcessResult {
    _n_suspended: boolean;
    _n_ff_tokens: Token[];
    _n_attention_masks: number[][];
    constructor();
    static continue_(): PreProcessResult;
    static suspend(): PreProcessResult;
    static fork(numForks: number): PreProcessResult;
    static ffTokens(toks: Token[]): PreProcessResult;
}
export declare class PostProcessResult {
    _n_stop_seq: boolean;
    constructor(stop_seq?: boolean);
    static continue_(): PostProcessResult;
    static stop(): PostProcessResult;
    static fromTokens(tokens: Token[]): PostProcessResult;
}
export declare class NextToken {
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
export declare function fixed(text: string): Promise<void>;
/**
 * Force the exact tokens to be generated; usage: await $`Some text`
 */
export declare function $(strings: TemplateStringsArray, ...values: any[]): Promise<void>;
/**
 * Forces next tokens to be exactly the given text.
 * If following is given, the text replaces everything that follows the label.
 */
declare class FixedTokens extends NextToken {
    fixedTokens: Token[];
    following: Label | null;
    constructor(text: string | Buffer, following?: Label | null);
    preProcess(): PreProcessResult;
    midProcess(): MidProcessResult;
}
/**
 * Indicates that the generation should stop.
 */
declare class StopToken extends NextToken {
    constructor();
    midProcess(): MidProcessResult;
    postProcess(_tokens: Token[]): PostProcessResult;
}
/**
 * Generates a token that satisfies the given constraint.
 * The constraint will be constructed in mid_process() phase, which has slightly longer time limit.
 */
export declare class ConstrainedToken extends NextToken {
    mkConstraint: () => Constraint;
    _constraint: Constraint | null;
    constructor(mkConstraint: () => Constraint);
    midProcess(): MidProcessResult;
    postProcess(tokens: Token[]): PostProcessResult;
}
export declare class PreToken extends NextToken {
    midProcess(): MidProcessResult;
}
/**
 * Forks the execution into `numForks` branches.
 * @param numForks how many branches
 * @returns a number from 0 to `numForks`-1, indicating the branch
 */
export declare function fork(numForks: number): Promise<number>;
/**
 * Suspends execution until all variables are available.
 * @param vars names of variables
 * @returns values of the variables
 */
export declare function waitVars(...vars: string[]): Promise<Buffer[]>;
/**
 *  Low-level interface for AICI. Use aici.start() to wrap a coroutine.
 */
export interface AiciCallbacks {
    init_prompt(prompt: Token[]): void;
    pre_process(): PreProcessResult;
    mid_process(fork_group: SeqId[]): MidProcessResult;
    post_process(backtrack: number, tokens: Token[]): PostProcessResult;
}
/**
 * Awaiting this returns the prompt passed by the user.
 * The code before call to this function has a long time limit (~1000ms).
 * Afterwards, the time limit is ~1ms before awaiting NextToken().
 */
export declare function getPrompt(): Promise<Token[]>;
declare class GetPrompt {
    _resolve?: (value: Token[]) => void;
    run(): Promise<Token[]>;
}
export type CbType = NextToken;
export declare class AiciAsync implements AiciCallbacks {
    static instance: AiciAsync;
    _tokens: Token[];
    _prompt_len: number;
    private _pendingCb;
    private _token;
    private _getPrompt;
    private midProcessReEntry;
    _setGetPrompt(g: GetPrompt): void;
    _nextToken(t: NextToken): void;
    constructor(f: () => Promise<void>);
    step(tokens: Token[]): void;
    init_prompt(prompt: Token[]): void;
    pre_process(): PreProcessResult;
    mid_process(fork_group: SeqId[]): MidProcessResult;
    post_process(backtrack: number, tokens: Token[]): PostProcessResult;
}
/**
 * Starts the AICI loop. The coroutine may first `await aici.getPrompt()` and
 * then can `await aici.gen_*()` or `await aici.FixedTokens()` multiple times.
 * @param f async function
 */
export declare function start(f: () => Promise<void>): AiciAsync;
/**
 * Runs the loop as a test.
 */
export declare function test(f: () => Promise<void>): AiciAsync;
export declare class Label {
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
export declare class ChooseConstraint extends Constraint {
    ptr: number;
    options: Token[][];
    constructor(options: string[]);
    eosAllowed(): boolean;
    eosForced(): boolean;
    tokenAllowed(t: Token): boolean;
    appendToken(t: Token): void;
    allowTokens(ts: TokenSet): void;
}
export declare function genTokens(options: GenOptions): Promise<Token[]>;
export declare function gen(options: GenOptions): Promise<string>;
export declare function checkVar(name: string, value: string): void;
export declare function checkVars(d: Record<string, string>): void;
export declare const helpers: {
    regex_constraint: typeof regexConstraint;
    cfg_constraint: typeof cfgConstraint;
    substr_constraint: typeof substrConstraint;
    FixedTokens: typeof FixedTokens;
    StopToken: typeof StopToken;
    panic: typeof panic;
};
