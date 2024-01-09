/// <reference path="./native.d.ts" />

import {
  TokenSet,
  tokenize,
  detokenize,
  RegexConstraint,
  CfgConstraint,
  SubStrConstraint,
  Constraint,
  get_var,
  set_var,
  append_var,
  eos_token,
} from "_aici";

export {
  TokenSet,
  tokenize,
  detokenize,
  RegexConstraint,
  CfgConstraint,
  SubStrConstraint,
  Constraint,
  get_var,
  set_var,
  append_var,
  eos_token,
};

import * as _aici from "_aici";

type Token = number;
type SeqId = number;
type int = number;
type Buffer = Uint8Array;

export class AssertionError extends Error {}

function assert(cond: boolean, msg = "Assertion failed"): asserts cond {
  if (!cond) throw new AssertionError("Assertion failed");
}

/**
 * Get list of tokens in the current sequence, including the prompt.
 */
export function get_tokens(): Token[] {
  assert(!!AiciAsync.instance);
  return AiciAsync.instance._tokens;
}

/**
 * Get the length of the prompt in the current sequence.
 */
export function get_prompt_len(): number {
  assert(!!AiciAsync.instance);
  return AiciAsync.instance._prompt_len;
}

export class MidProcessResult {
  _stop = false;
  _skipMe = false;
  _logitBias: TokenSet | null = null;
  _backtrack: number = 0;
  _ffTokens: Token[] = [];

  constructor() {}

  static stop(): MidProcessResult {
    const res = new MidProcessResult();
    res._stop = true;
    return res;
  }

  static skipMe(): MidProcessResult {
    const res = new MidProcessResult();
    res._skipMe = true;
    return res;
  }

  static bias(bias: TokenSet): MidProcessResult {
    const res = new MidProcessResult();
    res._logitBias = bias;
    return res;
  }

  static splice(backtrack: number, tokens: Token[]): MidProcessResult {
    const res = new MidProcessResult();
    assert(backtrack >= 0);
    assert(Array.isArray(tokens));
    res._backtrack = backtrack;
    res._ffTokens = tokens;
    return res;
  }
}

export class PreProcessResult {
  suspended = false;
  ff_tokens: Token[] = [];
  attention_masks: number[][] = [[]];

  constructor() {}

  static continue_(): PreProcessResult {
    return new PreProcessResult();
  }

  static suspend(): PreProcessResult {
    const res = new PreProcessResult();
    res.suspended = true;
    return res;
  }

  static fork(num_forks: number): PreProcessResult {
    const res = new PreProcessResult();
    res.attention_masks = Array.from({ length: num_forks }, () => []);
    return res;
  }

  static ff_tokens_pre(toks: Token[]): PreProcessResult {
    const res = new PreProcessResult();
    res.ff_tokens = toks;
    return res;
  }
}

export class PostProcessResult {
  stop_seq: boolean;

  constructor(stop_seq = false) {
    this.stop_seq = stop_seq;
  }

  static continue_(): PostProcessResult {
    return new PostProcessResult();
  }

  static stop(): PostProcessResult {
    return new PostProcessResult(true);
  }

  static from_tokens(tokens: Token[]): PostProcessResult {
    return new PostProcessResult(tokens.includes(eos_token()));
  }
}

export class NextToken {
  finished = false;
  curr_tokens: Token[] | null = null;
  fork_group: SeqId[] = [];
  _resolve?: (value: Token[]) => void;

  constructor() {}

  /**
   * Awaiting this will return generated token (or tokens, if fast-forwarding requested by self.mid_process()).
   * You have only ~1ms to process the results before awaiting a new instance of NextToken() again.
   */
  run(): Promise<Token[]> {
    assert(!this._resolve);
    AiciAsync.instance._nextToken(this);
    return new Promise((resolve) => {
      this._resolve = resolve;
    });
  }

  /**
   * Override to suspend, if the model cannot continue generating tokens
   * now (for example, not all variables are available to compute bias).
   * ~1ms time limit.
   */
  pre_process(): PreProcessResult {
    return PreProcessResult.continue_();
  }

  /**
   * This can be overridden to return a bias, fast-forward tokens, backtrack etc.
   * ~20ms time limit.
   */
  mid_process(): MidProcessResult {
    return MidProcessResult.bias(new TokenSet());
  }

  /**
   * This can be overridden to do something with generated tokens.
   * ~1ms time limit.
   * @param tokens tokens generated in the last step
   */
  post_process(tokens: Token[]): PostProcessResult {
    return PostProcessResult.continue_();
  }

  //
  // Internal methods
  //

  _pre_process(): PreProcessResult {
    this.reset();
    return this.pre_process();
  }

  _mid_process(fork_group: SeqId[]): MidProcessResult {
    this.fork_group = fork_group;
    return this.mid_process();
  }

  _post_process(_backtrack: int, tokens: Token[]): PostProcessResult {
    this.curr_tokens = tokens;
    this.finished = tokens.includes(eos_token());
    return this.post_process(tokens);
  }

  private reset(): void {
    this.curr_tokens = null;
    this.fork_group = [];
  }
}

/**
 * Forces next tokens to be exactly the given text.
 * If following is given, the text replaces everything that follows the label.
 */
export class FixedTokens extends NextToken {
  fixed_tokens: Token[];
  following: Label | null;

  constructor(text: string | Buffer, following: Label | null = null) {
    super();
    this.fixed_tokens = tokenize(text);
    this.following = following;
  }

  pre_process(): PreProcessResult {
    if (this.following === null) {
      return PreProcessResult.ff_tokens_pre(this.fixed_tokens);
    }
    return PreProcessResult.continue_();
  }

  mid_process(): MidProcessResult {
    let backtrack = 0;
    if (this.following !== null) {
      backtrack = get_tokens().length - this.following.ptr;
      assert(backtrack >= 0);
      console.log("backtrack", backtrack);
    }
    return MidProcessResult.splice(backtrack, this.fixed_tokens);
  }
}

/**
 * Indicates that the generation should stop.
 */
export class StopToken extends NextToken {
  constructor() {
    super();
  }

  mid_process(): MidProcessResult {
    return MidProcessResult.stop();
  }

  post_process(_tokens: Token[]): PostProcessResult {
    this.finished = false; // we're never finished, just keep yelling STOP!
    return PostProcessResult.stop();
  }
}

/**
 * Generates a token that satisfies the given constraint.
 * The constraint will be constructed in mid_process() phase, which has slightly longer time limit.
 */
export class ConstrainedToken extends NextToken {
  mk_constraint: () => Constraint;
  _constraint: Constraint | null;

  constructor(mk_constraint: () => Constraint) {
    super();
    this.mk_constraint = mk_constraint;
    this._constraint = null;
  }

  mid_process(): MidProcessResult {
    const bias = new TokenSet();
    if (this._constraint === null) {
      this._constraint = this.mk_constraint();
    }
    this._constraint.allow_tokens(bias);
    return MidProcessResult.bias(bias);
  }

  post_process(tokens: Token[]): PostProcessResult {
    const c = this._constraint;
    assert(!!c);
    tokens.forEach((t) => c.append_token(t));
    if (c.eos_forced()) {
      this.finished = true;
    }
    return PostProcessResult.continue_();
  }
}

export class PreToken extends NextToken {
  mid_process(): MidProcessResult {
    return MidProcessResult.skipMe();
  }
}

class _Fork extends PreToken {
  num_forks: number;

  constructor(num_forks: number) {
    super();
    this.num_forks = num_forks;
  }

  pre_process(): PreProcessResult {
    return PreProcessResult.fork(this.num_forks);
  }
}

/**
 * Forks the execution into `num_forks` branches.
 * @param num_forks how many branches
 * @returns a number from 0 to `num_forks`-1, indicating the branch
 */
export async function fork(num_forks: number): Promise<number> {
  const f = new _Fork(num_forks);
  await f.run();
  return f.fork_group.indexOf(_aici.self_seq_id());
}

class _WaitVars extends PreToken {
  vars: string[];
  values: Buffer[];

  constructor(vars: string[]) {
    super();
    this.vars = vars;
    this.values = [];
  }

  pre_process(): PreProcessResult {
    const values = this.vars.map((v) => get_var(v));
    if (values.includes(null)) {
      return PreProcessResult.suspend();
    }
    this.values = values as Buffer[];
    return PreProcessResult.continue_();
  }
}

/**
 * Suspends execution until all variables are available.
 * @param vars names of variables
 * @returns values of the variables
 */
export async function waitVars(...vars: string[]): Promise<Buffer[]> {
  const w = new _WaitVars(vars);
  await w.run();
  return w.values;
}

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
export function getPrompt(): Promise<Token[]> {
  return new GetPrompt().run();
}

class GetPrompt {
  _resolve?: (value: Token[]) => void;
  run(): Promise<Token[]> {
    assert(!this._resolve);
    return new Promise((resolve) => {
      AiciAsync.instance._setGetPrompt(this);
      this._resolve = resolve;
    });
  }
}

export type CbType = NextToken;
export class AiciAsync implements AiciCallbacks {
  static instance: AiciAsync;

  _tokens: Token[] = [];
  _prompt_len = 0;
  private _pending_cb: CbType | undefined;
  private _fork_group: SeqId[] = [];
  private _token: CbType | undefined;
  private _getPrompt: GetPrompt | undefined;

  _setGetPrompt(g: GetPrompt) {
    assert(!this._getPrompt);
    assert(!this._token);
    assert(g instanceof GetPrompt);
    this._getPrompt = g;
  }

  _nextToken(t: NextToken) {
    assert(!this._token);
    assert(!this._getPrompt);
    assert(t instanceof NextToken);
    this._token = t;
  }

  constructor(f: () => Promise<void>) {
    assert(AiciAsync.instance === null);
    AiciAsync.instance = this;
    (globalThis as any)._aici_cb = this;

    f();

    if (this._getPrompt) {
      assert(this._getPrompt instanceof GetPrompt);
      assert(!this._token);
    } else {
      assert(this._token instanceof NextToken);
    }
  }

  step(tokens: Token[]): void {
    if (this._pending_cb !== null) {
      // TODO
      this._token = this._pending_cb;
      this._pending_cb = undefined;
      return;
    }

    const nextToken = this._token;
    assert(nextToken instanceof NextToken);
    const resolve = nextToken._resolve;
    assert(!!resolve);
    this._token = undefined;
    nextToken._resolve = undefined;
    resolve(tokens);
    assert((this._token as any) instanceof NextToken);
  }

  init_prompt(prompt: Token[]): void {
    assert(!this._tokens.length);
    this._prompt_len = prompt.length;
    this._tokens.push(...prompt);

    if (this._getPrompt) {
      this._getPrompt._resolve!(prompt);
      this._getPrompt = undefined;
    }

    assert(this._token instanceof NextToken);
  }

  pre_process(): PreProcessResult {
    assert(this._token instanceof NextToken);
    if (this._token.finished) {
      this._token = new StopToken();
    }
    const r = this._token._pre_process();
    assert(r instanceof PreProcessResult);
    return r;
  }

  mid_process(fork_group: SeqId[]): MidProcessResult {
    assert(this._token instanceof NextToken);

    let r = this._token._mid_process(fork_group);
    assert(r instanceof MidProcessResult);

    while (r._skipMe) {
      this.step([]);
      assert(this._token instanceof NextToken);
      const r2 = this._token._pre_process();
      assert(r2 instanceof PreProcessResult);
      assert(r2.attention_masks.length === 1, "nested fork not allowed");
      if (r2.suspended) {
        // Need to generate one fake token...
        this._pending_cb = this._token;
        const f = new FixedTokens("░");
        assert(f.fixed_tokens.length === 1);
        this._token = f;
      }
      r = this._token._mid_process(fork_group);
      assert(r instanceof MidProcessResult);
    }

    assert(Array.isArray(r._ffTokens));
    return r;
  }

  post_process(backtrack: number, tokens: Token[]): PostProcessResult {
    if (backtrack > 0) {
      this._tokens.splice(-backtrack);
    }
    this._tokens.push(...tokens);

    assert(this._token instanceof NextToken);
    const r = this._token._post_process(backtrack, tokens.slice());
    assert(r instanceof PostProcessResult);
    this.step(tokens);
    return r;
  }
}

/**
 * Starts the AICI loop. The coroutine may first `await aici.getPrompt()` and
 * then can `await aici.gen_*()` or `await aici.FixedTokens()` multiple times.
 * @param f async function
 */
export function start(f: () => Promise<void>): AiciAsync {
  return new AiciAsync(f);
}

/**
 * Runs the loop as a test.
 */
export function test(f: () => Promise<void>): AiciAsync {
  return new AiciAsync(() =>
    f().then(() => {
      console.log("TEST OK");
    })
  );
}

export class Label {
  ptr: number;

  /**
   * Create a new label the indictes the current position in the sequence.
   * Can be passed as `following=` argument to `FixedTokens()`.
   */
  constructor() {
    this.ptr = get_tokens().length;
  }

  /**
   * Return tokens generated since the label.
   */
  tokens_since(): Token[] {
    return get_tokens().slice(this.ptr);
  }

  /**
   * Return text generated since the label.
   */
  text_since(): string {
    return detokenize(this.tokens_since()).toString();
  }
}

export class ChooseConstraint extends Constraint {
  ptr: number;
  options: Token[][];

  constructor(options: string[]) {
    super();
    this.ptr = 0;
    this.options = options.map((o) => tokenize(o));
  }

  eos_allowed(): boolean {
    return this.options.some((o) => o.length === this.ptr);
  }

  eos_forced(): boolean {
    return this.options.length === 1 && this.options[0].length === this.ptr;
  }

  token_allowed(t: Token): boolean {
    return this.options.some((o) => this.ptr < o.length && o[this.ptr] === t);
  }

  append_token(t: Token): void {
    this.options = this.options.filter(
      (o) => this.ptr < o.length && o[this.ptr] === t
    );
    this.ptr += 1;
  }

  allow_tokens(ts: TokenSet): void {
    for (const o of this.options) {
      if (this.ptr < o.length) {
        ts.add(o[this.ptr]);
      } else if (this.ptr === o.length) {
        ts.add(eos_token());
      }
    }
  }
}

export type GenOptions = {
  regex?: string;
  yacc?: string;
  substring?: string;
  substring_end?: string;
  options?: string[];
  store_var?: string;
  stop_at?: string;
  max_tokens?: number;
};

export async function gen_tokens(options: GenOptions): Promise<Token[]> {
  const res: Token[] = [];
  const {
    regex,
    yacc,
    substring,
    substring_end = '"',
    options: optionList,
    store_var,
    stop_at,
    max_tokens = 20,
  } = options;

  let constraint: Constraint;
  assert(
    [regex, substring, yacc, optionList].filter((x) => x !== undefined)
      .length <= 1
  );
  if (regex !== undefined) {
    constraint = new RegexConstraint(regex);
  } else if (substring !== undefined) {
    constraint = new SubStrConstraint(substring, substring_end);
  } else if (yacc !== undefined) {
    constraint = new CfgConstraint(yacc);
  } else if (optionList !== undefined) {
    constraint = new ChooseConstraint(optionList);
  } else {
    constraint = new Constraint();
  }

  const next_token = new ConstrainedToken(() => constraint!);

  for (let i = 0; i < max_tokens; i++) {
    const tokens = await next_token.run();
    res.push(...tokens);

    const text = detokenize(res).toString();

    if (stop_at !== undefined && text.includes(stop_at)) {
      break;
    }

    if (next_token.finished) {
      break;
    }
  }

  if (store_var !== undefined) {
    set_var(store_var, detokenize(res));
  }

  console.log("GEN", res, detokenize(res).toString());
  return res;
}

export async function gen_text(options: GenOptions): Promise<string> {
  const tokens = await gen_tokens(options);
  return detokenize(tokens).toString();
}

export function check_var(name: string, value: string): void {
  const v = get_var(name);
  if (v === null) {
    throw new AssertionError(`Variable ${name} is unset`);
  }
  const vStr = v.toString();
  if (vStr !== value) {
    throw new AssertionError(`Variable ${name}: ${vStr} != ${value}`);
  }
}

export function check_vars(d: Record<string, string>): void {
  for (const [k, v] of Object.entries(d)) {
    check_var(k, v);
  }
}
