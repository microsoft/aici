/// <reference path="./native.d.ts" />

import {
  TokenSet,
  tokenize,
  detokenize,
  regexConstraint,
  cfgConstraint,
  substrConstraint,
  Constraint,
  getVar,
  setVar,
  appendVar,
  eosToken,
  panic,
  tokenRepr,
  tokensRepr,
  getConfig,
} from "_aici";

export {
  TokenSet,
  tokenize,
  detokenize,
  getVar,
  setVar,
  appendVar,
  getConfig,
  eosToken,
  tokenRepr,
  tokensRepr,
};

import * as _aici from "_aici";

export type SeqId = number;
type int = number;

let logLevel = 1;

export function setLogLevel(level: number) {
  logLevel = level;
}

function dbgArg(arg: any, depth: number): string {
  const maxElts = 20;
  const maxDepth = 2;
  const maxStr = 200;

  if (arg === null) return "null";
  if (arg === undefined) return "undefined";
  if (typeof arg === "object") {
    if (arg instanceof RegExp) return arg.toString();
    if (arg instanceof Uint8Array) return _aici.bufferRepr(arg);
    if (Array.isArray(arg)) {
      if (depth >= maxDepth && arg.length > 0) return "[...]";
      let suff = "]";
      if (arg.length > maxElts) {
        arg = arg.slice(0, maxElts);
        suff = ", ...]";
      }
      return "[" + arg.map((x: any) => dbgArg(x, depth + 1)).join(", ") + suff;
    } else {
      let keys = Object.keys(arg);
      if (depth >= maxDepth && keys.length > 0) return "{...}";
      let suff = "}";
      if (keys.length > maxElts) {
        suff = ", ...}";
        keys = keys.slice(0, maxElts);
      }
      return (
        "{" +
        keys.map((k) => `${k}: ${dbgArg(arg[k], depth + 1)}`).join(", ") +
        suff
      );
    }
  } else {
    if (depth === 0 || typeof arg !== "string") {
      return arg.toString();
    } else {
      const r = arg.toString();
      if (r.length > maxStr) {
        return r.substring(0, maxStr) + "...";
      } else return r;
    }
  }
}

/**
 * Return debug representation of the argument, suitable for printing in the console.
 */
export function inspect(v: any) {
  return dbgArg(v, 0);
}

export function log(...args: any[]) {
  (console as any)._print(args.map((x) => inspect(x)).join(" "));
}

export class AssertionError extends Error {}

/**
 * Throw an exception if the condition is not met.
 */
export function assert(cond: boolean, msg = "Assertion failed"): asserts cond {
  if (!cond) throw new AssertionError(msg);
}

/**
 * Get list of tokens in the current sequence, including the prompt.
 */
export function getTokens(): Token[] {
  assert(!!AiciAsync.instance);
  return AiciAsync.instance._tokens;
}

/**
 * Get the length of the prompt in the current sequence.
 */
export function getPromptLen(): number {
  assert(!!AiciAsync.instance);
  return AiciAsync.instance._prompt_len;
}

/**
 * Represents a splice operation.
 */
class Splice {
  // the field names below are used by native
  constructor(
    public backtrack: number,
    public ffTokens: Token[],
    public whenSampled: Token[] = []
  ) {}

  /**
   * Adds a splice to the current splice.
   */
  addSplice(other: Splice) {
    assert(this.whenSampled.length === 0);
    if (other.backtrack >= this.ffTokens.length) {
      this.backtrack += other.backtrack - this.ffTokens.length;
      this.ffTokens = other.ffTokens.slice();
    } else {
      if (other.backtrack > 0) this.ffTokens.splice(-other.backtrack);
      this.ffTokens.push(...other.ffTokens);
    }
  }
}

class Branch {
  // field names used by native
  splices: Splice[];
  sampleMask: TokenSet | null;

  constructor({
    splices = [],
    sampleMask = null,
  }: {
    splices?: Splice[];
    sampleMask?: TokenSet | null;
  }) {
    this.splices = splices;
    this.sampleMask = sampleMask;
  }

  /**
   * Checks if the branch is a single splice.
   */
  isSplice(): boolean {
    return (
      this.splices.length === 1 && this.splices[0].whenSampled.length === 0
    );
  }

  static noop(): Branch {
    return new Branch({ splices: [new Splice(0, [])] });
  }
}

export class MidProcessResult {
  skip_me: boolean;
  // field name used by native
  branches: Branch[];

  /**
   * Constructs a MidProcessResult object.
   * @param branches - The list of branches.
   */
  constructor(branches: Branch[]) {
    assert(Array.isArray(branches));
    assert(branches.every((b) => b instanceof Branch));
    this.skip_me = false;
    this.branches = branches;
  }

  /**
   * Checks if the result is a single splice.
   */
  isSplice(): boolean {
    return this.branches.length === 1 && this.branches[0].isSplice();
  }

  static bias(bias: TokenSet): MidProcessResult {
    return new MidProcessResult([new Branch({ sampleMask: bias })]);
  }

  static splice(backtrack: number, ff_tokens: Token[]): MidProcessResult {
    return new MidProcessResult([
      new Branch({
        splices: [new Splice(backtrack, ff_tokens)],
      }),
    ]);
  }

  /**
   * Stops the generation process early.
   */
  static stop(): MidProcessResult {
    return new MidProcessResult([]);
  }

  static noop(): MidProcessResult {
    return new MidProcessResult([Branch.noop()]);
  }

  static skipMe(): MidProcessResult {
    const result = new MidProcessResult([]);
    result.skip_me = true;
    return result;
  }
}

export function allTokens() {
  const ts = new TokenSet();
  ts.setAll(true);
  return ts;
}

export class NextToken {
  finished = false;
  currTokens: Token[] | null = null;
  _resolve?: (value: Token[]) => void;

  constructor() {}

  /**
   * Awaiting this will return generated token (or tokens, if fast-forwarding requested by self.mid_process()).
   */
  run(): Promise<Token[]> {
    assert(!this._resolve);
    AiciAsync.instance._nextToken(this);
    return new Promise((resolve) => {
      this._resolve = resolve;
    });
  }

  /**
   * This can be overridden to return a bias, fast-forward tokens, backtrack etc.
   * ~20ms time limit.
   */
  midProcess(): MidProcessResult {
    return MidProcessResult.bias(allTokens());
  }

  /**
   * This can be overridden to do something with generated tokens.
   * ~1ms time limit.
   * @param tokens tokens generated in the last step
   */
  postProcess(backtrack: number, tokens: Token[]) {}

  /**
   * If true, the postProcess() has to be empty and always self.midProcess().isSplice()
   */
  isFixed() {
    return false;
  }

  //
  // Internal methods
  //

  _mid_process(): MidProcessResult {
    this.reset();
    const spl = this.isFixed();
    const r = this.midProcess();
    if (spl) assert(r.isSplice());
    return r;
  }

  _post_process(backtrack: int, tokens: Token[]) {
    if (logLevel >= 3)
      console.log(`POST-PROCESS: bt:${backtrack} ${tokensRepr(tokens)}`);
    this.currTokens = tokens;
    this.finished = tokens.includes(eosToken());
    this.postProcess(backtrack, tokens);
  }

  private reset(): void {
    this.currTokens = null;
  }
}

/**
 * Forces next tokens to be exactly the given text.
 */
export async function fixed(text: string) {
  await new FixedTokens(text).run();
}

/**
 * Force the exact tokens to be generated; usage: await $`Some text`
 */
export async function $(strings: TemplateStringsArray, ...values: any[]) {
  let result = "";
  strings.forEach((s, i) => {
    result += s;
    if (i < values.length) result += inspect(values[i]);
  });
  await fixed(result);
}

/**
 * Forces next tokens to be exactly the given text.
 * If following is given, the text replaces everything that follows the label.
 */
class FixedTokens extends NextToken {
  fixedTokens: Token[];
  following: Label | null;

  constructor(text: string | Buffer, following: Label | null = null) {
    super();
    this.fixedTokens = tokenize(text);
    if (logLevel >= 1) console.log("FIXED", tokensRepr(this.fixedTokens));
    this.following = following;
  }

  override isFixed(): boolean {
    return true;
  }

  override midProcess(): MidProcessResult {
    let backtrack = 0;
    if (this.following !== null) {
      backtrack = getTokens().length - this.following.ptr;
      assert(backtrack >= 0);
      if (logLevel >= 1) console.log("BACKTRACK", backtrack);
    }
    return MidProcessResult.splice(backtrack, this.fixedTokens);
  }
}

/**
 * Indicates that the generation should stop.
 */
class StopToken extends NextToken {
  constructor() {
    super();
  }

  override midProcess(): MidProcessResult {
    return MidProcessResult.stop();
  }

  override postProcess() {
    this.finished = false; // we're never finished, just keep yelling STOP!
  }
}

/**
 * Generates a token that satisfies the given constraint.
 * The constraint will be constructed in mid_process() phase, which has slightly longer time limit.
 */
export class ConstrainedToken extends NextToken {
  _constraint: Constraint | null = null;

  constructor(public mkConstraint: () => Constraint) {
    super();
  }

  override midProcess(): MidProcessResult {
    const bias = new TokenSet();
    if (this._constraint === null) {
      this._constraint = this.mkConstraint();
    }
    this._constraint.allowTokens(bias);
    if (logLevel >= 2) console.log("ALLOW:", bias.toString());
    if (bias.numSet() === 0) {
      if (logLevel >= 1)
        console.log("Constraint doesn't allow any tokens; adding EOS");
      bias.add(eosToken());
    }
    return MidProcessResult.bias(bias);
  }

  override postProcess(backtrack: number, tokens: Token[]) {
    const c = this._constraint;
    assert(!!c);
    assert(backtrack == 0);
    tokens.forEach((t) => c.appendToken(t));
    if (c.eosForced()) {
      this.finished = true;
    }
  }
}

class Fork extends NextToken {
  constructor(public forks: Branch[]) {
    super();
  }

  override midProcess(): MidProcessResult {
    return new MidProcessResult(this.forks);
  }
}

/**
 * Forks the execution into `numForks` branches.
 * @param numForks how many branches
 * @returns a number from 0 to `numForks`-1, indicating the branch
 */
export async function fork(forks: number | Branch[]): Promise<number> {
  if (typeof forks === "number") {
    forks = Array.from({ length: forks }, () => Branch.noop());
  }
  if (!getConfig("forks") && forks.length > 1) {
    throw new AssertionError("Forking is disabled on this host");
  }
  const f = new Fork(forks);
  await f.run();
  const r = AiciAsync.instance._fork_group.indexOf(_aici.selfSeqId());
  assert(r >= 0);
  return r;
}

class WaitVars extends NextToken {
  vars: string[];
  values: Buffer[];

  constructor(vars: string[]) {
    super();
    this.vars = vars;
    this.values = [];
  }

  override midProcess(): MidProcessResult {
    const values = this.vars.map((v) => getVar(v));
    if (values.includes(null)) return MidProcessResult.noop();
    this.values = values as Buffer[];
    return MidProcessResult.skipMe();
  }
}

/**
 * Suspends execution until all variables are available.
 * @param vars names of variables
 * @returns values of the variables
 */
export async function waitVars(...vars: string[]): Promise<Buffer[]> {
  if (vars.length === 0) return [];
  const w = new WaitVars(vars);
  while (w.values.length == 0) await w.run();
  return w.values;
}

/**
 *  Low-level interface for AICI. Use aici.start() to wrap a coroutine.
 */
export interface AiciCallbacks {
  init_prompt(prompt: Token[]): void;
  mid_process(backtrack: number, tokens: Token[], fork_group: SeqId[]): void;
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
  _fork_group: SeqId[] = [];
  _went_ahead = false;
  private _nextTokenCb?: () => void;
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
    const f = this._nextTokenCb;
    this._nextTokenCb = undefined;
    if (f) f();
  }

  constructor(f: () => Promise<void>) {
    assert(!AiciAsync.instance);
    AiciAsync.instance = this;
    (globalThis as any)._aici_cb = this;

    this.init_prompt = this.init_prompt.bind(this);
    this.mid_process = this.mid_process.bind(this);

    f()
      .then(async () => {
        if (logLevel >= 1) console.log("JsCtrl: done");
        while (true) {
          await new StopToken().run();
        }
      })
      .then(
        () => {},
        (e) => {
          // make sure we catch errors from promises, otherwise they silently stop a thread
          panic(e);
        }
      );

    if (this._getPrompt) {
      assert(this._getPrompt instanceof GetPrompt);
      assert(!this._token);
    } else {
      assert(this._token instanceof NextToken);
    }
  }

  async step(tokens: Token[]) {
    const nextToken = this._token;
    assert(nextToken instanceof NextToken);
    const resolve = nextToken._resolve;
    assert(!!resolve);
    this._token = undefined;
    nextToken._resolve = undefined;
    resolve(tokens);
    await new Promise<void>((resolve) => {
      assert(!this._nextTokenCb);
      this._nextTokenCb = resolve;
    });
  }

  init_prompt(prompt: Token[]): void {
    assert(!this._tokens.length);
    this._prompt_len = prompt.length;
    this._tokens.push(...prompt);

    if (this._getPrompt) {
      this._getPrompt._resolve!(prompt);
      this._getPrompt = undefined;
    } else {
      assert(this._token instanceof NextToken);
    }
    this._went_ahead = true;
  }

  private async applyTokens(backtrack: number, tokens: Token[]) {
    if (backtrack > 0) {
      this._tokens.splice(-backtrack);
    }
    this._tokens.push(...tokens);
    assert(this._token instanceof NextToken);
    this._token._post_process(backtrack, tokens.slice());
    await this.step(tokens);
  }

  private async midProcessWithSkip() {
    while (true) {
      const r = this._token!._mid_process();
      assert(r instanceof MidProcessResult);
      if (!r.skip_me) return r;
      await this.applyTokens(0, []);
    }
  }

  mid_process(backtrack: number, tokens: Token[], fork_group: SeqId[]) {
    this.mid_process_inner(backtrack, tokens, fork_group).then(
      () => {},
      (e) => {
        panic(e);
      }
    );
  }

  private async mid_process_inner(
    backtrack: number,
    tokens: Token[],
    fork_group: SeqId[]
  ) {
    if (logLevel >= 2)
      console.log("MID-PROCESS", backtrack, tokensRepr(tokens), fork_group);

    this._fork_group = fork_group;
    assert(this._token instanceof NextToken, "mid_process - no token");

    if (this._went_ahead) this._went_ahead = false;
    else await this.applyTokens(backtrack, tokens);

    const r = await this.midProcessWithSkip();
    let r0 = r;

    while (r0.isSplice()) {
      const s = r0.branches[0].splices[0];
      await this.applyTokens(s.backtrack, s.ffTokens);
      this._went_ahead = true;
      if (!this._token.isFixed()) break;
      r0 = await this.midProcessWithSkip();
      assert(r0.isSplice());
      r.branches[0].splices[0].addSplice(r0.branches[0].splices[0]);
    }

    if (logLevel >= 2) console.log("MID-PROCESS-RETURN", r);
    _aici._midProcessReturn(r);
  }
}

/**
 * Starts the AICI loop.
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
   * Create a new label the indicates the current position in the sequence.
   * Can be passed as `following=` argument to `FixedTokens()`.
   */
  constructor() {
    this.ptr = getTokens().length;
  }

  /**
   * Return tokens generated since the label.
   */
  tokensSince(): Token[] {
    return getTokens().slice(this.ptr);
  }

  /**
   * Return text generated since the label.
   */
  textSince(): string {
    return detokenize(this.tokensSince()).decode();
  }

  /**
   * Generate given prompt text, replacing all text after the current label.
   */
  async fixedAfter(text: string) {
    await new FixedTokens(text, this).run();
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

  eosAllowed(): boolean {
    return this.options.some((o) => o.length === this.ptr);
  }

  eosForced(): boolean {
    return this.options.length === 1 && this.options[0].length === this.ptr;
  }

  tokenAllowed(t: Token): boolean {
    return this.options.some((o) => this.ptr < o.length && o[this.ptr] === t);
  }

  appendToken(t: Token): void {
    this.options = this.options.filter(
      (o) => this.ptr < o.length && o[this.ptr] === t
    );
    this.ptr += 1;
  }

  allowTokens(ts: TokenSet): void {
    for (const o of this.options) {
      if (this.ptr < o.length) {
        ts.add(o[this.ptr]);
      } else if (this.ptr === o.length) {
        ts.add(eosToken());
      }
    }
  }
}

export async function genTokens(options: GenOptions): Promise<Token[]> {
  if (logLevel >= 2) console.log("GEN-OPT", options);
  const res: Token[] = [];
  const {
    regex,
    yacc,
    substring,
    substringEnd = '"',
    options: optionList,
    storeVar,
    stopAt,
    maxTokens = 20,
  } = options;

  let constraint: Constraint;
  assert(
    [regex, substring, yacc, optionList].filter((x) => x !== undefined)
      .length <= 1
  );
  if (regex !== undefined) {
    const rx = typeof regex === "string" ? regex : regex.source;
    constraint = regexConstraint(rx);
  } else if (substring !== undefined) {
    constraint = substrConstraint(substring, substringEnd);
  } else if (yacc !== undefined) {
    constraint = cfgConstraint(yacc);
  } else if (optionList !== undefined) {
    constraint = new ChooseConstraint(optionList);
  } else {
    constraint = new Constraint();
  }

  const next_token = new ConstrainedToken(() => constraint!);

  for (let i = 0; i < maxTokens; i++) {
    const tokens = await next_token.run();

    if (tokens?.length) {
      res.push(...tokens);

      if (logLevel >= 2) console.log("GEN-STEP:", tokensRepr(tokens));

      const text = detokenize(res).decode();

      if (stopAt !== undefined && text.includes(stopAt)) {
        break;
      }
    }

    if (next_token.finished) {
      break;
    }
  }

  if (storeVar !== undefined) setVar(storeVar, detokenize(res));

  if (logLevel >= 1) console.log("GEN", tokensRepr(res));

  return res;
}

export async function gen(options: GenOptions): Promise<string> {
  const tokens = await genTokens(options);
  return detokenize(tokens).decode();
}

export function checkVar(name: string, value: string): void {
  const v = getVar(name);
  if (v == null) {
    throw new AssertionError(`Variable ${name} is unset`);
  }
  const vStr = v.decode();
  if (vStr !== value) {
    throw new AssertionError(
      `Variable ${name}: ${JSON.stringify(vStr)} != ${JSON.stringify(value)}`
    );
  }
}

export function checkVars(d: Record<string, string>): void {
  for (const [k, v] of Object.entries(d)) {
    checkVar(k, v);
  }
}

// stuff we don't want to export top-level
export const helpers = {
  regex_constraint: regexConstraint,
  cfg_constraint: cfgConstraint,
  substr_constraint: substrConstraint,
  FixedTokens,
  StopToken,
  panic,
};

String.prototype.toBuffer = function (this: string) {
  return _aici.stringToBuffer(this);
};

String.fromBuffer = _aici.bufferToString;
Uint8Array.prototype.toString = function (this: Uint8Array) {
  return _aici.bufferRepr(this);
};

Uint8Array.prototype.decode = function (this: Uint8Array) {
  return _aici.bufferToString(this);
};

console.log = log;
console.info = log;
console.warn = log;
console.debug = log;
console.trace = log;

globalThis.$ = $;
globalThis.fixed = fixed;
globalThis.assert = assert;
globalThis.gen = gen;
globalThis.genTokens = genTokens;
globalThis.start = start;
globalThis.waitVars = waitVars;
globalThis.fork = fork;
