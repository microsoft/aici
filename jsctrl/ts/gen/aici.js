/// <reference path="./native.d.ts" />
import { TokenSet, tokenize, detokenize, regexConstraint, cfgConstraint, substrConstraint, Constraint, getVar, setVar, appendVar, eosToken, panic, } from "_aici";
export { TokenSet, tokenize, detokenize, getVar, setVar, appendVar, eosToken };
import * as _aici from "_aici";
function dbgArg(arg, depth) {
    const maxElts = 20;
    const maxDepth = 2;
    const maxStr = 200;
    if (arg === null)
        return "null";
    if (arg === undefined)
        return "undefined";
    if (typeof arg === "object") {
        if (arg instanceof RegExp)
            return arg.toString();
        if (arg instanceof Uint8Array)
            return _aici.bufferRepr(arg);
        if (Array.isArray(arg)) {
            if (depth >= maxDepth && arg.length > 0)
                return "[...]";
            let suff = "]";
            if (arg.length > maxElts) {
                arg = arg.slice(0, maxElts);
                suff = ", ...]";
            }
            return "[" + arg.map((x) => dbgArg(x, depth + 1)).join(", ") + suff;
        }
        else {
            let keys = Object.keys(arg);
            if (depth >= maxDepth && keys.length > 0)
                return "{...}";
            let suff = "}";
            if (keys.length > maxElts) {
                suff = ", ...}";
                keys = keys.slice(0, maxElts);
            }
            return ("{" +
                keys.map((k) => `${k}: ${dbgArg(arg[k], depth + 1)}`).join(", ") +
                suff);
        }
    }
    else {
        if (depth === 0 || typeof arg !== "string") {
            return arg.toString();
        }
        else {
            const r = arg.toString();
            if (r.length > maxStr) {
                return r.substring(0, maxStr) + "...";
            }
            else
                return r;
        }
    }
}
export function inspect(v) {
    return dbgArg(v, 0);
}
export function log(...args) {
    console._print(args.map((x) => inspect(x)).join(" "));
}
export class AssertionError extends Error {
}
/**
 * Throw an exception if the condition is not met.
 */
export function assert(cond, msg = "Assertion failed") {
    if (!cond)
        throw new AssertionError(msg);
}
/**
 * Get list of tokens in the current sequence, including the prompt.
 */
export function getTokens() {
    assert(!!AiciAsync.instance);
    return AiciAsync.instance._tokens;
}
/**
 * Get the length of the prompt in the current sequence.
 */
export function getPromptLen() {
    assert(!!AiciAsync.instance);
    return AiciAsync.instance._prompt_len;
}
export class MidProcessResult {
    constructor() {
        this._n_skip_me = false;
        this._n_stop = false;
        this._n_logit_bias = null;
        this._n_backtrack = 0;
        this._n_ff_tokens = [];
    }
    static stop() {
        const res = new MidProcessResult();
        res._n_stop = true;
        return res;
    }
    static skipMe() {
        const res = new MidProcessResult();
        res._n_skip_me = true;
        return res;
    }
    static bias(bias) {
        const res = new MidProcessResult();
        res._n_logit_bias = bias;
        return res;
    }
    static splice(backtrack, tokens) {
        const res = new MidProcessResult();
        assert(backtrack >= 0);
        assert(Array.isArray(tokens));
        res._n_backtrack = backtrack;
        res._n_ff_tokens = tokens;
        return res;
    }
}
export class PreProcessResult {
    constructor() {
        this._n_suspended = false;
        this._n_ff_tokens = [];
        this._n_attention_masks = [[]];
    }
    static continue_() {
        return new PreProcessResult();
    }
    static suspend() {
        const res = new PreProcessResult();
        res._n_suspended = true;
        return res;
    }
    static fork(numForks) {
        const res = new PreProcessResult();
        res._n_attention_masks = Array.from({ length: numForks }, () => []);
        return res;
    }
    static ffTokens(toks) {
        const res = new PreProcessResult();
        res._n_ff_tokens = toks;
        return res;
    }
}
export class PostProcessResult {
    constructor(stop_seq = false) {
        this._n_stop_seq = stop_seq;
    }
    static continue_() {
        return new PostProcessResult();
    }
    static stop() {
        return new PostProcessResult(true);
    }
    static fromTokens(tokens) {
        return new PostProcessResult(tokens.includes(eosToken()));
    }
}
export class NextToken {
    constructor() {
        this.finished = false;
        this.currTokens = null;
        this.forkGroup = [];
    }
    /**
     * Awaiting this will return generated token (or tokens, if fast-forwarding requested by self.mid_process()).
     * You have only ~1ms to process the results before awaiting a new instance of NextToken() again.
     */
    run() {
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
    preProcess() {
        return PreProcessResult.continue_();
    }
    /**
     * This can be overridden to return a bias, fast-forward tokens, backtrack etc.
     * ~20ms time limit.
     */
    midProcess() {
        return MidProcessResult.bias(new TokenSet());
    }
    /**
     * This can be overridden to do something with generated tokens.
     * ~1ms time limit.
     * @param tokens tokens generated in the last step
     */
    postProcess(tokens) {
        return PostProcessResult.continue_();
    }
    //
    // Internal methods
    //
    _pre_process() {
        this.reset();
        return this.preProcess();
    }
    _mid_process(fork_group) {
        this.forkGroup = fork_group;
        return this.midProcess();
    }
    _post_process(_backtrack, tokens) {
        this.currTokens = tokens;
        this.finished = tokens.includes(eosToken());
        return this.postProcess(tokens);
    }
    reset() {
        this.currTokens = null;
        this.forkGroup = [];
    }
}
/**
 * Forces next tokens to be exactly the given text.
 */
export async function fixed(text) {
    await new FixedTokens(text).run();
}
/**
 * Force the exact tokens to be generated; usage: await $`Some text`
 */
export async function $(strings, ...values) {
    let result = "";
    strings.forEach((s, i) => {
        result += s;
        if (i < values.length)
            result += inspect(values[i]);
    });
    await fixed(result);
}
/**
 * Forces next tokens to be exactly the given text.
 * If following is given, the text replaces everything that follows the label.
 */
class FixedTokens extends NextToken {
    constructor(text, following = null) {
        super();
        this.fixedTokens = tokenize(text);
        this.following = following;
    }
    preProcess() {
        if (this.following === null) {
            return PreProcessResult.ffTokens(this.fixedTokens);
        }
        return PreProcessResult.continue_();
    }
    midProcess() {
        let backtrack = 0;
        if (this.following !== null) {
            backtrack = getTokens().length - this.following.ptr;
            assert(backtrack >= 0);
            console.log("backtrack", backtrack);
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
    midProcess() {
        return MidProcessResult.stop();
    }
    postProcess(_tokens) {
        this.finished = false; // we're never finished, just keep yelling STOP!
        return PostProcessResult.stop();
    }
}
/**
 * Generates a token that satisfies the given constraint.
 * The constraint will be constructed in mid_process() phase, which has slightly longer time limit.
 */
export class ConstrainedToken extends NextToken {
    constructor(mkConstraint) {
        super();
        this.mkConstraint = mkConstraint;
        this._constraint = null;
    }
    midProcess() {
        const bias = new TokenSet();
        if (this._constraint === null) {
            this._constraint = this.mkConstraint();
        }
        this._constraint.allowTokens(bias);
        return MidProcessResult.bias(bias);
    }
    postProcess(tokens) {
        const c = this._constraint;
        assert(!!c);
        tokens.forEach((t) => c.appendToken(t));
        if (c.eosForced()) {
            this.finished = true;
        }
        return PostProcessResult.continue_();
    }
}
export class PreToken extends NextToken {
    midProcess() {
        return MidProcessResult.skipMe();
    }
}
class Fork extends PreToken {
    constructor(numForks) {
        super();
        this.numForks = numForks;
    }
    preProcess() {
        return PreProcessResult.fork(this.numForks);
    }
}
/**
 * Forks the execution into `numForks` branches.
 * @param numForks how many branches
 * @returns a number from 0 to `numForks`-1, indicating the branch
 */
export async function fork(numForks) {
    const f = new Fork(numForks);
    await f.run();
    return f.forkGroup.indexOf(_aici.selfSeqId());
}
class WaitVars extends PreToken {
    constructor(vars) {
        super();
        this.vars = vars;
        this.values = [];
    }
    preProcess() {
        const values = this.vars.map((v) => getVar(v));
        if (values.includes(null)) {
            return PreProcessResult.suspend();
        }
        this.values = values;
        return PreProcessResult.continue_();
    }
}
/**
 * Suspends execution until all variables are available.
 * @param vars names of variables
 * @returns values of the variables
 */
export async function waitVars(...vars) {
    const w = new WaitVars(vars);
    await w.run();
    return w.values;
}
/**
 * Awaiting this returns the prompt passed by the user.
 * The code before call to this function has a long time limit (~1000ms).
 * Afterwards, the time limit is ~1ms before awaiting NextToken().
 */
export function getPrompt() {
    return new GetPrompt().run();
}
class GetPrompt {
    run() {
        assert(!this._resolve);
        return new Promise((resolve) => {
            AiciAsync.instance._setGetPrompt(this);
            this._resolve = resolve;
        });
    }
}
export class AiciAsync {
    _setGetPrompt(g) {
        assert(!this._getPrompt);
        assert(!this._token);
        assert(g instanceof GetPrompt);
        this._getPrompt = g;
    }
    _nextToken(t) {
        assert(!this._token);
        assert(!this._getPrompt);
        assert(t instanceof NextToken);
        this._token = t;
    }
    constructor(f) {
        this._tokens = [];
        this._prompt_len = 0;
        this.midProcessReEntry = false;
        assert(!AiciAsync.instance);
        AiciAsync.instance = this;
        globalThis._aici_cb = this;
        this.init_prompt = this.init_prompt.bind(this);
        this.pre_process = this.pre_process.bind(this);
        this.mid_process = this.mid_process.bind(this);
        this.post_process = this.post_process.bind(this);
        f()
            .then(async () => {
            console.log("JsCtrl: done");
            while (true) {
                await new StopToken().run();
            }
        })
            .then(() => { }, (e) => {
            // make sure we catch errors from promises, otherwise they silently stop a thread
            panic(e);
        });
        if (this._getPrompt) {
            assert(this._getPrompt instanceof GetPrompt);
            assert(!this._token);
        }
        else {
            assert(this._token instanceof NextToken);
        }
    }
    step(tokens) {
        if (this._pendingCb != null) {
            // TODO
            this._token = this._pendingCb;
            this._pendingCb = undefined;
            return;
        }
        const nextToken = this._token;
        assert(nextToken instanceof NextToken);
        const resolve = nextToken._resolve;
        assert(!!resolve);
        // console.log("reset");
        this._token = undefined;
        nextToken._resolve = undefined;
        resolve(tokens);
        // console.log("t2", this._token, resolve);
        // this happens only in the deferred jobs...
        // assert((this._token as any) instanceof NextToken);
    }
    init_prompt(prompt) {
        assert(!this._tokens.length);
        this._prompt_len = prompt.length;
        this._tokens.push(...prompt);
        if (this._getPrompt) {
            this._getPrompt._resolve(prompt);
            this._getPrompt = undefined;
        }
        else {
            assert(this._token instanceof NextToken);
        }
    }
    pre_process() {
        // console.log("tok", this._token);
        assert(this._token instanceof NextToken, "pre_process - jobs finished");
        if (this._token.finished) {
            this._token = new StopToken();
        }
        const r = this._token._pre_process();
        assert(r instanceof PreProcessResult);
        return r;
    }
    mid_process(fork_group) {
        assert(this._token instanceof NextToken, "mid_process - no token");
        if (this.midProcessReEntry) {
            const r2 = this._token._pre_process();
            assert(r2 instanceof PreProcessResult);
            assert(r2._n_attention_masks.length === 1, "nested fork not allowed");
            if (r2._n_suspended) {
                // Need to generate one fake token...
                this._pendingCb = this._token;
                const f = new FixedTokens("░");
                assert(f.fixedTokens.length === 1);
                this._token = f;
            }
            this.midProcessReEntry = false;
        }
        const r = this._token._mid_process(fork_group);
        assert(r instanceof MidProcessResult);
        if (r._n_skip_me) {
            this.step([]);
            this.midProcessReEntry = true;
            return r;
        }
        else {
            assert(Array.isArray(r._n_ff_tokens));
            return r;
        }
    }
    post_process(backtrack, tokens) {
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
export function start(f) {
    return new AiciAsync(f);
}
/**
 * Runs the loop as a test.
 */
export function test(f) {
    return new AiciAsync(() => f().then(() => {
        console.log("TEST OK");
    }));
}
export class Label {
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
    tokensSince() {
        return getTokens().slice(this.ptr);
    }
    /**
     * Return text generated since the label.
     */
    textSince() {
        return detokenize(this.tokensSince()).decode();
    }
    /**
     * Generate given prompt text, replacing all text after the current label.
     */
    async fixedAfter(text) {
        await new FixedTokens(text, this).run();
    }
}
export class ChooseConstraint extends Constraint {
    constructor(options) {
        super();
        this.ptr = 0;
        this.options = options.map((o) => tokenize(o));
    }
    eosAllowed() {
        return this.options.some((o) => o.length === this.ptr);
    }
    eosForced() {
        return this.options.length === 1 && this.options[0].length === this.ptr;
    }
    tokenAllowed(t) {
        return this.options.some((o) => this.ptr < o.length && o[this.ptr] === t);
    }
    appendToken(t) {
        this.options = this.options.filter((o) => this.ptr < o.length && o[this.ptr] === t);
        this.ptr += 1;
    }
    allowTokens(ts) {
        for (const o of this.options) {
            if (this.ptr < o.length) {
                ts.add(o[this.ptr]);
            }
            else if (this.ptr === o.length) {
                ts.add(eosToken());
            }
        }
    }
}
export async function genTokens(options) {
    console.log("GEN", options);
    const res = [];
    const { regex, yacc, substring, substringEnd = '"', options: optionList, storeVar, stopAt, maxTokens = 20, } = options;
    let constraint;
    assert([regex, substring, yacc, optionList].filter((x) => x !== undefined)
        .length <= 1);
    if (regex !== undefined) {
        const rx = typeof regex === "string" ? regex : regex.source;
        constraint = regexConstraint(rx);
    }
    else if (substring !== undefined) {
        constraint = substrConstraint(substring, substringEnd);
    }
    else if (yacc !== undefined) {
        constraint = cfgConstraint(yacc);
    }
    else if (optionList !== undefined) {
        constraint = new ChooseConstraint(optionList);
    }
    else {
        constraint = new Constraint();
    }
    const next_token = new ConstrainedToken(() => constraint);
    for (let i = 0; i < maxTokens; i++) {
        const tokens = await next_token.run();
        res.push(...tokens);
        const text = detokenize(res).decode();
        if (stopAt !== undefined && text.includes(stopAt)) {
            break;
        }
        // console.log(`GEN-${i}`, next_token);
        if (next_token.finished) {
            break;
        }
    }
    if (storeVar !== undefined) {
        setVar(storeVar, detokenize(res));
    }
    console.log("GEN", res, detokenize(res));
    return res;
}
export async function gen(options) {
    const tokens = await genTokens(options);
    return detokenize(tokens).decode();
}
export function checkVar(name, value) {
    const v = getVar(name);
    if (v == null) {
        throw new AssertionError(`Variable ${name} is unset`);
    }
    const vStr = v.decode();
    if (vStr !== value) {
        throw new AssertionError(`Variable ${name}: ${JSON.stringify(vStr)} != ${JSON.stringify(value)}`);
    }
}
export function checkVars(d) {
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
String.prototype.toBuffer = function () {
    return _aici.stringToBuffer(this);
};
String.fromBuffer = _aici.bufferToString;
Uint8Array.prototype.toString = function () {
    return _aici.bufferRepr(this);
};
Uint8Array.prototype.decode = function () {
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
