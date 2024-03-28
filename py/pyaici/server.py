# This module is provided as part of the pyaici package to help with auto-completion in IDEs
# while editing Python files to be uploaded to AICI server.
#
# It will not work with the standard Python interpreter.
#

from typing import Any, Optional, Coroutine, Union, Callable, List, Union, Dict

# these are to provide re-exports
from pyaici.server_native import (
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
    token_repr,
)
import pyaici.server_native as _aici

Token = int
SeqId = int

log_level = 1


def all_tokens():
    ts = TokenSet()
    ts.set_all(True)
    return ts


def get_tokens() -> List[Token]:
    """
    Get list of tokens in the current sequence, including the prompt.
    """
    assert AiciAsync.instance
    return AiciAsync.instance._tokens


class Splice:
    def __init__(
        self,
        *,
        when_sampled: List[Token] = [],
        backtrack: int = 0,
        ff_tokens: List[Token],
    ):
        self.when_sampled = when_sampled
        self.backtrack = backtrack
        self.ff_tokens = ff_tokens


class Branch:
    def __init__(
        self, *, splices: List[Splice] = [], sample_mask: Optional[TokenSet] = None
    ) -> None:
        self.sample_mask = sample_mask
        self.splices = splices


def get_prompt_len() -> int:
    """
    Get the length of the prompt in the current sequence.
    """
    assert AiciAsync.instance
    return AiciAsync.instance._prompt_len


class MidProcessResult:
    def __init__(self, branches: List[Branch]):
        self.branches = branches

    @classmethod
    def bias(cls, bias: TokenSet):
        return cls([Branch(sample_mask=bias)])

    @classmethod
    def splice(cls, backtrack: int, ff_tokens: List[Token]):
        assert backtrack >= 0
        assert isinstance(ff_tokens, list)
        return cls([Branch(splices=[Splice(backtrack=backtrack, ff_tokens=ff_tokens)])])

    @classmethod
    def stop(cls):
        return cls([])


class NextToken:
    """
    Awaiting this will return generated token (or tokens, if fast-forwarding requested by self.mid_process()).
    You have only ~1ms to process the results before awaiting a new instance of NextToken() again.
    """

    # to be overridden
    def mid_process(self) -> MidProcessResult:
        """
        This can be overridden to return a bias, fast-forward tokens, backtrack etc.
        ~20ms time limit.
        """
        return MidProcessResult.bias(all_tokens())

    # internals
    def __init__(self) -> None:
        self.finished = False
        self._reset()

    def _reset(self):
        self.curr_tokens: Optional[List[Token]] = None
        self.fork_group: List[SeqId] = []

    def _mid_process(self, fork_group: List[SeqId]) -> MidProcessResult:
        self.fork_group = fork_group
        return self.mid_process()

    def _post_process(self, backtrack: int, tokens: List[Token]):
        # 'backtrack' is not very useful - it's just what we passed in MidProcessResult
        self.curr_tokens = tokens
        self.finished = eos_token() in tokens
        return self.post_process(tokens)

    def __await__(self):
        yield self
        assert self.curr_tokens is not None
        return self.curr_tokens


class FixedTokens(NextToken):
    def __init__(self, text: Union[str, bytes], following: Optional["Label"] = None):
        """
        Forces next tokens to be exactly the given text.
        If following is given, the text replaces everything that follows the label.
        """
        super().__init__()
        self.fixed_tokens: List[Token] = tokenize(text)
        if log_level >= 1:
            print("FIXED", repr(detokenize(self.fixed_tokens).decode(errors="replace")))
        self.following = following

    def pre_process(self) -> PreProcessResult:
        if self.following is None:
            return PreProcessResult.ff_tokens_pre(self.fixed_tokens)
        return PreProcessResult.continue_()

    def mid_process(self) -> MidProcessResult:
        backtrack = 0
        if self.following is not None:
            backtrack = len(get_tokens()) - self.following.ptr
            assert backtrack >= 0
            if log_level >= 1:
                print("BACKTRACK", backtrack)
        return MidProcessResult.splice(backtrack, ff_tokens=self.fixed_tokens)


class StopToken(NextToken):
    def __init__(self) -> None:
        """
        Indicates that the generation should stop.
        """
        super().__init__()

    def mid_process(self) -> MidProcessResult:
        return MidProcessResult(stop=True)

    def post_process(self, tokens: List[Token]):
        self.finished = False  # we're never finished, just keep yelling STOP!
        return PostProcessResult.stop()


class ConstrainedToken(NextToken):
    def __init__(self, mk_constraint: Callable[[], Constraint]):
        """
        Generates a token that satisfies the given constraint.
        The constraint will be constructed in mid_process() phase, which has slightly longer time limit.
        """
        super().__init__()
        self.mk_constraint = mk_constraint
        self._constraint: Optional[Constraint] = None

    def mid_process(self) -> MidProcessResult:
        bias = TokenSet()
        # we build the constraint lazily, in mid_process() which has reasonably long time limit
        if self._constraint is None:
            self._constraint = self.mk_constraint()
        self._constraint.allow_tokens(bias)
        if log_level >= 2:
            print("ALLOW:", bias)
        if bias.num_set() == 0:
            if log_level >= 1:
                print("Constraint doesn't allow any tokens; adding EOS")
            bias[eos_token()] = True
        return MidProcessResult.bias(bias)

    def post_process(self, tokens: List[Token]):
        assert self._constraint is not None
        for t in tokens:
            self._constraint.append_token(t)
        if self._constraint.eos_forced():
            self.finished = True
        return PostProcessResult.continue_()


class PreToken(NextToken):
    def __await__(self):
        yield self
        return None

    def mid_process(self) -> MidProcessResult:
        return MidProcessResult(skip_me=True)


class _Fork(PreToken):
    def __init__(self, num_forks: int):
        super().__init__()
        self.num_forks = num_forks

    def pre_process(self) -> PreProcessResult:
        return PreProcessResult.fork(self.num_forks)


async def fork(num_forks: int):
    """
    Forks the execution into `num_forks` branches.
    Returns a number from 0 to `num_forks`-1, indicating the branch.
    """
    f = _Fork(num_forks)
    await f
    return f.fork_group.index(_aici.self_seq_id())


class _WaitVars(PreToken):
    def __init__(self, vars: List[str]):
        super().__init__()
        self.vars = vars
        self.values: List[bytes] = []

    def pre_process(self) -> PreProcessResult:
        values = [get_var(v) for v in self.vars]
        if None in values:
            return PreProcessResult.suspend()
        self.values = values  # type: ignore
        return PreProcessResult.continue_()


async def wait_vars(*vars: str) -> List[bytes]:
    """
    Suspends execution until all variables are available.
    Returns values of the variables.
    """
    w = _WaitVars(list(vars))
    await w
    return w.values


class AiciCallbacks:
    """
    Low-level interface for AICI.
    Use pyaici.server.start() to wrap a coroutine.
    """

    def init_prompt(self, prompt: List[Token]):
        pass

    def mid_process(self, fork_group: List[SeqId]) -> MidProcessResult:
        ts = TokenSet()
        ts.set_all(True)
        return MidProcessResult.bias(ts)


class GetPrompt:
    """
    Awaiting this returns the prompt passed by the user.
    The code before call to this function has a long time limit (~1000ms).
    """

    def __init__(self) -> None:
        self.prompt: Optional[list[Token]] = None

    def __await__(self):
        yield self
        assert self.prompt is not None
        return self.prompt


CbType = Union[GetPrompt, NextToken]


class AiciAsync(AiciCallbacks):
    instance: Optional["AiciAsync"] = None

    def __init__(self, f: Coroutine[CbType, None, None]):
        assert AiciAsync.instance is None
        AiciAsync.instance = self

        self._skip_prompt = False
        self._prompt_len = 0
        self._coro = f
        self._tokens: List[Token] = []
        self._pending_cb: Optional[CbType] = None
        self._fork_group: List[SeqId] = []
        _aici.register(self)
        self.step()
        if isinstance(self._cb, NextToken):
            self._skip_prompt = True
        else:
            assert isinstance(self._cb, GetPrompt)

    def step(self):
        if self._pending_cb is not None:
            self._cb = self._pending_cb
            self._pending_cb = None
            return

        try:
            self._cb: CbType = self._coro.send(None)
        except StopIteration:

            async def _stop():
                while True:
                    await StopToken()

            self._coro = _stop()

    def init_prompt(self, prompt: List[Token]):
        assert not self._tokens
        self._prompt_len = len(prompt)
        self._tokens.extend(prompt)
        if self._skip_prompt:
            self._skip_prompt = False
        else:
            assert isinstance(self._cb, GetPrompt)
            self._cb.prompt = prompt
            self.step()
        assert isinstance(self._cb, NextToken)

    def pre_process(self) -> PreProcessResult:
        assert isinstance(self._cb, NextToken)
        if self._cb.finished:
            self._cb = StopToken()
        r = self._cb._pre_process()
        assert isinstance(r, PreProcessResult)
        return r

    def mid_process(
        self, backtrack: int, tokens: List[Token], fork_group: List[SeqId]
    ) -> MidProcessResult:
        assert isinstance(self._cb, NextToken)

        if backtrack > 0:
            del self._tokens[-backtrack:]
        self._tokens.extend(tokens)

        r = self._cb._mid_process(fork_group)
        assert isinstance(r, MidProcessResult)

        while r.skip_me:
            self.step()
            assert isinstance(self._cb, NextToken)
            r2 = self._cb._pre_process()
            assert isinstance(r2, PreProcessResult)
            assert r2.num_forks == 1, "nested fork not allowed"
            if r2.suspended:
                # need to generate one fake token...
                self._pending_cb = self._cb
                f = FixedTokens(" ")
                assert len(f.fixed_tokens) == 1
                self._cb = f
            r = self._cb._mid_process(fork_group)
            assert isinstance(r, MidProcessResult)

        assert isinstance(r.ff_tokens, list)
        return r

    def post_process(self, backtrack: int, tokens: List[Token]):
        if backtrack > 0:
            del self._tokens[-backtrack:]
        self._tokens.extend(tokens)

        assert isinstance(self._cb, NextToken)
        r = self._cb._post_process(backtrack, tokens)
        assert isinstance(r, PostProcessResult)
        self.step()
        assert isinstance(self._cb, NextToken)
        return r


def start(f: Coroutine[CbType, None, None]):
    """
    Starts the AICI loop.
    The coroutine may first `await getPrompt()` and then can `await gen_*()` or
    `await FixedTokens()` multiple times.
    """
    return AiciAsync(f)


def test(f: Coroutine[CbType, None, None]):
    """
    Runs the loop as a test.
    """

    async def wrap():
        await f
        print("TEST OK")

    return AiciAsync(wrap())


class Label:
    def __init__(self):
        """
        Create a new label the indicates the current position in the sequence.
        Can be passed as `following=` argument to `FixedTokens()`.
        """
        self.ptr = len(get_tokens())

    def tokens_since(self) -> List[Token]:
        """
        Return tokens generated since the label.
        """
        return get_tokens()[self.ptr :]

    def text_since(self) -> str:
        """
        Return text generated since the label.
        """
        return detokenize(self.tokens_since()).decode(errors="replace")


class ChooseConstraint(Constraint):
    def __init__(self, options: List[str]):
        # super().__init__()
        self.ptr = 0
        self.options = [tokenize(o) for o in options]

    def eos_allowed(self) -> bool:
        return any(len(o) == self.ptr for o in self.options)

    def eos_forced(self) -> bool:
        return len(self.options) == 1 and len(self.options[0]) == self.ptr

    def token_allowed(self, t: int) -> bool:
        return any(self.ptr < len(o) and o[self.ptr] == t for o in self.options)

    def append_token(self, t: int):
        self.options = [
            o for o in self.options if self.ptr < len(o) and o[self.ptr] == t
        ]
        self.ptr += 1

    def allow_tokens(self, ts: TokenSet):
        for o in self.options:
            if self.ptr < len(o):
                ts[o[self.ptr]] = True
            elif self.ptr == len(o):
                ts[eos_token()] = True


async def gen_tokens(
    regex: Optional[str] = None,
    yacc: Optional[str] = None,
    substring: Optional[str] = None,
    substring_end: str = '"',
    options: Optional[List[str]] = None,
    store_var: Optional[str] = None,
    stop_at: Optional[str] = None,
    max_tokens=20,
) -> List[Token]:
    """
    Generates tokens with the given constraint.
    If `stop_at` is given, the generation stops when the given text is generated. The stop text is included in result.
    If `store_var` is given, the generated tokens are stored in the variable.
    `regex` and `options` are mutually exclusive.
    """
    res: List[Token] = []
    assert len([x for x in [regex, options, yacc, substring] if x is not None]) <= 1
    if regex is not None:
        next_token = ConstrainedToken(lambda: RegexConstraint(regex))
    elif substring is not None:
        next_token = ConstrainedToken(
            lambda: SubStrConstraint(substring, substring_end)
        )
    elif yacc is not None:
        next_token = ConstrainedToken(lambda: CfgConstraint(yacc))
    elif options is not None:
        next_token = ConstrainedToken(lambda: ChooseConstraint(options))
    else:
        next_token = ConstrainedToken(lambda: Constraint())
    for _ in range(max_tokens):
        tokens = await next_token
        res += tokens

        if log_level >= 2:
            print("GEN-STEP:", ", ".join([token_repr(t) for t in tokens]))

        # this may get slow when the output is veeeeeery long
        # not a problem for a few k tokens
        text = detokenize(res).decode(errors="replace")

        if stop_at is not None:
            if stop_at in text:
                break

        if next_token.finished:
            break
    if store_var is not None:
        set_var(store_var, detokenize(res))
    if log_level >= 1:
        print("GEN", repr(detokenize(res).decode(errors="replace")))
    return res


async def gen_text(**kwargs: Any) -> str:
    """
    Same as gen_tokens(), but tries to decode the output as text.
    """
    tokens = await gen_tokens(**kwargs)
    return detokenize(tokens).decode(errors="replace")


def check_var(name: str, value: str):
    """
    Check if the variable has the given value.
    """
    v = get_var(name)
    if v is None:
        raise AssertionError(f"Variable {name} is unset")
    v = v.decode()
    if v != value:
        raise AssertionError(f"Variable {name}: {repr(v)} != {repr(value)}")


def check_vars(d: Dict[str, str]):
    """
    Check if all the variables have the given values.
    """
    for k, v in d.items():
        check_var(k, v)
