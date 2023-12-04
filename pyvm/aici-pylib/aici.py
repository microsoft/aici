from typing import Any, Optional, Coroutine, Union, Callable

# these are to provide re-exports
from _aici import (
    TokenSet,
    tokenize,
    detokenize,
    RegexConstraint,
    Constraint,
    get_var,
    set_var,
    append_var,
    eos_token,
)
import _aici

Token = int
SeqId = int


def get_tokens() -> list[Token]:
    assert AiciAsync.instance
    return AiciAsync.instance._tokens


def get_prompt_len() -> int:
    assert AiciAsync.instance
    return AiciAsync.instance._prompt_len


class MidProcessResult:
    def __init__(self, *, stop=False):
        self.stop = stop
        self.logit_bias: Optional[TokenSet] = None
        self.backtrack = 0
        self.ff_tokens: list[Token] = []

    @classmethod
    def bias(cls, bias: TokenSet):
        res = cls()
        res.logit_bias = bias
        return res

    @classmethod
    def splice(cls, backtrack: int, tokens: list[Token]):
        res = cls()
        assert backtrack >= 0
        assert isinstance(tokens, list)
        res.backtrack = backtrack
        res.ff_tokens = tokens
        return res


# Typically not needed.
class PreProcessResult:
    def __init__(self, *, suspended=False):
        self.suspended = suspended
        self.attention_masks: list[list[float]] = [[]]

    @classmethod
    def continue_(cls):
        return cls()

    @classmethod
    def suspend(cls):
        return cls(suspended=True)

    @classmethod
    def fork(cls, num_forks: int):
        res = cls()
        res.attention_masks = [[] for _ in range(num_forks)]
        return res


class PostProcessResult:
    def __init__(self, *, stop_seq=False):
        self.stop_seq = stop_seq

    @classmethod
    def continue_(cls):
        return cls()

    @classmethod
    def stop(cls):
        return cls(stop_seq=True)

    @classmethod
    def from_tokens(cls, tokens: list[int]):
        return cls(stop_seq=(eos_token() in tokens))


class NextToken:
    """
    Awaiting this will return generated token (or tokens, if fast-forwarding requested by self.mid_process()).
    You have only ~1ms to process the results before awaiting a new instance of NextToken() again.
    """

    # to be overridden
    def pre_process(self) -> PreProcessResult:
        """
        Override to suspend, if the model cannot continue generating tokens
        now (for example, not all variables are available to compute bias).
        ~1ms time limit.
        """
        return PreProcessResult.continue_()

    def mid_process(self) -> MidProcessResult:
        """
        This can be overridden to return a bias, fast-forward tokens, backtrack etc.
        ~20ms time limit.
        """
        return MidProcessResult.bias(TokenSet())

    def post_process(self, tokens: list[Token]):
        """
        This can be overridden to do something with generated tokens.
        ~1ms time limit.
        """
        return PostProcessResult.continue_()

    # internals
    def __init__(self) -> None:
        self.finished = False
        self._reset()

    def _reset(self):
        self.curr_tokens: Optional[list[Token]] = None
        self.fork_group: list[SeqId] = []

    def _pre_process(self) -> PreProcessResult:
        self._reset()
        return self.pre_process()

    def _mid_process(self, fork_group: list[SeqId]) -> MidProcessResult:
        self.fork_group = fork_group
        return self.mid_process()

    def _post_process(self, backtrack: int, tokens: list[Token]):
        # 'backtrack' is not very useful - it's just what we passed in MidProcessResult
        self.curr_tokens = tokens
        self.finished = eos_token() in tokens
        return self.post_process(tokens)

    def __await__(self):
        yield self
        assert self.curr_tokens is not None
        return self.curr_tokens


class FixedTokens(NextToken):
    def __init__(self, text: str | bytes, following: Optional["Label"] = None):
        super().__init__()
        self.fixed_tokens: list[Token] = tokenize(text)
        self.following = following

    def mid_process(self) -> MidProcessResult:
        backtrack = 0
        if self.following is not None:
            backtrack = len(get_tokens()) - self.following.ptr
            assert backtrack >= 0
            print("backtrack", backtrack)
        return MidProcessResult.splice(backtrack, tokens=self.fixed_tokens)


class StopToken(NextToken):
    def __init__(self) -> None:
        super().__init__()

    def mid_process(self) -> MidProcessResult:
        return MidProcessResult(stop=True)

    def post_process(self, tokens: list[Token]):
        self.finished = False  # we're never finished, just keep yelling STOP!
        return PostProcessResult.stop()


class ConstrainedToken(NextToken):
    def __init__(self, mk_constraint: Callable[[], Constraint]):
        super().__init__()
        self.mk_constraint = mk_constraint
        self._constraint: Constraint | None = None

    def mid_process(self) -> MidProcessResult:
        bias = TokenSet()
        # we build the constraint lazily, in mid_process() which has reasonably long time limit
        if self._constraint is None:
            self._constraint = self.mk_constraint()
        self._constraint.allow_tokens(bias)
        return MidProcessResult.bias(bias)

    def post_process(self, tokens: list[Token]):
        assert self._constraint is not None
        for t in tokens:
            self._constraint.append_token(t)
        return PostProcessResult.continue_()


class AiciCallbacks:
    """
    Low-level interface for AICI.
    Use aici_start() to wrap a coroutine.
    """

    def init_prompt(self, prompt: list[Token]):
        pass

    def pre_process(self) -> PreProcessResult:
        return PreProcessResult()

    def mid_process(self, fork_group: list[SeqId]) -> MidProcessResult:
        return MidProcessResult.bias(TokenSet())

    def post_process(self, backtrack: int, tokens: list[Token]):
        return PostProcessResult.from_tokens(tokens)


class GetPrompt:
    """
    Awaiting this returns the prompt passed by the user.
    The code before call to this function has a long time limit (~1000ms).
    Afterwards, the time limit is ~1ms before awaiting NextToken().
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

        self._coro = f
        self._skip_prompt = False
        self._tokens: list[Token] = []
        self._prompt_len = 0
        _aici.register(self)
        self.step()
        if isinstance(self._cb, NextToken):
            self._skip_prompt = True
        else:
            assert isinstance(self._cb, GetPrompt)

    def step(self):
        try:
            self._cb: CbType = self._coro.send(None)
        except StopIteration:

            async def _stop():
                while True:
                    await StopToken()

            self._coro = _stop()

    def init_prompt(self, prompt: list[Token]):
        assert not self._tokens
        self._prompt_len = len(prompt)
        self._tokens.extend(prompt)

        if self._skip_prompt:
            self._skip_prompt = False
            return

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

    def mid_process(self, fork_group: list[SeqId]) -> MidProcessResult:
        assert isinstance(self._cb, NextToken)
        r = self._cb._mid_process(fork_group)
        assert isinstance(r, MidProcessResult)
        assert isinstance(r.ff_tokens, list)
        return r

    def post_process(self, backtrack: int, tokens: list[Token]):
        if backtrack > 0:
            del self._tokens[-backtrack:]
        self._tokens.extend(tokens)

        assert isinstance(self._cb, NextToken)
        r = self._cb._post_process(backtrack, tokens)
        assert isinstance(r, PostProcessResult)
        self.step()
        assert isinstance(self._cb, NextToken)
        return r


def aici_start(f: Coroutine[CbType, None, None]):
    """
    Starts the AICI loop.
    The coroutine may first `await GetPrompt()` and then should `await NextToken()` (typically in a loop).
    """
    return AiciAsync(f)


class Label:
    def __init__(self):
        self.ptr = len(get_tokens())


class ChooseConstraint(Constraint):
    def __init__(self, options: list[str]):
        # super().__init__()
        self.ptr = 0
        self.options = [tokenize(o) for o in options]

    def eos_allowed(self) -> bool:
        return any(len(o) == self.ptr for o in self.options)

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
    regex: str | None = None,
    options: list[str] | None = None,
    store_var: str | None = None,
    max_tokens=20,
) -> list[Token]:
    res: list[Token] = []
    if regex is not None:
        assert options is None
        next_token = ConstrainedToken(lambda: RegexConstraint(regex))
    elif options is not None:
        next_token = ConstrainedToken(lambda: ChooseConstraint(options))
    else:
        next_token = ConstrainedToken(lambda: Constraint())
    for _ in range(max_tokens):
        t = await next_token
        res += t
        if next_token.finished:
            break
    if store_var is not None:
        set_var(store_var, detokenize(res))
    return res
