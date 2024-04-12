# This module is provided as part of the pyaici package to help with auto-completion in IDEs
# while editing Python files to be uploaded to AICI server.
#
# It will not work with the standard Python interpreter.
#

from typing import Any, Optional, Coroutine, Union, Callable, List, Dict

# these are to provide re-exports
from pyaici.server_native import (
    TokenSet,
    tokenize,
    detokenize,
    RegexConstraint,
    CfgConstraint,
    SubStrConstraint,
    Constraint,
    get_config,
    get_var,
    set_var,
    append_var,
    eos_token,
    token_repr,
    tokens_repr,
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

    def add_splice(self, other: "Splice"):
        assert other.when_sampled == []
        if other.backtrack >= len(self.ff_tokens):
            self.backtrack += other.backtrack - len(self.ff_tokens)
            self.ff_tokens = other.ff_tokens[:]
        else:
            if other.backtrack > 0:
                del self.ff_tokens[-other.backtrack :]
            self.ff_tokens += other.ff_tokens


class Branch:
    def __init__(
        self, *, splices: List[Splice] = [], sample_mask: Optional[TokenSet] = None
    ) -> None:
        self.sample_mask = sample_mask
        self.splices = splices

    def is_splice(self) -> bool:
        return len(self.splices) == 1 and self.splices[0].when_sampled == []

    @classmethod
    def noop(cls):
        return cls(splices=[Splice(ff_tokens=[])])


def get_prompt_len() -> int:
    """
    Get the length of the prompt in the current sequence.
    """
    assert AiciAsync.instance
    return AiciAsync.instance._prompt_len


class MidProcessResult:
    def __init__(self, branches: List[Branch]):
        self.skip_me = False
        self.branches = branches

    def is_splice(self) -> bool:
        return len(self.branches) == 1 and self.branches[0].is_splice()

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
        """
        Use this to stop the entire generation process early.
        Otherwise, it stops when it reaches the end of generation co-routine.
        """
        return cls([])

    @classmethod
    def noop(cls):
        return cls([Branch.noop()])

    @classmethod
    def skip(cls):
        r = cls([])
        r.skip_me = True
        return r


class NextToken:
    """
    Awaiting this will return generated token (or tokens, if fast-forwarding requested by self.mid_process()).
    You have only ~1ms to process the results before awaiting a new instance of NextToken() again.
    """

    # to be overridden
    def mid_process(self) -> MidProcessResult:
        """
        This can be overridden to return a bias, fast-forward tokens, backtrack etc.
        """
        return MidProcessResult.bias(all_tokens())

    def post_process(self, backtrack: int, tokens: List[int]):
        """
        This can be overridden to do some processing after the token is generated.
        """
        pass

    def is_fixed(self) -> bool:
        """
        If true, the post_process() has to be empty and always self.mid_process().is_splice()
        """
        return False

    # internals
    def __init__(self) -> None:
        self.finished = False
        self._reset()

    def _reset(self):
        self.curr_tokens: Optional[List[Token]] = None
        self.value = None

    def _mid_process(self) -> MidProcessResult:
        if log_level >= 4:
            print(f"MID-PROCESS: {self}")
        self._reset()
        spl = self.is_fixed()
        r = self.mid_process()
        if spl:
            assert r.is_splice()
        return r

    def _post_process(self, backtrack: int, tokens: List[Token]):
        if log_level >= 3:
            print(f"POST-PROCESS: bt={backtrack} {tokens_repr(tokens)} {self}")
        self.curr_tokens = tokens
        self.value = tokens
        self.finished = eos_token() in tokens
        self.post_process(backtrack, tokens)

    def __await__(self):
        if log_level >= 4:
            print(f"AWAIT-IN: {self}")
        yield self
        if log_level >= 4:
            print(f"AWAIT-OUT: {self}")
        assert self.curr_tokens is not None
        return self.value


class Noop(NextToken):
    def __init__(self):
        super().__init__()

    def mid_process(self) -> MidProcessResult:
        return MidProcessResult.noop()

class FixedTokens(NextToken):
    def __init__(self, text: Union[str, bytes], following: Optional["Label"] = None):
        """
        Forces next tokens to be exactly the given text.
        If following is given, the text replaces everything that follows the label.
        """
        super().__init__()
        self.fixed_tokens: List[Token] = tokenize(text)
        if log_level >= 1:
            print(f"FIXED {tokens_repr(self.fixed_tokens)}")
        self.following = following

    def is_fixed(self) -> bool:
        return True

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
        Indicates that the generation should stop immedietely.
        """
        super().__init__()

    def mid_process(self) -> MidProcessResult:
        return MidProcessResult.stop()

    def post_process(self, backtrack: int, tokens: List[int]):
        self.finished = False  # we're never finished, just keep yelling STOP!


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
        # we build the constraint lazily, in mid_process() which has reasonably long time limit
        # TODO remove this
        if self._constraint is None:
            self._constraint = self.mk_constraint()
        bias = TokenSet()
        self._constraint.allow_tokens(bias)
        if log_level >= 2:
            print("ALLOW:", bias)
        if bias.num_set() == 0:
            if log_level >= 1:
                print("Constraint doesn't allow any tokens; adding EOS")
            bias[eos_token()] = True
        return MidProcessResult.bias(bias)

    def post_process(self, backtrack: int, tokens: List[int]):
        assert self._constraint
        assert backtrack == 0
        for t in tokens:
            self._constraint.append_token(t)
        if self._constraint.eos_forced():
            self.finished = True


class _Fork(NextToken):
    def __init__(self, forks: List[Branch]):
        super().__init__()
        self.forks = forks

    def mid_process(self) -> MidProcessResult:
        return MidProcessResult(self.forks)


def fork_supported():
    """
    Check if the current host supports forking.
    """
    return get_config("fork") != 0


async def fork(forks: Union[int, List[Branch]]):
    """
    Forks the execution into `num_forks` branches.
    Returns a number from 0 to `num_forks`-1, indicating the branch.
    """
    if isinstance(forks, int):
        forks = [Branch.noop() for _ in range(forks)]
    if not fork_supported() and len(forks) > 1:
        raise ValueError("Forking is disabled on this host")
    f = _Fork(forks)
    await f
    assert AiciAsync.instance
    fg = AiciAsync.instance.fork_group
    return fg.index(_aici.self_seq_id())


class _WaitVars(NextToken):
    def __init__(self, vars: List[str]):
        super().__init__()
        self.vars = vars
        self.values: List[bytes] = []

    def mid_process(self) -> MidProcessResult:
        values = [get_var(v) for v in self.vars]
        if None in values:
            return MidProcessResult.noop()
        self.values = values  # type: ignore
        return MidProcessResult.skip()


async def wait_vars(*vars: str) -> List[bytes]:
    """
    Suspends execution until all variables are available.
    Returns values of the variables.
    """
    w = _WaitVars(list(vars))
    if not w.vars:
        return []
    while w.values == []:
        await w
    return w.values


class AiciCallbacks:
    """
    Low-level interface for AICI.
    Use pyaici.server.start() to wrap a coroutine.
    """

    def init_prompt(self, prompt: List[Token]):
        pass

    def mid_process(
        self, backtrack: int, tokens: List[Token], fork_group: List[SeqId]
    ) -> MidProcessResult:
        return MidProcessResult.bias(all_tokens())


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

        self._prompt_len = 0
        self._coro = f
        self._tokens: List[Token] = []
        _aici.register(self)
        self._cb = None  # type: ignore
        self._prompt_cb: Optional[GetPrompt] = None
        self._went_ahead = False
        self.fork_group: List[SeqId] = []
        cb = self._step_core()
        if isinstance(cb, NextToken):
            self._cb: NextToken = cb
        else:
            assert isinstance(cb, GetPrompt)
            self._prompt_cb = cb

    def step(self):
        cb = self._step_core()
        assert isinstance(cb, NextToken)
        self._cb = cb

    def _step_core(self) -> CbType:
        if log_level >= 4:
            print("STEP")
        try:
            return self._coro.send(None)
        except StopIteration:

            async def _stop():
                while True:
                    await StopToken()

            self._coro = _stop()
            return self._step_core()

    def init_prompt(self, prompt: List[Token]):
        assert not self._tokens
        self._prompt_len = len(prompt)
        self._tokens.extend(prompt)
        if self._prompt_cb:
            self._prompt_cb.prompt = prompt
            self._prompt_cb = None
            self.step()
        self._went_ahead = True
        assert isinstance(self._cb, NextToken)

    def _apply_tokens(self, backtrack: int, tokens: List[Token]):
        if backtrack > 0:
            del self._tokens[-backtrack:]
        self._tokens.extend(tokens)
        self._cb._post_process(backtrack, tokens)
        self.step()

    def mid_process(
        self, backtrack: int, tokens: List[Token], fork_group: List[SeqId]
    ) -> MidProcessResult:
        assert isinstance(self._cb, NextToken)

        self.fork_group = fork_group

        if self._went_ahead:
            self._went_ahead = False
        else:
            self._apply_tokens(backtrack, tokens)

        r = self._mid_process_with_skip()
        r0 = r
        while r0.is_splice():
            spl = r0.branches[0].splices[0]
            self._apply_tokens(spl.backtrack, spl.ff_tokens)
            self._went_ahead = True
            if not self._cb.is_fixed():
                break
            r0 = self._mid_process_with_skip()
            assert r0.is_splice()
            r.branches[0].splices[0].add_splice(r0.branches[0].splices[0])

        return r

    def _mid_process_with_skip(self) -> MidProcessResult:
        while True:
            r: MidProcessResult = self._cb._mid_process()
            assert isinstance(r, MidProcessResult)
            if not r.skip_me:
                return r
            self._cb._post_process(0, [])
            self.step()


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
        if tokens:
            res += tokens

            if log_level >= 2:
                print("GEN-STEP:", tokens_repr(tokens))

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
        print("GEN:", tokens_repr(res))
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
