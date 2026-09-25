"""A parsed formula, evaluated over whole arrays and differentiated in its parameters.

Each node evaluates itself against an :class:`Env` - the variables as
arrays, one per dimension, and the parameters - and answers the same shape
the variables are. It can also give its derivative with respect to one
parameter alongside its value, forward through the tree by the chain rule:
that is what makes the gradient of ``[0]*exp(-x/[1]) + [2]`` exact rather
than a difference of two evaluations.

A derivative is an array, or :data:`ZERO` when the node does not depend on
the parameter at all - which is most nodes for most parameters, and saves
the arithmetic - or ``None`` when the node cannot say, because it calls a
function whose derivative is not known here. ``None`` is contagious, and a
parameter it reaches is differentiated numerically instead.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Optional, Union

import numpy as np

from .library import Call, power_partials

__all__ = ["Env", "Node", "Const", "Var", "Param", "Unary", "Binary", "Cond", "Apply", "ZERO"]

Array = Any
#: A node's derivative with respect to one parameter: nothing, an array, or unknown.
Tangent = Optional[Union[int, Array]]
#: The derivative of a node that does not depend on the parameter asked about.
ZERO = 0


def is_zero(tangent: Tangent) -> bool:
    """Whether a derivative is :data:`ZERO`, which is the one that is an ``int``."""
    return isinstance(tangent, int)


def _add(first: Tangent, second: Tangent) -> Tangent:
    if first is None or second is None:
        return None
    if is_zero(first):
        return second
    if is_zero(second):
        return first
    return first + second


def _scale(tangent: Tangent, factor: Callable[[], Array]) -> Tangent:
    """``tangent * factor()``, with the factor worked out only when it matters."""
    if tangent is None or is_zero(tangent):
        return tangent
    return tangent * factor()


class Env:
    """What a formula is evaluated against: the variables, and the parameters."""

    __slots__ = ("variables", "params")

    def __init__(self, variables: list[Array], params: Array) -> None:
        self.variables = variables
        self.params = params


class Node:
    """Anything a formula is made of."""

    __slots__ = ()

    def evaluate(self, env: Env) -> Array:
        raise NotImplementedError

    def dual(self, env: Env, k: int) -> tuple[Array, Tangent]:
        """The value, and the derivative with respect to parameter ``k``."""
        raise NotImplementedError

    def children(self) -> tuple[Node, ...]:
        return ()

    def walk(self) -> Iterator[Node]:
        """This node and every node under it."""
        yield self
        for child in self.children():
            yield from child.walk()


class Const(Node):
    """A number, or a named constant such as ``pi``."""

    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def evaluate(self, env: Env) -> Array:
        return self.value

    def dual(self, env: Env, k: int) -> tuple[Array, Tangent]:
        return self.value, ZERO


class Var(Node):
    """``x``, ``y``, ``z``, or ``x[i]``: one of the coordinates."""

    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index

    def evaluate(self, env: Env) -> Array:
        return env.variables[self.index]

    def dual(self, env: Env, k: int) -> tuple[Array, Tangent]:
        return env.variables[self.index], ZERO


class Param(Node):
    """``[i]``, however the parameter was named in the text."""

    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index

    def evaluate(self, env: Env) -> Array:
        return env.params[self.index]

    def dual(self, env: Env, k: int) -> tuple[Array, Tangent]:
        return env.params[self.index], (1.0 if self.index == k else ZERO)


#: What each unary operator does to a value.
UNARY: dict[str, Callable[[Array], Array]] = {
    "-": np.negative,
    "+": np.positive,
    "!": lambda value: (np.asarray(value) == 0).astype(np.float64),
}


class Unary(Node):
    """``-a``, ``+a`` or ``!a``."""

    __slots__ = ("op", "operand")

    def __init__(self, op: str, operand: Node) -> None:
        self.op = op
        self.operand = operand

    def children(self) -> tuple[Node, ...]:
        return (self.operand,)

    def evaluate(self, env: Env) -> Array:
        return UNARY[self.op](self.operand.evaluate(env))

    def dual(self, env: Env, k: int) -> tuple[Array, Tangent]:
        value, tangent = self.operand.dual(env, k)
        if self.op == "!":
            return UNARY["!"](value), ZERO
        if self.op == "-":
            return -value, _scale(tangent, lambda: -1.0)
        return value, tangent


def _truth(values: Array) -> Array:
    return np.asarray(values) != 0


def _flag(ufunc: Callable[..., Array]) -> Callable[[Array, Array], Array]:
    """A comparison or a logical operator, answering one or zero as C++ does."""
    return lambda a, b: ufunc(a, b).astype(np.float64)


#: What each binary operator does to two values.
BINARY: dict[str, Callable[[Array, Array], Array]] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
    "%": np.fmod,
    "^": np.power,
    "<": _flag(np.less),
    "<=": _flag(np.less_equal),
    ">": _flag(np.greater),
    ">=": _flag(np.greater_equal),
    "==": _flag(np.equal),
    "!=": _flag(np.not_equal),
    "&&": lambda a, b: np.logical_and(_truth(a), _truth(b)).astype(np.float64),
    "||": lambda a, b: np.logical_or(_truth(a), _truth(b)).astype(np.float64),
}

#: The operators whose answer is flat wherever it is defined.
STEPS = ("<", "<=", ">", ">=", "==", "!=", "&&", "||")

Rule = Callable[[Array, Tangent, Array, Tangent], Tangent]


def _sum_rule(a: Array, da: Tangent, b: Array, db: Tangent) -> Tangent:
    return _add(da, db)


def _difference_rule(a: Array, da: Tangent, b: Array, db: Tangent) -> Tangent:
    return _add(da, _scale(db, lambda: -1.0))


def _product_rule(a: Array, da: Tangent, b: Array, db: Tangent) -> Tangent:
    return _add(_scale(da, lambda: b), _scale(db, lambda: a))


def _quotient_rule(a: Array, da: Tangent, b: Array, db: Tangent) -> Tangent:
    return _add(_scale(da, lambda: 1.0 / b), _scale(db, lambda: -a / (b * b)))


def _power_rule(a: Array, da: Tangent, b: Array, db: Tangent) -> Tangent:
    return _add(
        _scale(da, lambda: power_partials(a, b)[0]), _scale(db, lambda: power_partials(a, b)[1])
    )


def _remainder_rule(a: Array, da: Tangent, b: Array, db: Tangent) -> Tangent:
    return _add(da, _scale(db, lambda: -np.trunc(a / b)))


#: How each arithmetic operator passes a derivative on.
RULES: dict[str, Rule] = {
    "+": _sum_rule,
    "-": _difference_rule,
    "*": _product_rule,
    "/": _quotient_rule,
    "^": _power_rule,
    "%": _remainder_rule,
}


class Binary(Node):
    """Two operands and the operator between them."""

    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: Node, right: Node) -> None:
        self.op = op
        self.left = left
        self.right = right

    def children(self) -> tuple[Node, ...]:
        return (self.left, self.right)

    def evaluate(self, env: Env) -> Array:
        return BINARY[self.op](self.left.evaluate(env), self.right.evaluate(env))

    def dual(self, env: Env, k: int) -> tuple[Array, Tangent]:
        a, da = self.left.dual(env, k)
        b, db = self.right.dual(env, k)
        value = BINARY[self.op](a, b)
        if self.op in STEPS:
            return value, ZERO
        return value, RULES[self.op](a, da, b, db)


class Cond(Node):
    """``condition ? then : otherwise``, element by element."""

    __slots__ = ("condition", "then", "otherwise")

    def __init__(self, condition: Node, then: Node, otherwise: Node) -> None:
        self.condition = condition
        self.then = then
        self.otherwise = otherwise

    def children(self) -> tuple[Node, ...]:
        return (self.condition, self.then, self.otherwise)

    def evaluate(self, env: Env) -> Array:
        chosen = _truth(self.condition.evaluate(env))
        return np.where(chosen, self.then.evaluate(env), self.otherwise.evaluate(env))

    def dual(self, env: Env, k: int) -> tuple[Array, Tangent]:
        chosen = _truth(self.condition.evaluate(env))
        then, dthen = self.then.dual(env, k)
        otherwise, dotherwise = self.otherwise.dual(env, k)
        value = np.where(chosen, then, otherwise)
        if dthen is None or dotherwise is None:
            return value, None
        if is_zero(dthen) and is_zero(dotherwise):
            return value, ZERO
        return value, np.where(chosen, dthen, dotherwise)


class Apply(Node):
    """A call of one of the functions in :data:`~.library.CALLS`."""

    __slots__ = ("name", "call", "args")

    def __init__(self, name: str, call: Call, args: tuple[Node, ...]) -> None:
        self.name = name
        self.call = call
        self.args = args

    def children(self) -> tuple[Node, ...]:
        return self.args

    def evaluate(self, env: Env) -> Array:
        return self.call.apply(*(arg.evaluate(env) for arg in self.args))

    def dual(self, env: Env, k: int) -> tuple[Array, Tangent]:
        pairs = [arg.dual(env, k) for arg in self.args]
        values = [value for value, _ in pairs]
        return self.call.apply(*values), _chain(self.call, values, [t for _, t in pairs])


def _chain(call: Call, values: list[Array], tangents: list[Tangent]) -> Tangent:
    """The chain rule through a call: each argument's derivative times the call's partial."""
    if all(tangent is not None and is_zero(tangent) for tangent in tangents):
        return ZERO
    if call.partials is None or any(tangent is None for tangent in tangents):
        return None
    total: Tangent = ZERO
    for tangent, partial in zip(tangents, call.partials(*values)):
        if not is_zero(tangent):
            total = _add(total, tangent * partial)
    return total
