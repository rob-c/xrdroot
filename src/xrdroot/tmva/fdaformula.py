"""The formula of ``FDA``: TMVA's ``(i)`` parameters and ``xi`` variables, evaluated for many at once.

TMVA rewrites ``(0)+(1)*x0`` into ``[0]+[1]*[2]`` and hands it to
``TFormula``, one event and one parameter set at a time. Here the rewritten
text is parsed once, as the arithmetic ``TFormula`` knows - ``+ - * / ^``,
``**``, parentheses and its common functions, ``TMath::`` or not - into a
NumPy expression, so that one call evaluates every event for a whole batch of
parameter sets: parameters of shape ``(k, 1)`` against variables of shape
``(n,)`` give outputs of shape ``(k, n)``. Anything else in a formula is
refused by name when the method is booked.
"""

from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np

__all__ = ["FormulaError", "compile_fda", "tformula_text"]


class FormulaError(ValueError):
    """A formula FDA cannot use."""


#: The functions a formula may call, by the names ``TFormula`` knows them by.
FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "fabs": np.abs,
    "int": np.trunc,
    "pow": np.power,
    "Power": np.power,
    "Exp": np.exp,
    "Log": np.log,
    "Log10": np.log10,
    "Sqrt": np.sqrt,
    "Abs": np.abs,
    "Sin": np.sin,
    "Cos": np.cos,
    "Tan": np.tan,
    "TanH": np.tanh,
    "ATan": np.arctan,
    "ATan2": np.arctan2,
    "min": np.minimum,
    "max": np.maximum,
    "Min": np.minimum,
    "Max": np.maximum,
}
#: The operators, by their node types.
BINARY = {
    ast.Mod: np.fmod,
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
}


def tformula_text(formula: str, npars: int, nvar: int) -> str:
    """``CreateFormula``'s rewriting: ``(i)`` as ``[i]``, then ``xi`` as ``[i+npars]``."""
    text = formula
    for index in range(npars):
        text = text.replace(f"({index})", f"[{index}]")
    for index in range(npars, 1000):
        if f"({index})" in text:
            raise FormulaError(
                f'<CreateFormula> Formula contains expression: "({index})", which cannot be '
                "attributed to a parameter; it may be that the number of variable ranges given "
                'via "ParRanges" does not match the number of parameters in the formula '
                "expression, please verify!"
            )
    for index in reversed(range(nvar)):
        text = text.replace(f"x{index}", f"[{index + npars}]")
    for index in range(nvar, 1000):
        if f"x{index}" in text:
            raise FormulaError(
                f'<CreateFormula> Formula contains expression: "x{index}", which cannot be '
                "attributed to an input variable"
            )
    return text


def compile_fda(text: str) -> Any:
    """The rewritten formula as a function of ``(values)``, ``values[i]`` being its ``[i]``."""
    source = re.sub(r"\[(\d+)\]", r"_v[\1]", text.replace("TMath::", "").replace("^", "**"))
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        raise FormulaError(
            f'<ProcessOptions> Formula expression "{text}" could not be properly compiled'
        ) from None
    body = tree.body

    def run(values: list[Any]) -> Any:
        return _evaluate(body, values)

    _evaluate(body, None)
    return run


def _evaluate(node: ast.AST, values: list[Any] | None) -> Any:
    """The value of one node; with no values, only a check that the node is allowed."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, int):
            return None if values is None else values[index.value]
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        left, right = _evaluate(node.left, values), _evaluate(node.right, values)
        return None if values is None else BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _evaluate(node.operand, values)
        if values is None:
            return None
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        function = FUNCTIONS.get(node.func.id)
        if function is not None:
            arguments = [_evaluate(argument, values) for argument in node.args]
            return None if values is None else function(*arguments)
    raise FormulaError(
        f'"{ast.unparse(node)}" is not arithmetic xrdroot\'s FDA formula can evaluate: it knows '
        "numbers, the parameters and variables, + - * / ^ and TFormula's common functions"
    )
