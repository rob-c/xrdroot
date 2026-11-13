"""The translator's code names nothing of any library's: a library's names are its dialect's."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import xrdroot.pycxx

#: What one of ROOT's names looks like - ``ROOT::``, ``RooRealVar``, ``RVec``, ``R__ASSERT``,
#: ``Double_t`` - the library whose dialect is the first one written, and so the likeliest to
#: leak back into the translator. (``XRDROOT``, this project's own name, is not one.)
ROOTS = re.compile(
    r"(?<![A-Z])ROOT|RVec|TMath|TString|TTree|\bRoo[A-Z]|R__|pyroot|pycint|\b[A-Z][A-Za-z0-9]*_t\b"
)


def _strings(path: Path) -> list[tuple[int, str]]:
    """Every string constant in a module's code that is not a docstring."""
    tree = ast.parse(path.read_text())
    docs = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            docs.add(id(body[0].value))
    return [(node.lineno, node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docs]  # fmt: skip


def test_no_string_in_the_translators_code_names_one_of_roots() -> None:
    package = Path(xrdroot.pycxx.__file__).parent
    found = [
        f"{path.relative_to(package)}:{line}: {text!r}"
        for path in sorted(package.rglob("*.py"))
        for line, text in _strings(path)
        if ROOTS.search(text)
    ]
    assert found == []
