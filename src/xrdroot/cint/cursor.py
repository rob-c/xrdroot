"""Walking a macro's tokens: looking ahead, backing up, and refusing at the right line.

C++ cannot be parsed without guessing - ``a * b;`` is a declaration or a
product depending on what ``a`` is - so the parser tries one reading, and
backs up to try the other when the first does not fit. :meth:`Cursor.trial`
is that: it runs a parse and, if the parse does not fit, puts the cursor
back where it was and says so with ``None``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TypeVar

from .ctype import BUILTIN_WORDS, ROOT_TYPEDEFS
from .errors import Refusal, Where
from .tokens import Token

__all__ = ["Cursor", "NoParse", "KEYWORDS", "STD_NAMES", "looks_like_type"]

T = TypeVar("T")

#: C++'s reserved words that can never be the name of a variable or a type.
KEYWORDS = frozenset(
    """alignas alignof asm auto bool break case catch char char16_t char32_t class const
    constexpr const_cast continue decltype default delete do double dynamic_cast else enum
    explicit export extern false float for friend goto if inline int long mutable namespace
    new noexcept nullptr operator private protected public register reinterpret_cast return
    short signed sizeof static static_assert static_cast struct switch template this
    thread_local throw true try typedef typeid typename union unsigned using virtual void
    volatile wchar_t while override final""".split()
)

#: Names the standard library declares that a macro uses unqualified after ``using namespace std``.
STD_NAMES = frozenset(
    """string vector map multimap unordered_map set multiset unordered_set list deque array pair
    tuple make_pair make_tuple unique_ptr shared_ptr make_unique make_shared function
    numeric_limits complex cout cerr clog endl flush setw setprecision setfill fixed
    scientific left right boolalpha noboolalpha ostringstream stringstream istringstream
    ifstream ofstream fstream ostream istream sort min max swap abs fabs sqrt pow exp log
    log10 sin cos tan atan atan2 floor ceil round accumulate iota fill copy find
    count reverse to_string stoi stod stof getline ios string_view hex dec showpos
    noshowpos distance begin end transform any_of all_of none_of max_element min_element
    iterator size_t get tie ref cref move forward runtime_error exception
    exp2 log2 cbrt hypot fmod erf erfc tgamma lgamma asin acos sinh cosh tanh trunc isnan
    isinf""".split()
)

#: The words that begin a template argument list after them, whatever else is known.
KNOWN_TEMPLATES = frozenset(
    """vector map multimap unordered_map set multiset unordered_set list deque array pair
    tuple make_pair make_tuple unique_ptr shared_ptr make_unique make_shared function
    numeric_limits complex get basic_string static_cast dynamic_cast reinterpret_cast
    const_cast RVec TMatrixT TMatrixTSym TVectorT TMatrixTSparse TParameter TTreeReaderValue
    TTreeReaderArray RResultPtr RNode TArrayT atomic optional variant valarray bitset
    initializer_list less greater hash reference_wrapper weak_ptr remove_reference decay
    is_same enable_if RVecOps span TNDArrayT THnT THnSparseT RHist RVec RTensor
    TMatrixTRow TMatrixTColumn TMatrixTDiag""".split()
)

#: What a ROOT class name looks like, when nothing declared it: TH1F, RooRealVar, RVec...
ROOT_CLASS = re.compile(r"^(?:T[A-Z]|Roo[A-Z]|R[A-Z][a-z]|TMVA|Math[A-Z])\w*$")

#: The namespaces whose members named with a capital are types rather than functions.
TYPE_NAMESPACES = frozenset({"ROOT", "RooFit", "RooStats", "TMVA", "Math", "RDF", "VecOps"})

#: The standard names of types, as opposed to the functions std also declares.
STD_TYPES = frozenset(
    """string vector map multimap unordered_map set multiset unordered_set list deque array pair
    tuple unique_ptr shared_ptr function numeric_limits complex ostringstream stringstream
    istringstream ifstream ofstream fstream ostream istream string_view size_t iterator
    runtime_error exception optional variant atomic bitset valarray initializer_list
    int32_t uint32_t int64_t uint64_t uint8_t ptrdiff_t mutex thread chrono mt19937
    default_random_engine uniform_real_distribution normal_distribution""".split()
)


class NoParse(Exception):
    """A trial reading did not fit; the cursor goes back and another is tried."""


def looks_like_type(parts: list[str], known: set[str]) -> bool:
    """Would ``a::b::c`` be the name of a type, from its spelling and what has been declared?"""
    last = parts[-1]
    if last in known or last in BUILTIN_WORDS or "::".join(parts) in ROOT_TYPEDEFS:
        return True
    if len(parts) > 1:
        return _qualified_type(parts, known)
    return bool(ROOT_CLASS.match(last)) or (last.endswith("_t") and last[:1].isupper())


def _qualified_type(parts: list[str], known: set[str]) -> bool:
    last = parts[-1]
    if parts[0] == "std":
        return last in STD_TYPES or last.endswith("_t")
    if parts[-2] in TYPE_NAMESPACES or parts[-2] in known:
        return last[:1].isupper()
    return bool(ROOT_CLASS.match(last))


class Cursor:
    """A position in a macro's tokens, and the names declared so far that guide the guesses."""

    def __init__(self, tokens: list[Token]) -> None:
        end = tokens[-1].where if tokens else Where("<macro>", 1)
        self.tokens = [*tokens, Token("eof", "", end)]
        self.at = 0
        #: The names declared as types: classes, typedefs, enums, template parameters.
        self.types: set[str] = set()
        #: The names declared as templates.
        self.templates: set[str] = set()
        #: The variables in scope, innermost last - a declared variable is never a type.
        self.variables: list[set[str]] = [set()]
        #: Nonzero while ``>`` closes a template argument list rather than comparing.
        self.angle = 0

    # -- looking --------------------------------------------------------------

    def peek(self, ahead: int = 0) -> Token:
        index = min(self.at + ahead, len(self.tokens) - 1)
        return self.tokens[index]

    @property
    def where(self) -> Where:
        return self.peek().where

    def at_(self, *texts: str) -> bool:
        token = self.peek()
        return token.kind in ("op", "id") and token.text in texts

    def take(self) -> Token:
        token = self.peek()
        if token.kind == "eof":
            raise Refusal("the macro ends in the middle of a declaration or statement", token.where)
        self.at += 1
        return token

    def accept(self, *texts: str) -> bool:
        if self.at_(*texts):
            self.at += 1
            return True
        return False

    def expect(self, text: str) -> Token:
        if not self.at_(text):
            self.fail(f"{text!r} was expected")
        return self.take()

    def identifier(self) -> str:
        token = self.peek()
        if token.kind != "id" or token.text in KEYWORDS:
            self.fail("a name was expected")
        self.at += 1
        return token.text

    def fail(self, why: str) -> None:
        """Refuse what is at the cursor: the parse cannot go on from here."""
        token = self.peek()
        found = "the end of the macro" if token.kind == "eof" else repr(token.text)
        raise Refusal(f"{why}, and {found} is there instead", token.where)

    def refuse(self, why: str, where: Where | None = None) -> Refusal:
        """A refusal of a construct, by name, at ``where`` or the cursor."""
        return Refusal(why, where or self.where)

    # -- guessing -------------------------------------------------------------

    def trial(self, parse: Callable[[], T]) -> T | None:
        """What ``parse`` reads from here, or ``None`` with the cursor back where it was."""
        at, tokens, angle = self.at, self.tokens, self.angle
        depth = len(self.variables)
        try:
            return parse()
        except (NoParse, Refusal):
            self.at, self.tokens, self.angle = at, tokens, angle
            del self.variables[depth:]
            return None

    def lookahead(self, parse: Callable[[], object]) -> bool:
        """Does ``parse`` read from here? The cursor stays where it is either way."""
        at, tokens, angle = self.at, self.tokens, self.angle
        found = self.trial(parse) is not None
        self.at, self.tokens, self.angle = at, tokens, angle
        return found

    def listed(self, read: Callable[[], T], *closing: str) -> list[T]:
        """What ``read`` reads, comma after comma, until one of ``closing`` (left unread)."""
        items: list[T] = []
        while not self.at_(*closing):
            items.append(read())
            if not self.accept(","):
                break
        return items

    def split_shift(self) -> None:
        """Make a ``>>`` that closes two template argument lists into two ``>``."""
        token = self.peek()
        if token.is_(">>"):
            half = Token("op", ">", token.where)
            self.tokens = [*self.tokens[: self.at], half, half, *self.tokens[self.at + 1 :]]

    # -- scopes ---------------------------------------------------------------

    def declare(self, name: str) -> None:
        self.variables[-1].add(name)

    def is_variable(self, name: str) -> bool:
        return any(name in scope for scope in self.variables)

    def push(self) -> None:
        self.variables.append(set())

    def pop(self) -> None:
        self.variables.pop()

    def is_type(self, parts: list[str]) -> bool:
        if len(parts) == 1 and self.is_variable(parts[0]):
            return False
        return looks_like_type(parts, self.types)
