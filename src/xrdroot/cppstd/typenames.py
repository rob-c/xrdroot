"""C++ type names, taken apart and put back together the way ROOT spells them.

A type name is a name and, if it is a template, its arguments:
``std::vector<std::pair<int,double> >``. ROOT has opinions about how such a
name is spelled - the standard library's own long names shortened
(``basic_string<char,char_traits<char>,allocator<char> >`` is ``string``), the
default template arguments left out, two closing brackets written apart
(``> >``, as the C++ of ROOT's own headers had to), and the name a
``unique_ptr`` member is read and written by (``T*``).

What is here is the parsing and the spelling; ``TClassEdit`` in the ROOT layer
is the class whose functions they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["TypeName", "parse", "render", "short_type", "name_for_io"]

#: The template arguments that are the standard library's own defaults, which a shortened
#: name leaves out: an allocator, and the character traits of a string.
DEFAULT_ARGUMENTS = ("allocator", "char_traits")

#: ``std::basic_string``'s short name, and the names it may be spelled with first.
STRING_TEMPLATES = frozenset({"basic_string", "std::basic_string", "std::__cxx11::basic_string",
                              "__cxx11::basic_string"})  # fmt: skip


@dataclass
class TypeName:
    """One type name: what it is called, its template arguments if any, and its modifiers.

    ``modifiers`` is what follows the name in C++'s declarator syntax - ``*``, ``&``,
    ``const`` - kept as written, since where it goes is the caller's business.
    """

    name: str
    arguments: list[TypeName] = field(default_factory=list)
    modifiers: str = ""

    def __str__(self) -> str:
        return render(self)


def _matching(text: str, at: int) -> int:
    """Where the ``>`` that closes the ``<`` at ``at`` is, or the end of the text."""
    depth = 0
    for where in range(at, len(text)):
        if text[where] == "<":
            depth += 1
        elif text[where] == ">":
            depth -= 1
            if depth == 0:
                return where
    return len(text) - 1


def _arguments(text: str) -> list[str]:
    """The arguments of a template, split at the commas that are not inside another one."""
    found, depth, start = [], 0, 0
    for at, letter in enumerate(text):
        if letter == "<":
            depth += 1
        elif letter == ">":
            depth -= 1
        elif letter == "," and depth == 0:
            found.append(text[start:at])
            start = at + 1
    return [*found, text[start:]]


def parse(text: str) -> TypeName:
    """``parse("std::vector<int>")``: the name and its arguments, as a :class:`TypeName`."""
    whole = str(text).strip()
    open_at = whole.find("<")
    if open_at < 0:
        return TypeName(whole)
    closed = _matching(whole, open_at)
    inside = whole[open_at + 1 : closed]
    return TypeName(
        whole[:open_at].strip(),
        [parse(argument) for argument in _arguments(inside)],
        whole[closed + 1 :].strip(),
    )


def render(node: TypeName) -> str:
    """The name as C++ writes it here: arguments after commas with no space, and a space
    between the two ``>``s of nested templates - ``vector<vector<int> >``."""
    arguments = ",".join(render(argument) for argument in node.arguments)
    text = node.name if not node.arguments else f"{node.name}<{arguments}"
    if node.arguments:
        text += " >" if arguments.endswith(">") else ">"
    return f"{text}{node.modifiers}"


def _shortened(node: TypeName, mode: int) -> TypeName:
    """A name with the standard library's long spellings shortened and its defaults dropped."""
    name = node.name
    if name in STRING_TEMPLATES:
        # ``basic_string<char,char_traits<char>,allocator<char> >`` is ``string`` and its
        # arguments - which are all defaults - go with the long name.
        dropped = mode & DROP_STD or not name.startswith("std::")
        return TypeName("string" if dropped else "std::string", [], node.modifiers)
    arguments = [_shortened(argument, mode) for argument in node.arguments]
    arguments = [argument for argument in arguments
                 if argument.name.split("::")[-1] not in DEFAULT_ARGUMENTS]  # fmt: skip
    if mode & DROP_STD:
        name = name.removeprefix("std::")
    return TypeName(name, arguments, node.modifiers)


#: ``TClassEdit::kDropStd``: the ``std::`` dropped from every name.
DROP_STD = 0x10


def short_type(text: str, mode: int = 0) -> str:
    """``TClassEdit::ShortType(name, mode)``: the type name as ROOT shortens it."""
    return render(_shortened(parse(text), int(mode)))


def _converted(node: TypeName, inside_std: bool) -> TypeName:
    """A name with the ``unique_ptr``s inside standard containers turned into pointers.

    A member declared ``std::unique_ptr<T>`` is read and written as a ``T*``, and so
    is one inside a ``std::vector`` - but one inside a template ROOT knows nothing
    about (``edm::Wrapper<...>``) keeps its own name, as ROOT's does.
    """
    arguments = [_converted(argument, inside_std and _is_standard(node)) for argument in
                 node.arguments]  # fmt: skip
    if node.name.split("::")[-1] == "unique_ptr" and node.arguments and inside_std:
        held = arguments[0]
        return TypeName(f"{held.name}{held.modifiers}*", held.arguments)
    return TypeName(node.name, arguments, node.modifiers)


def _is_standard(node: TypeName) -> bool:
    """Is this one of the standard library's containers, through which a member's own type
    reaches its ``unique_ptr``s?"""
    short = node.name.removeprefix("std::")
    return short in ("vector", "list", "deque", "map", "multimap", "set", "multiset", "pair",
                     "array", "tuple", "unordered_map", "unordered_set", "forward_list")


def name_for_io(text: str) -> str:
    """``TClassEdit::GetNameForIO(name)``: the name a member of that type is read by."""
    return render(_converted(parse(text), True))
