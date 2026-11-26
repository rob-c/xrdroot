"""``sizeof``: the bytes a type takes on LP64 macOS on ARM64, as clang and libc++ lay it out.

A fundamental type's size is its width; a pointer's, a reference's to
anything, eight bytes. An array is its elements end to end. A class of the
program's is laid out as the Itanium C++ ABI does: a pointer to its virtual
table first when it is polymorphic and inherits none, then its bases, then
its data members in order, each at the next offset its alignment allows,
and the whole rounded up to the largest alignment; an empty class is one
byte, and an empty base none. The standard library's classes take what
libc++'s do, and a library's classes what its dialect says they take
(:attr:`~.dialect.Dialect.layouts`) - as a base too. Anything else - a class
nothing gives the layout of, a member whose size is not known - has no size
here, and ``sizeof`` of it is refused.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .ctype import SIZES, CType
from .dialect import active
from .nodes import Binary, EnumDecl, Literal, Name, Unary, VarDecl

__all__ = ["layout", "member_offsets", "size_of"]

#: A size and an alignment, in bytes.
Layout = tuple[int, int]

#: The pointer's size, and its alignment.
POINTER: Layout = (8, 8)

#: libc++'s classes of a fixed size, whatever their template arguments: each its size, and
#: the alignment of the pointers it holds.
STANDARD: dict[str, Layout] = {
    **dict.fromkeys(
        """std::string std::vector std::map std::set std::multimap std::multiset std::list
        std::basic_string""".split(),
        (24, 8),
    ),
    **dict.fromkeys(["std::string_view", "std::shared_ptr", "std::weak_ptr"], (16, 8)),
    **dict.fromkeys(["std::unique_ptr", "std::nullptr_t", "nullptr_t"], POINTER),
    **dict.fromkeys(
        """std::unordered_map std::unordered_set std::unordered_multimap
        std::unordered_multiset""".split(),
        (40, 8),
    ),
    "std::deque": (48, 8),
    "std::function": (32, 8),
    "std::byte": (1, 1),
}

#: The standard's class templates that hold one of their arguments and what says which - an
#: ``unsigned int`` for libc++'s ``std::variant``, a ``bool`` for ``std::optional``.
TAGGED: dict[str, Layout] = {"std::variant": (4, 4), "std::optional": (1, 1)}

#: The standard's class templates that hold their argument a number of times, and how many.
REPEATS = {"std::complex": 2, "std::atomic": 1}

#: The standard's class templates whose size is their arguments laid out as a struct's members.
AGGREGATES = frozenset({"std::pair", "std::tuple"})


def size_of(ctype: CType, program: Any) -> int | None:
    """``sizeof`` of ``ctype``, or ``None`` when this translator does not know it."""
    found = layout(ctype, program)
    return found[0] if found is not None else None


def layout(ctype: CType, program: Any) -> Layout | None:
    """The size and alignment of ``ctype``: a reference's are those of what it refers to."""
    if ctype.pointer and not ctype.dims:
        return POINTER
    element = _element(ctype, program)
    if element is None or not ctype.dims:
        return element
    count = 1
    for dim in ctype.dims:
        value = constant(dim, program)
        if value is None:
            return None
        count *= value
    return element[0] * count, element[1]


def _element(ctype: CType, program: Any) -> Layout | None:
    """The layout of one element of ``ctype`` - of the pointer, if its elements are pointers."""
    if ctype.pointer:
        return POINTER
    if ctype.enum is not None:
        return _enum(program.enums.get(ctype.enum), program)
    size = SIZES.get(ctype.name)
    if size is not None:
        return size, size
    if ctype.name in program.classes:
        return _class(program.classes[ctype.name], program)
    library = _library(ctype)
    return library[:2] if library is not None else _standard(ctype, program)


def _library(ctype: CType) -> tuple[int, int, int | None, bool] | None:
    """A library's class, as its dialect lays it out: by its name, or its name with its
    template arguments as the library spells them."""
    layouts = active().layouts
    found = layouts.get(ctype.name)
    if found is None and ctype.args:
        spelled = ",".join(str(getattr(arg, "name", arg)) for arg in ctype.args)
        found = layouts.get(f"{ctype.name}<{spelled}>")
    return found


def _enum(decl: EnumDecl | None, program: Any) -> Layout | None:
    """An enum is the integer type under it: ``int`` unless it is written."""
    if decl is None or decl.underlying is None:
        return 4, 4
    return layout(decl.underlying, program)


def _standard(ctype: CType, program: Any) -> Layout | None:
    """A class of the standard library's, as libc++ on ARM64 lays it out."""
    fixed = STANDARD.get(ctype.name)
    if fixed is not None:
        return fixed
    kinds = [arg for arg in ctype.args if isinstance(arg, CType)]
    if kinds and len(kinds) == len(ctype.args):
        return _composed(ctype.name, kinds, program)
    return _sized_template(ctype, program)


def _composed(name: str, kinds: list[CType], program: Any) -> Layout | None:
    """A standard class template of types, laid out from theirs."""
    if name in AGGREGATES:
        return _struct(kinds, program)
    tag = TAGGED.get(name)
    if tag is not None:
        return _tagged(kinds, tag, program)
    count = REPEATS.get(name)
    part = layout(kinds[0], program) if count is not None else None
    return (count * part[0], part[1]) if count is not None and part is not None else None


def _sized_template(ctype: CType, program: Any) -> Layout | None:
    """``std::array<T, N>``: ``N`` of ``T``; ``std::bitset<N>``: ``N`` bits in 64-bit words."""
    if ctype.name == "std::array" and len(ctype.args) == 2 and isinstance(ctype.args[0], CType):
        count = constant(ctype.args[1], program)
        part = layout(ctype.args[0], program)
        return (part[0] * count, part[1]) if part is not None and count is not None else None
    if ctype.name == "std::bitset" and len(ctype.args) == 1:
        bits = constant(ctype.args[0], program)
        return (8 * max(1, -(-bits // 64)), 8) if bits is not None else None
    return None


#: The classes whose placement is being worked out right now: a class that reaches itself - deriving
#: from itself, or holding one of itself, neither of which a C++ compiler accepts but a macro may
#: still declare - would otherwise ask for its own layout for ever.
_LAID_OUT: set[str] = set()


def _class(info: Any, program: Any) -> Layout | None:
    """A class of the program's, as the Itanium ABI lays it out: one byte when empty."""
    found = _extent(info, program)
    if found is None:
        return None
    end, align = found
    if info.decl.aligned is not None:
        wanted = _alignment(info.decl.aligned, program)
        if wanted is None:
            return None
        align = max(align, wanted)
    return (_aligned(max(end, 1), align), align)


def _alignment(given: Any, program: Any) -> int | None:
    """What ``alignas(...)`` asks for: a type's alignment, or a number of bytes."""
    if isinstance(given, CType):
        found = layout(given, program)
        return found[1] if found is not None else None
    return constant(given, program)


def _extent(info: Any, program: Any) -> Layout | None:
    """Where a class's last member ends - its size before the padding that rounds it to its
    alignment, which a derived class may reuse - and its alignment.

    This is where a class's layout is entered, so this is where a class that reaches itself stops:
    ``class A : public A`` and ``struct Self { Self inner; }`` are declarations no C++ compiler
    accepts, but a macro may still write one, and following it would never end.
    """
    if info.name in _LAID_OUT:
        return None
    _LAID_OUT.add(info.name)
    try:
        return _measured(info, program)
    finally:
        _LAID_OUT.discard(info.name)


def _measured(info: Any, program: Any) -> Layout | None:
    """A class's extent, one level in - :func:`_extent` keeps the walk from reaching itself."""
    if info.decl.kind == "union":
        return _union([_member_type(var.ctype) for var in info.fields.values()], program)
    start = _start(info, program)
    if start is None:
        return None
    fields = [_member_type(var.ctype) for var in info.fields.values()]
    return _laid(fields, program, start[0], start[1])


def _start(info: Any, program: Any) -> Layout | None:
    """Where a class's own members begin - after its virtual table's pointer, when it has one of its
    own, and after its bases - and the alignment they give it."""
    found = _placement(info, program)
    return None if found is None else found[:2]


#: Where a class's own members begin, their alignment, and where each of its bases sits.
Placement = tuple[int, int, tuple[tuple[CType, int], ...]]


def _placement(info: Any, program: Any) -> Placement | None:
    """The one walk over a class's virtual table's pointer and its bases, which both ``sizeof``
    (:func:`_extent`) and the members' offsets (:func:`member_offsets`) read."""
    decl = info.decl
    offset, align, bases = 0, 1, []
    if _polymorphic(info, program) and not any(
        _polymorphic_base(base.ctype, program) for base in decl.bases
    ):
        offset, align = POINTER
    for base in decl.bases:
        found = _base(base.ctype, program)
        if found is None:
            return None
        at = _aligned(offset, found[1])
        bases.append((base.ctype, at))
        offset = at + found[0]
        align = max(align, found[1])
    return offset, align, tuple(bases)


def member_offsets(info: Any, program: Any) -> dict[str, int] | None:
    """Each data member's byte offset by name - the numbers a dictionary hands out through
    ``TDataMember::GetOffset`` - or ``None`` when this kit does not work the class's layout out.

    This is the one authority for those numbers: the translation writes them on the class it emits
    (``_cxx_offsets_``), so the reflection layer and ``cppstd.cells.byte_offset`` read the same map
    and a macro can neither be told one offset and write at another.

    A base's members are in it at the offsets they sit at in a class of this one's, because a macro
    that walks a class walks its bases too - roottest's ``datamodelevolution`` recurses with
    ``((char *)obj) + cl->GetBaseClassOffset(baseCl)`` and the base's own ``TClass``, so the object
    is this one while the member asked for is the base's. A union's members all start where it does;
    a static member has no offset and is not in ``info.fields``.
    """
    found: dict[str, int] = {}
    if info.decl.kind == "union":
        return dict.fromkeys(info.fields, 0)
    placed = _placement(info, program)
    if placed is None:
        return None
    offset = placed[0]
    for ctype, at in placed[2]:
        inherited = _base_members(ctype, program, at)
        if inherited is None:
            return None
        found.update(inherited)
    for name, var in info.fields.items():
        laid = layout(_member_type(var.ctype), program)
        if laid is None:
            return None
        size, own = laid
        offset = _aligned(offset, own)
        found[name] = offset
        offset += size
    return found


def _base_members(ctype: CType, program: Any, at: int) -> dict[str, int] | None:
    """A base's members, moved to where the base sits in the class derived from it: none for a base
    this kit has no layout of, whose members then answer no offset at all."""
    info = program.classes.get(ctype.name)
    if info is None:
        return None
    own = member_offsets(info, program)
    return None if own is None else {name: at + offset for name, offset in own.items()}


def _member_type(ctype: CType) -> CType:
    """A reference member takes a pointer's room."""
    return CType("void", pointer=1) if ctype.reference else ctype.value()


def _base(ctype: CType, program: Any) -> Layout | None:
    """The room a base takes in a class derived from it: none when it is empty, all of it when
    it is plain old data, and up to its last member otherwise - the Itanium ABI lets a
    derived class's members use the padding after it."""
    info = program.classes.get(ctype.name)
    if info is None:
        library = _library(ctype)
        if library is None or library[2] is None:
            return None
        return library[2], library[1]
    extent = _extent(info, program)
    if extent is None or not extent[0] or not _plain(info, program):
        return extent
    return _class(info, program)


def _plain(info: Any, program: Any) -> bool:
    """Is a class plain old data, as clang has it for layout - trivial, with no virtual table
    and no constructor, destructor or assignment of its own, and of standard layout: its
    members all public or all not, in one class of its bases and it, and plain themselves?"""
    own = (info.name, "~" + info.name, "operator=")
    if _polymorphic(info, program) or any(name in own for name in info.methods):
        return False
    if len({var.hidden for var in info.fields.values()}) > 1:
        return False
    return _plain_bases(info, program) and all(
        _plain_member(var, program) for var in info.fields.values()
    )


def _plain_bases(info: Any, program: Any) -> bool:
    """Bases of plain old data, none of which has members when the class has its own."""
    bases = [program.classes.get(base.ctype.name) for base in info.decl.bases]
    if not all(base is not None and _plain(base, program) for base in bases):
        return False
    return not (info.fields and any(base.fields for base in bases))


def _plain_member(var: VarDecl, program: Any) -> bool:
    """A member that is a number, a pointer, or of a class that is plain old data."""
    inner = program.classes.get(var.ctype.name)
    plain = var.ctype.pointer > 0 or inner is None or _plain(inner, program)
    return plain and not var.ctype.reference


def _laid(fields: list[CType], program: Any, offset: int, align: int) -> Layout | None:
    """Members laid out one after the other from ``offset``, each at its alignment: where the
    last ends, and the strictest alignment."""
    for ctype in fields:
        found = layout(ctype, program)
        if found is None:
            return None
        size, own = found
        offset = _aligned(offset, own) + size
        align = max(align, own)
    return offset, align


def _struct(fields: list[CType], program: Any) -> Layout | None:
    """A struct of ``fields``, its size rounded up to its alignment."""
    found = _laid(fields, program, 0, 1)
    if found is None:
        return None
    return _aligned(max(found[0], 1), found[1]), found[1]


def _tagged(kinds: list[CType], tag: Layout, program: Any) -> Layout | None:
    """``std::variant<A, B>`` and ``std::optional<T>`` as libc++ lays them out: room for the
    largest alternative, then the ``tag`` that tells which is held."""
    found = _union(kinds, program)
    if found is None:
        return None
    align = max(found[1], tag[1])
    return _aligned(_aligned(found[0], tag[1]) + tag[0], align), align


def _union(fields: list[CType], program: Any) -> Layout | None:
    """A union reaches as far as its largest member, at the strictest member's alignment."""
    found = [layout(ctype, program) for ctype in fields]
    sizes = [item for item in found if item is not None]
    if len(sizes) != len(found):
        return None
    return max((item[0] for item in sizes), default=0), max((i[1] for i in sizes), default=1)


def _polymorphic(info: Any, program: Any, seen: frozenset[str] = frozenset()) -> bool:
    """Has the class, or a class of the program's it derives from, a virtual function?

    ``seen`` is the lineage being asked, so a class that reaches itself - ``class A : public A`` -
    stops there instead of asking for ever."""
    if info.name in seen:
        return False
    if any(func.virtual for funcs in info.methods.values() for func in funcs):
        return True
    return any(
        _polymorphic_base(base.ctype, program, seen | {info.name}) for base in info.decl.bases
    )


def _polymorphic_base(ctype: CType, program: Any, seen: frozenset[str] = frozenset()) -> bool:
    info = program.classes.get(ctype.name)
    if info is None:
        library = _library(ctype)
        return library is not None and library[3]
    return _polymorphic(info, program, seen)


def _aligned(offset: int, align: int) -> int:
    return -(-offset // align) * align


def constant(value: Any, program: Any) -> int | None:
    """An array's size or a template's value argument, when it is an integer constant: a
    number, an enumerator or ``const`` global given a number, or arithmetic on them."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    rule = CONSTANTS.get(type(value))
    return rule(value, program) if rule is not None else None


def _literal_constant(value: Literal, program: Any) -> int | None:
    return int(value.value) if value.kind in ("int", "char") else None


def _negated(value: Unary, program: Any) -> int | None:
    inner = constant(value.operand, program) if value.op == "-" else None
    return -inner if inner is not None else None


def _named_constant(value: Name, program: Any) -> int | None:
    """A ``const`` integral global given a constant."""
    found = program.globals.get(value.last) if program is not None else None
    if len(value.parts) > 1 or found is None:
        return None
    return constant(found.init, program) if found.ctype.const and found.ctype.integral else None


def _operated(value: Binary, program: Any) -> int | None:
    operation = OPERATIONS.get(value.op)
    left, right = constant(value.left, program), constant(value.right, program)
    if operation is None or left is None or right is None or (value.op == "/" and right == 0):
        return None
    return int(operation(left, right))


#: The arithmetic a constant array size may be written with.
OPERATIONS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1),
}


#: How each kind of expression gives the integer constant it is, if it is one.
CONSTANTS: dict[type, Callable[[Any, Any], int | None]] = {
    Literal: _literal_constant,
    Name: _named_constant,
    Binary: _operated,
    Unary: _negated,
}
