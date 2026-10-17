"""RooGenContext describes its direct and numerical parts after sampler initialization."""

from __future__ import annotations

from typing import Any

from .. import cout
from ..printing import kArgs, kClassName, kName, kSingleLine


def describe(context: Any) -> None:
    cout.write(_description(context, ""))


def _description(context: Any, indent: str) -> str:
    from .contexts import DirectContext, MixedContext, NumericContext, SumContext

    if isinstance(context, SumContext):
        text = f"{indent}--- RooAddGenContext ---\n" + _pdf(context, indent)
        text += f"{indent}List of component generators\n"
        return text + "".join(_description(part, indent + "    ") for part in context.parts)

    if not isinstance(context, (DirectContext, MixedContext, NumericContext)):
        return ""
    text = f"{indent} --- RooGenContext --- \n" + _pdf(context, indent)
    direct = context.names if isinstance(context, DirectContext) else getattr(context, "direct", ())
    text += f"{indent}Use PDF generator for ({','.join(sorted(direct))})\n"
    if isinstance(context, DirectContext):
        numeric = []
        method = "<none>"
    else:
        numeric = [one.GetName() for one in context.order]
        method = _sampler_name(context.sampler) if numeric else "<none>"
    return text + f"{indent}Use MC sampling generator {method} for ({','.join(numeric)})\n"


def _pdf(context: Any, indent: str) -> str:
    contents = kClassName | kName | kArgs
    return f"{indent}Using PDF " + str(context.pdf.printStream(contents, kSingleLine, indent))


def _sampler_name(sampler: Any) -> str:
    from .foam import FoamGenerator

    return "RooFoamGenerator" if isinstance(sampler, FoamGenerator) else "RooAcceptReject"
