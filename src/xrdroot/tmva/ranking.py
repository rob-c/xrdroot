"""``TMVA::Ranking::Print``: a ranking of the variables, best first, as TMVA prints it."""

from __future__ import annotations

from .log import Logger

__all__ = ["print_ranking"]


def print_ranking(source: str, title: str, entries: list[tuple[str, float]]) -> None:
    """The table of ``entries`` - a variable and its value each - under ``title``, best first.

    The order is TMVA's bubble sort's, which keeps equal values in the order
    they were added.
    """
    ranked = sorted(entries, key=lambda entry: -float(entry[1]))
    width = max((len(name) for name, _ in ranked), default=0)
    rule = "-" * (width + 15 + len(title))
    log = Logger(source)
    log.header("Ranking result (top variable is best ranked)")
    log.info(rule)
    log.info(f"{'Rank : ':<5}{'Variable '.ljust(width)} : {title}")
    log.info(rule)
    for rank, (name, value) in enumerate(ranked, start=1):
        log.info(f"{rank:4d} : {name.ljust(max(width, 9))} : {float(value):3.3e}")
    log.info(rule)
