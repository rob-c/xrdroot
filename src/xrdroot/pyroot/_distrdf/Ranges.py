"""How the data is cut into the ranges of entries the tasks process.

Empty entries are cut evenly (:func:`get_balanced_ranges`). Trees are cut in
two steps, as ROOT cuts them. Where the tasks are made, only the files are
known, not what is in them, so the files are shared out as fractions -
:func:`get_percentage_ranges` gives each task a share of the list of files,
from a fraction of one file to a fraction of another. Each task then opens
its files and turns its fractions into entries, moved to the nearest
cluster boundaries (:func:`get_clustered_range_from_percs`), so that no
cluster is read by two tasks; a task whose share falls within one cluster
has nothing to do. A task also says which trees it saw with how many
entries, so the client can check that every entry was processed once.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from itertools import accumulate
from math import floor
from typing import TYPE_CHECKING, Any

from ...rdf.distributed import clusters_and_entries

if TYPE_CHECKING:
    from ._graph_cache import ExecutionIdentifier

__all__ = [
    "DataRange",
    "EmptySourceRange",
    "TreeRangePerc",
    "TreeRange",
    "TaskTreeEntries",
    "get_balanced_ranges",
    "get_clusters_and_entries",
    "get_percentage_ranges",
    "get_entryrange_at_cluster_boundaries",
    "get_clustered_range_from_percs",
]


@dataclass
class DataRange:
    """A task's range: which run it belongs to, and its number among the run's ranges."""

    exec_id: ExecutionIdentifier
    id: int


@dataclass
class EmptySourceRange(DataRange):
    """Empty entries from ``start`` up to ``end``."""

    start: int
    end: int


@dataclass
class TreeRangePerc(DataRange):
    """A task's share of the files, before they are opened: from ``first_tree_start_perc`` of
    the file at ``first_file_idx`` to ``last_tree_end_perc`` of the one before
    ``last_file_idx``. With friends, every file of the data is listed, so that the friends'
    entries line up with the whole chain's."""

    treenames: list[str]
    filenames: list[str]
    first_file_idx: int
    last_file_idx: int
    first_tree_start_perc: float
    last_tree_end_perc: float
    friendinfo: Any
    samples: list[Any] | None


@dataclass
class TreeRange(DataRange):
    """A task's entries, ``globalstart`` up to ``globalend`` of the chain of its trees."""

    treenames: list[str]
    filenames: list[str]
    globalstart: int
    globalend: int
    friendinfo: Any
    samples: list[Any] | None


@dataclass
class TaskTreeEntries:
    """The entries a task processed, and the entries of each tree it opened, by
    ``file/tree``: added up over the tasks, they say whether every entry was read once."""

    processed_entries: int = 0
    trees_with_entries: dict[str, int] = field(default_factory=dict)


def get_balanced_ranges(
    nentries: int, npartitions: int, exec_id: Any
) -> list[EmptySourceRange]:
    """``nentries`` empty entries cut into ``npartitions`` ranges, the first ones an entry
    longer when they do not divide evenly - and no empty range."""
    size, remainder = divmod(nentries, npartitions)
    ranges: list[EmptySourceRange] = []
    start = 0
    while start < nentries:
        end = start + size + (1 if len(ranges) < remainder else 0)
        ranges.append(EmptySourceRange(exec_id, len(ranges), start, end))
        start = end
    return ranges


def get_clusters_and_entries(
    rdf_uuid: Any, datasetname: str, filename: str
) -> tuple[list[int], int]:
    """Where the clusters of a tree or RNTuple begin, with where the last ends; its entries."""
    return clusters_and_entries(datasetname, filename)


def _shares(nfiles: int, npartitions: int) -> tuple[list[int], list[int], list[float], list[float]]:
    """Each task's first file, the file after its last, and the fractions of those it starts
    and ends at - a task ending at a file's end having all of that file."""
    percentages = [nfiles / npartitions * at for at in range(npartitions + 1)]
    files = [floor(percentage) for percentage in percentages]
    within = [percentage - file for percentage, file in zip(percentages, files, strict=True)]
    starts = files[:-1]
    ends = [
        file + 1 if perc > 0 else file for file, perc in zip(files[1:], within[1:], strict=True)
    ]
    return starts, ends, within[:-1], [perc if perc > 0 else 1 for perc in within[1:]]


def _sample(sampleMap: dict[str, Any] | None, filename: str, treename: str) -> Any:
    return None if sampleMap is None else sampleMap.get(f"{filename}/{treename}")


def get_percentage_ranges(
    treenames: list[str],
    filenames: list[str],
    npartitions: int,
    friendinfo: Any,
    exec_id: Any,
    sampleMap: dict[str, Any] | None,
) -> list[TreeRangePerc]:
    """The trees in these files shared out between ``npartitions`` tasks, by fraction."""
    starts, ends, first_percs, last_percs = _shares(len(filenames), npartitions)
    samples = [
        _sample(sampleMap, file, tree) for file, tree in zip(filenames, treenames, strict=True)
    ]
    ranges = []
    for rangeid in range(npartitions):
        if friendinfo:
            trees, files, first, last = treenames, filenames, starts[rangeid], ends[rangeid]
            mine = samples
        else:
            span = slice(starts[rangeid], ends[rangeid])
            trees, files, first, last = treenames[span], filenames[span], 0, len(filenames[span])
            mine = samples[span]
        ranges.append(TreeRangePerc(
            exec_id, rangeid, trees, files, first, last, first_percs[rangeid],
            last_percs[rangeid], friendinfo, None if sampleMap is None else mine,
        ))  # fmt: skip
    return ranges


def get_entryrange_at_cluster_boundaries(
    percstart: float, percend: float, entries: int, clusters: list[int]
) -> tuple[int, int, int]:
    """The entries of a tree from one fraction of it to another, each moved to the cluster
    boundary at or after it; and the tree's entries again when no cluster is left between
    them - entries this task will not process after all - else zero."""
    startcluster = bisect_left(clusters, int(percstart * entries))
    endcluster = bisect_left(clusters, int(percend * entries))
    discarded = entries if startcluster == endcluster else 0
    return clusters[startcluster], clusters[endcluster], discarded


def get_clustered_range_from_percs(
    percrange: TreeRangePerc,
) -> tuple[TreeRange | None, TaskTreeEntries]:
    """A task's share of the files as entries of the chain of its trees, at cluster
    boundaries - ``None`` when that leaves nothing - and the entries of the trees it saw."""
    first, last = percrange.first_file_idx, percrange.last_file_idx
    found = [
        get_clusters_and_entries(percrange.exec_id.rdf_uuid, tree, file)
        for tree, file in zip(percrange.treenames, percrange.filenames, strict=True)
    ]
    clusters = [each[0] for each in found]
    entries = [each[1] for each in found]
    offsets = list(accumulate([0, *entries[:-1]]))
    mine = slice(first, last)
    trees_with_entries = {
        f"{file}/{tree}": count
        for file, tree, count in zip(
            percrange.filenames[mine], percrange.treenames[mine], entries[mine], strict=True
        )
    }
    if last - first == 1:
        start, end, discarded = get_entryrange_at_cluster_boundaries(
            percrange.first_tree_start_perc, percrange.last_tree_end_perc,
            entries[first], clusters[first],
        )  # fmt: skip
    else:
        start, _, at_first = get_entryrange_at_cluster_boundaries(
            percrange.first_tree_start_perc, 1, entries[first], clusters[first]
        )
        _, end, at_last = get_entryrange_at_cluster_boundaries(
            0, percrange.last_tree_end_perc, entries[last - 1], clusters[last - 1]
        )
        discarded = at_first + at_last
    if sum(entries[mine]) - discarded == 0:
        return None, TaskTreeEntries(0, trees_with_entries)
    globalstart = offsets[first] + start
    globalend = offsets[last - 1] + end
    treerange = TreeRange(
        percrange.exec_id, percrange.id, percrange.treenames, percrange.filenames,
        globalstart, globalend, percrange.friendinfo, percrange.samples,
    )  # fmt: skip
    return treerange, TaskTreeEntries(globalend - globalstart, trees_with_entries)
