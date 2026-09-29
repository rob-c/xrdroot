"""TMVA's decision trees as arrays: walked, written as TMVA's XML, and read back from it."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from xrdroot.tmva.trees import Packed, Tree, forest_responses, read_tree
from xrdroot.tmva.xmlfile import Node


def stump(cut: float = 0.5, fisher: dict | None = None) -> Tree:
    """A root cutting at ``cut`` and its two leaves: background on the left, signal on the right."""
    return Tree(
        var=np.array([0, -1, -1]),
        cut=np.array([cut, 0.0, 0.0]),
        ctype=np.array([1, 1, 1]),
        left=np.array([1, -1, -1]),
        right=np.array([2, -1, -1]),
        response=np.array([0.0, -1.0, 1.0]),
        purity=np.array([0.5, 0.2, 0.8]),
        ntype=np.array([0, -1, 1]),
        depth=np.array([0, 1, 1]),
        fisher=fisher or {},
    )


def element(tree: Tree) -> ET.Element:
    """The tree as TMVA's ``<BinaryTree>``, parsed back as the weight file's reader parses it."""
    parent = Node("Weights")
    tree.add_xml(parent, 0.75, 3, {})
    return ET.fromstring("\n".join(parent.lines()))[0]


def test_a_tree_sends_each_event_to_the_side_of_the_cut_it_is_on():
    values = np.array([[0.0, 9.0], [1.0, -9.0]])
    assert list(stump().leaves(values)) == [1, 2]
    assert list(stump().respond(values, True)) == [-1.0, 1.0]
    assert list(stump().respond(values, False, "purity")) == [0.2, 0.8]
    assert list(stump().respond(values, False)) == [-1.0, 1.0]


def test_a_node_cutting_on_a_fisher_discriminant_projects_every_event_onto_its_coefficients():
    # The discriminant 2*x0 - x1 + 0.5: the first event's is 0.5 - 9, the second's 2 + 9 + 0.5.
    tree = stump(0.0, {0: np.array([2.0, -1.0, 0.5])})
    assert list(tree.leaves(np.array([[0.0, 9.0], [1.0, -9.0]]))) == [1, 2]


def test_a_fisher_cut_below_the_root_is_used_only_by_the_events_that_reach_it():
    tree = Tree(
        var=np.array([0, 0, -1, -1, -1]),
        cut=np.array([0.5, 0.0, 0.0, 0.0, 0.0]),
        ctype=np.ones(5, dtype=int),
        left=np.array([1, 3, -1, -1, -1]),
        right=np.array([2, 4, -1, -1, -1]),
        response=np.array([0.0, 0.0, 2.0, 3.0, 4.0]),
        purity=np.zeros(5),
        ntype=np.zeros(5, dtype=int),
        depth=np.array([0, 1, 1, 2, 2]),
        fisher={1: np.array([0.0, 1.0, 0.0])},
    )
    # The node below the root on the left cuts on the second variable, as its coefficients say.
    values = np.array([[0.0, -1.0], [0.0, 1.0], [1.0, 0.0]])
    assert list(tree.respond(values, False)) == [3.0, 4.0, 2.0]


def test_a_tree_written_as_xml_is_read_back_node_for_node_with_its_fisher_coefficients():
    tree = stump(0.25, {0: np.array([2.0, -1.0, 0.5])})
    found = element(tree)
    assert found.get("boostWeight").startswith("7.5") and found.get("itree") == "3"
    back = read_tree(found)
    assert list(back.left) == [1, -1, -1] and list(back.right) == [2, -1, -1]
    assert list(back.fisher[0]) == [2.0, -1.0, 0.5] and back.cut[0] == 0.25
    assert list(back.ntype) == [0, -1, 1]


def test_a_tree_of_tmvas_older_files_takes_its_purity_from_the_signal_and_background_it_counted():
    text = (
        '<BinaryTree><Node pos="s" IVar="0" Cut="0.5" cType="1" nS="3" nB="1">'
        '<Other/><Node pos="l" IVar="-1" nS="1" nB="3" nType="-1"/>'
        '<Node pos="r" IVar="-1" nS="4" nB="0" nType="1"/></Node></BinaryTree>'
    )
    found = read_tree(ET.fromstring(text))
    assert list(found.purity) == [0.75, 0.25, 1.0]
    assert list(found.response) == [-99.0, -99.0, -99.0]


def test_a_packed_forest_walks_all_its_trees_at_once_as_each_walks_alone():
    trees = [stump(0.5), stump(-0.5)]
    values = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
    together = Packed(trees).responses(values, "purity")
    alone = np.array([tree.respond(values, False, "purity") for tree in trees])
    assert np.array_equal(together, alone)
    assert Packed(trees).responses(np.zeros((0, 2))).shape == (2, 0)


def test_a_forest_with_a_fisher_cut_is_walked_a_tree_at_a_time():
    trees = [stump(0.0, {0: np.array([1.0, 1.0, 0.0])}), stump(0.5)]
    values = np.array([[0.0, 1.0], [-1.0, -2.0]])
    assert np.array_equal(Packed(trees).responses(values, "ntype"), [[1, -1], [-1, -1]])


def test_the_responses_of_a_forest_are_its_packed_responses_and_nothing_for_no_trees():
    values = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.array_equal(forest_responses([stump()], values), [[-1.0, 1.0]])
    assert forest_responses([], values).shape == (0, 2)
