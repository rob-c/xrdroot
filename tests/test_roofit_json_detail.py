"""``RooFit::Detail``'s JSON document: the nodes ROOT's HS3 writers build and walk.

The shape and the member names are ROOT 6.40's ``RooFit/Detail/JSONInterface.h``; what the
test of that header asks of them is a map of maps of a ``TVectorD``, stored a node at a time.
"""

from __future__ import annotations

import io

import pytest

from xrdroot.cppstd.containers import std
from xrdroot.roofit.detail import JSONNode, JSONTree
from xrdroot.roofit.matrix import TVectorD


def test_a_node_made_with_no_value_is_an_empty_object():
    node = JSONNode()
    assert node.is_map() and not node.is_seq() and not node.is_null()
    assert node.num_children() == 0 and node.key() == ""


def test_a_subscript_makes_the_child_it_names_and_the_same_one_afterwards():
    root = JSONTree.create().rootnode()
    root["a"]["b"] << 1
    assert root["a"]["b"] is root["a"]["b"]
    assert root.has_child("a") and not root.has_child("missing")
    assert root.writeJSON() == '{\n   "a": {\n      "b": 1\n   }\n}'


def test_a_bool_a_number_and_a_string_are_stored_as_themselves():
    root = JSONTree.create().rootnode()
    root["yes"] << True
    root["no"] << False
    root["n"] << 3
    root["x"] << 1.5
    root["s"] << "text"
    assert root.writeJSON() == (
        '{\n   "yes": true,\n   "no": false,\n   "n": 3,\n   "x": 1.5,\n   "s": "text"\n}'
    )
    assert (root["yes"].val(), root["n"].val(), root["s"].val()) == ("true", "3", "text")


def test_a_map_is_stored_as_an_object_of_its_pairs_a_key_with_no_order():
    root = JSONTree.create().rootnode()
    root["map"] << std.map["std::string", "TVectorD"]([("vec", TVectorD(3))])
    root["unordered_map"] << {"vec": TVectorD([1.0, 2.0])}
    assert root["map"].is_map() and root["map"]["vec"].is_seq()
    assert [root["map"]["vec"].child(at).val_double() for at in range(3)] == [0.0, 0.0, 0.0]
    assert [each.val_double() for each in root["unordered_map"]["vec"]] == [1.0, 2.0]


def test_a_document_read_from_json_is_a_tree_of_the_same_nodes():
    tree = JSONTree.create('{"a": [1, {"b": 2}], "c": null}')
    root = tree.rootnode()
    assert root["a"].is_seq() and root["a"].child(1)["b"].val_int() == 2
    assert root["c"].is_null()
    assert root.writeJSON() == (
        '{\n   "a": [\n      1,\n      {\n         "b": 2\n      }\n   ],\n   "c": null\n}'
    )


def test_an_array_is_made_by_set_seq_and_appended_to_a_child_at_a_time():
    node = JSONNode().set_seq()
    node.append_child() << "first"
    node.append_child() << 2
    assert node.is_seq() and node.num_children() == 2 and len(node) == 2
    assert [each.val() for each in node] == ["first", "2"]


def test_a_node_that_is_not_the_kind_asked_for_says_so_rather_than_guessing():
    scalar = JSONNode()
    with pytest.raises(TypeError, match="is not an array"):
        scalar.append_child()
    with pytest.raises(TypeError, match="is not an array"):
        scalar.child(0)
    array = JSONNode().set_seq()
    with pytest.raises(TypeError, match="is not an object"):
        array["key"]
    with pytest.raises(TypeError, match="asked of a scalar"):
        array.val()
    with pytest.raises(TypeError, match="holds a string"):
        scalar << object()


def test_a_node_is_emptied_keeping_the_kind_it_is_and_a_null_is_its_own_kind():
    node = JSONNode()
    node["a"] << 1
    node.clear()
    assert node.is_map() and node.num_children() == 0
    scalar = JSONNode("s", 1)
    scalar.clear()
    assert scalar.val() == "" and not scalar.is_container()
    assert JSONNode().set_null().is_null()
    assert list(JSONNode().set_null()) == []


def test_a_scalar_says_whether_it_is_true_and_a_value_given_at_making_is_stored():
    root = JSONNode("root", {"a": 1, "yes": True})
    assert root["a"].val_int() == 1 and root["yes"].val_bool() is True
    assert JSONNode("other", 0).val_bool() is False


def test_write_json_writes_into_a_stream_as_roots_does():
    node = JSONNode()
    node << {"a": 1}
    stream = io.StringIO()
    assert node.writeJSON(stream) == stream.getvalue() == '{\n   "a": 1\n}'
    with pytest.raises(TypeError, match="takes a stream"):
        node.writeJSON(3)


def test_a_tvector_built_from_a_size_is_that_many_zeros_and_from_a_list_its_items():
    """``TVectorD(3)`` is ROOT's ``TVectorD(Int_t n)``: three zeros, not the vector ``[3]``."""
    assert TVectorD(3).values.tolist() == [0.0, 0.0, 0.0]
    assert TVectorD([1.0, 2.0]).values.tolist() == [1.0, 2.0]
    assert TVectorD().GetNrows() == 0


def test_the_namespace_holds_the_two_classes_and_says_what_it_is():
    from xrdroot.pyroot.roofit.commands import RooFit

    assert RooFit.Detail.JSONNode is JSONNode and RooFit.Detail.JSONTree is JSONTree
    assert repr(JSONNode("root", {"a": 1})).startswith("JSONNode(")
    assert repr(JSONTree()).startswith("JSONTree(")
