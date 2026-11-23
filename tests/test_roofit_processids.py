"""Native session references retain ownership, stable UIDs and file-local PID partitions."""

from __future__ import annotations

import gc
import weakref

import pytest

import xrdroot.pyroot as ROOT
from xrdroot import create, open_root
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.roofit.io.links import declarations, validate
from xrdroot.roofit.io.processids import MAX_UID, SESSION, Session
from xrdroot.roofit.io.records import named, record
from xrdroot.roofit.io.wnodes import Nodes
from xrdroot.roofit.io.wworkspace import Payload, write_workspace
from xrdroot.wupdate import update


def model():
    x = ROOT.RooRealVar("x", "x", 2., 0., 10.)
    mean = ROOT.RooRealVar("mean", "mean", 5., 0., 10.)
    sigma = ROOT.RooRealVar("sigma", "sigma", 1., .1, 3.)
    return ROOT.RooGaussian("g", "g", x, mean, sigma)


def identifiers(value):
    return set(declarations(value))


def test_repeated_owner_members_reuse_uids_without_colliding_with_other_models():
    first, second = model(), model()
    first_record = Nodes(3).node(first)
    repeated = Nodes(3).node(first)
    other = Nodes(3).node(second)
    assert identifiers(first_record) == identifiers(repeated)
    assert len(identifiers(first_record)) == 3
    assert identifiers(first_record).isdisjoint(identifiers(other))
    # Root proxy arrays are finalized with the same file-local PID as each member.
    nodes = Nodes(3)
    held = nodes.node(first)
    nodes.finalize_links(list(first._walk()))
    assert validate(held) == ["x", "mean", "sigma"]
    assert held.get("_proxyList").get("pid") == 3
    assert all(target[1]["pid"] == 3 for target in declarations(held).values())


def test_dead_owners_are_not_retained_and_their_uids_are_never_recycled():
    session = Session()
    owner = model()
    old = session.uid(owner, "Proxy", "x")
    observed = weakref.ref(owner)
    del owner
    gc.collect()
    assert observed() is None and len(session.owners) == 0
    replacement = model()
    assert session.uid(replacement, "Proxy", "x") > old
    assert session.uid(replacement, "Proxy", "x") != session.uid(replacement, "Proxy", "mean")


def test_a_full_native_uid_namespace_refuses_new_references_without_corrupting_old_ones():
    session = Session()
    owner = model()
    existing = session.uid(owner, "Proxy", "x")
    session.next_uid = MAX_UID
    assert session.uid(owner, "Proxy", "x") == existing
    with pytest.raises(UnsupportedFeatureError, match="identifier namespace"):
        session.uid(owner, "Proxy", "mean")
    assert session.next_uid == MAX_UID
    assert session.uid(owner, "Proxy", "x") == existing


def workspace(value):
    made = ROOT.RooWorkspace("w", "w")
    made.Import(value)
    return made


def test_two_workspace_keys_reuse_one_process_identity_and_resolve_their_own_proxies(tmp_path):
    made = workspace(model())
    path = tmp_path / "two-keys.root"
    with create(path) as file:
        assert write_workspace(file, "first", made) == ("ProcessID0", SESSION.title)
        assert write_workspace(file, "second", made) is None
    with open_root(path) as file:
        assert file.classnames() == {
            "first": "RooWorkspace", "ProcessID0": "TProcessID", "second": "RooWorkspace"
        }
        assert file.key("ProcessID0").title == SESSION.title
        first, second = file["first"], file["second"]
    assert first is not second
    for held in (first, second):
        assert held.pdf("g").x is held.var("x")
        assert held.pdf("g").mean is held.var("mean")
        assert held.pdf("g")._persisted_proxy_order == ["x", "mean", "sigma"]


def test_distinct_workspaces_in_one_file_share_a_pid_without_sharing_proxy_uids(tmp_path):
    first, second = workspace(model()), workspace(model())
    assert identifiers(Nodes().node(first.pdf("g"))).isdisjoint(
        identifiers(Nodes().node(second.pdf("g")))
    )
    path = tmp_path / "different-models.root"
    with create(path) as file:
        write_workspace(file, "first", first)
        write_workspace(file, "second", second)
    with open_root(path) as file:
        assert set(file.classnames()) == {"first", "second", "ProcessID0"}


def test_an_existing_foreign_process_partition_is_preserved_and_the_session_added_once(tmp_path):
    path = tmp_path / "foreign.root"
    foreign = record("TProcessID")
    title = "00000000-0000-0000-0000-000000000001"
    assert title != SESSION.title
    named(foreign, "ProcessID0", title)
    made = workspace(model())
    with create(path) as file:
        file.write("ProcessID0", Payload(foreign, title))
        assert write_workspace(file, "first", made) == ("ProcessID1", SESSION.title)
        assert write_workspace(file, "second", made) is None
    with open_root(path) as file:
        assert file.key("ProcessID0").title == title
        assert file.key("ProcessID1").title == SESSION.title
        assert set(file.classnames()) == {"ProcessID0", "ProcessID1", "first", "second"}
        assert file["first"].pdf("g")._persisted_proxy_order == ["x", "mean", "sigma"]
        assert file["second"].pdf("g")._persisted_proxy_order == ["x", "mean", "sigma"]


def test_an_unrelated_named_object_with_the_session_uuid_is_not_a_process_identity(tmp_path):
    path = tmp_path / "named.root"
    named_object = record("TNamed")
    named(named_object, "ProcessID0", SESSION.title)
    with create(path) as file:
        file.write("ProcessID0", Payload(named_object, SESSION.title))
        assert write_workspace(file, "first", workspace(model())) == ("ProcessID0", SESSION.title)
    with open_root(path) as file:
        assert file.key("ProcessID0;1").classname == "TNamed"
        assert file.key("ProcessID0;2").classname == "TProcessID"



def test_nonstandard_process_key_names_do_not_become_file_pid_indexes(tmp_path):
    session = Session()
    path = tmp_path / "nonstandard.root"
    with create(path) as file:
        for name in ("Other", "ProcessIDbad"):
            value = record("TProcessID")
            named(value, name, session.title)
            file.write(name, Payload(value, session.title))
        assert session.file_index(file) == (0, True)


def check_reopened_process_partition_keys(file):
    assert file.classnames() == {"w": "RooWorkspace", "ProcessID0": "TProcessID",
                                "ProcessID1": "TProcessID", "second": "RooWorkspace"}
    assert file.key("ProcessID0").title == file.key("ProcessID1").title == SESSION.title


def check_reopened_proxy_references(held):
    assert held.pdf("g").mean is held.var("mean")
    assert held.pdf("g")._persisted_proxy_order == ["x", "mean", "sigma"]


def test_reopening_update_installs_a_new_file_pid_for_the_same_session_and_live_model(tmp_path):
    made = workspace(model())
    path = tmp_path / "reopened.root"
    before = identifiers(Nodes().node(made.pdf("g")))
    with create(path) as file:
        assert write_workspace(file, "w", made) == ("ProcessID0", SESSION.title)
    made.var("mean").setVal(7.)
    with update(path) as file:
        assert write_workspace(file, "w", made) == ("ProcessID1", SESSION.title)
        assert write_workspace(file, "second", made) is None
    assert identifiers(Nodes().node(made.pdf("g"))) == before
    with open_root(path) as file:
        check_reopened_process_partition_keys(file)
        first, second = file["w;1"], file["w;2"]
        assert first.var("mean").getVal() == 5.
        assert second.var("mean").getVal() == 7.
        assert first is not second
        for held in (first, second, file["second"]):
            check_reopened_proxy_references(held)


def test_closed_writer_sessions_do_not_keep_files_or_their_models_alive(tmp_path):
    session = Session()
    with create(tmp_path / "weak.root") as file:
        session.remember(file, 2)
        assert session.file_index(file) == (2, False)
        observed = weakref.ref(file)
    del file
    gc.collect()
    assert observed() is None and len(session.files) == 0


def test_failed_or_empty_workspace_output_does_not_install_a_session_pid(tmp_path):
    path = tmp_path / "uninstalled.root"
    with create(path) as file:
        assert write_workspace(file, "empty", ROOT.RooWorkspace("empty", "")) is None
        bad = workspace(model())
        bad._generic["unsupported"] = object()
        with pytest.raises(UnsupportedFeatureError, match="generic or embedded"):
            write_workspace(file, "refused", bad)
        assert write_workspace(file, "valid", workspace(model())) == ("ProcessID0", SESSION.title)
        assert write_workspace(file, "again", workspace(model())) is None
    with open_root(path) as file:
        assert set(file.classnames()) == {"empty", "valid", "again", "ProcessID0"}
