"""``TFile``, ``TDirectoryFile`` and ``TKey``: files read, made, added to and listed as ROOT's."""

from __future__ import annotations

import pathlib

import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.pyroot.core import files

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_file_made_keeps_what_is_booked_after_it_and_writes_it(capsys):
    out = ROOT.TFile("out.root", "RECREATE", "the title")
    expect(
        (out.GetPath(), "out.root:/"),
        (ROOT.gDirectory, out),
        (out.GetOption(), "CREATE"),
        (bool(out.IsWritable()), True),
        (bool(out.IsOpen()), True),
        (bool(not out.IsZombie()), True),
        (bool(bool(out)), True),
    )
    h = ROOT.TH1F("hpx", "px", 10, 0, 1)
    h.Fill(0.5)
    sub = out.mkdir("sub", "the sub")
    expect(
        (sub.GetPath(), "out.root:/sub"),
        (ROOT.gDirectory, out),
    )
    sub.cd()
    inner = ROOT.TH1D("inner", "in", 2, 0, 1)
    expect(
        (bool(inner.GetDirectory() is sub), True),
        (bool(sub.GetFile() is out), True),
    )
    out.cd()
    expect(
        (out.Write(), 2),
        (h.Write(), 1),
    )
    out.ls()
    lines = capsys.readouterr().out.splitlines()
    expect(
        (lines[:2], ["TFile**\t\tout.root\tthe title", " TFile*\t\tout.root\tthe title"]),
        (bool(lines[2].startswith("  OBJ: TH1F\thpx\tpx : 0 at: 0x")), True),
        (lines[3], "  TDirectoryFile*\t\tsub\tthe sub"),
        (bool(lines[4].startswith("   OBJ: TH1D\tinner")), True),
        (
            lines[5:],
            [
                "   KEY: TH1D\tinner;1\tin",
                "  KEY: TDirectoryFile\tsub;1\tthe sub",
                "  KEY: TH1F\thpx;2\tpx [current cycle]",
                "  KEY: TH1F\thpx;1\tpx [backup cycle]",
            ],
        ),
        (bool(out.Get("hpx") is h), True),
        (bool(out.Get("sub/inner") is inner), True),
        (bool(out.Get("nothing") is None), True),
    )
    out.Close()
    expect(
        (bool(out.GetEND() == out.GetSize() > 0), True),
        (ROOT.gDirectory, ROOT.gROOT),
        (bool(not out.IsOpen()), True),
        (bool(out.GetSize() > 0), True),
    )
    out.Close()


def test_the_title_a_file_is_made_with_is_the_one_its_bytes_carry():
    """ROOT's ``new TFile(name, option, ftitle)`` puts the title in the key of the file
    itself, in all four places that key is written; ``hsimple.root`` has four of them."""
    out = ROOT.TFile("titled.root", "RECREATE", "Demo ROOT file with histograms")
    ROOT.TH1F("hpx", "px", 10, 0, 1).Write()
    out.Close()
    raw = pathlib.Path("titled.root").read_bytes()
    assert raw.count(b"\x1eDemo ROOT file with histograms") == 4  # ROOT's own hsimple.root
    with xrdroot.open_root("titled.root") as again:
        assert files._file_title(again) == "Demo ROOT file with histograms"


def test_a_file_read_hands_back_roots_classes_and_keeps_its_histograms(capsys):
    source = ROOT.TFile.Open(str(DATA / "gauss-h1.root"))
    expect(
        (source.GetOption(), "READ"),
        (bool(not source.IsWritable()), True),
    )
    key = source.GetListOfKeys().At(0)
    name = key.GetName()
    h = source.Get(name)
    expect(
        (bool(h.InheritsFrom("TH1")), True),
        (bool(source.Get(name) is h), True),
        (bool(h.GetDirectory() is source), True),
        (bool(getattr(source, name) is h), True),
        (source.GetNkeys(), source.GetListOfKeys().GetSize()),
    )
    with pytest.raises(AttributeError, match="has no object"):
        source.not_there  # noqa: B018
    with pytest.raises(AttributeError):
        source._private  # noqa: B018
    expect(
        (bool(key.ReadObj() is not None), True),
        (key.ReadObject["TH1"]().GetName(), key.GetName()),
        (key.GetCycle(), 1),
        (key.GetClassName(), h.ClassName()),
        (bool(key.GetNbytes() > 0), True),
        (bool(key.GetObjlen() > 0), True),
        (bool(key.GetKeylen() > 0), True),
        (bool(key.GetSeekKey() > 0), True),
        (bool(key.GetDatime().GetYear() > 2000), True),
        (bool(not key.IsFolder()), True),
        (bool(key.GetMotherDir() is source), True),
    )
    key.Print()
    expect(
        (bool(capsys.readouterr().out.startswith(f"TKey Name = {name}, Title = ")), True),
        (bool(source.FindKey(name) is key), True),
        (bool(source.GetKey("nothing") is None), True),
        (bool(source.GetVersion() > 0), True),
        (bool(source.GetCompressionSettings() >= 0), True),
        (bool(source.GetCompressionLevel() >= 0), True),
        (bool(source.GetCompressionAlgorithm() >= 0), True),
    )
    source.SetCompressionLevel(5)
    source.SetCompressionSettings(101)
    source.Flush()
    source.Close()


def test_directories_on_file_are_walked_as_root_walks_them(capsys):
    source = ROOT.TFile(str(DATA / "dirs-6.14.00.root"))
    below = source.Get("dir1")
    expect(
        (below.ClassName(), "TDirectoryFile"),
        (bool(below.GetPath().endswith("dirs-6.14.00.root:/dir1")), True),
        (bool(source.GetDirectory("dir1") is below), True),
        (bool(source.GetDirectory("dir1/..") is source), True),
        (bool(source.GetDirectory(f"{source.GetName()}:/dir1") is below), True),
        (bool(source.GetDirectory("nowhere") is None), True),
        (bool(source.Get("nowhere/h") is None), True),
        (bool(below.cd()), True),
        (ROOT.gDirectory, below),
    )
    below.ls("-d")
    source.ls("-m")
    expect(
        (bool("KEY: TDirectoryFile\tdir11;1" in capsys.readouterr().out), True),
        (bool(below.mkdir("new") is None), True),
        (bool("Can not create directory" in capsys.readouterr().err), True),
        (below.WriteTObject(ROOT.TNamed("n", "")), 0),
        (below.Write(), 0),
        (bool("not writable" in capsys.readouterr().err), True),
    )
    source.Close()
    assert ROOT.gDirectory == ROOT.gROOT


def test_a_listing_can_be_narrowed_by_a_wildcard(capsys):
    source = ROOT.TFile(str(DATA / "graphs.root"))
    source.ls("t*")
    lines = capsys.readouterr().out.splitlines()
    assert all("KEY" not in line or "\tt" in line for line in lines)
    source.Close()


def test_a_file_is_added_to_and_read_back(capsys):
    made = ROOT.TFile("up.root", "RECREATE")
    ROOT.TH1D("first", "one", 2, 0, 1)
    made.Write()
    made.Close()
    added = ROOT.TFile("up.root", "UPDATE")
    expect(
        (bool(added.IsWritable()), True),
        (added.GetListOfKeys().GetSize(), 1),
    )
    first = added.Get("first")
    ROOT.TH1D("second", "two", 2, 0, 1)
    added.Write()
    added.Close()
    back = ROOT.TFile("up.root")
    expect(
        (
            bool(
                [key.GetName() for key in back.GetListOfKeys()] == ["first", "second"]
                or len(back.GetListOfKeys()) >= 2
            ),
            True,
        ),
        (first.GetName(), "first"),
    )
    back.Close()


def test_modes_refuse_as_root_refuses(capsys):
    expect(
        (bool(ROOT.TFile.Open("missing.root") is None), True),
        (bool(ROOT.TFile("missing.root").IsZombie()), True),
    )
    ROOT.TFile("there.root", "RECREATE").Close()
    assert ROOT.TFile("there.root", "NEW").IsZombie()
    pathlib.Path("junk.root").write_text("not a root file at all, but long enough to read" * 3)
    assert ROOT.TFile("junk.root").IsZombie()
    err = capsys.readouterr().err
    expect(
        (bool("file missing.root does not exist" in err), True),
        (bool("file there.root already exists" in err), True),
        (bool("file junk.root is not a ROOT file" in err), True),
        (bool(ROOT.TFile("fresh.root", "UPDATE").IsWritable()), True),
    )
    zombie = ROOT.TFile("missing.root")
    zombie.Close()
    assert not zombie


def test_a_file_is_a_context_manager_and_closes_at_the_end():
    with ROOT.TFile("ctx.root", "RECREATE") as made:
        ROOT.TH1D("h", "", 1, 0, 1)
        made.Write()
    expect(
        (bool(not made.IsOpen()), True),
        (bool(made not in files._OPEN), True),
    )
    ROOT.TFile("left.root", "RECREATE")
    files._close_all()
    assert files._OPEN == []


def test_a_remote_name_is_taken_to_be_there():
    expect(
        (bool(files._exists("root://host//file.root")), True),
        (bool(files._exists("file:///no/such.root") is False), True),
        (bool(not files._local("root://host//f.root")), True),
        (bool(files._local("file:///x")), True),
    )


def test_objects_of_classes_with_no_wrapper_come_back_named_by_their_class():
    source = ROOT.TFile(str(DATA / "tformula.root"))
    other = source.Get("fconv")
    expect(
        (other.ClassName(), "TF1Convolution"),
        (other.GetName(), "TF1Convolution"),
        (other.GetTitle(), ""),
        (bool(isinstance(other, files.TOther)), True),
    )
    named = files.TOther("TThing", {"TNamed": {"fName": "n", "fTitle": "t"}})
    expect(
        ((named.GetName(), named.GetTitle()), ("n", "t")),
        (files.TOther().GetName(), "TObject"),
    )
    source.Close()


def test_a_memory_directory_below_a_file_is_one_on_file_too():
    made = ROOT.TFile("deep.root", "RECREATE")
    deep = made.mkdir("a/b")
    expect(
        (deep.GetPath(), "deep.root:/a/b"),
        (bool(made.mkdir("a") is None), True),
        (made.mkdir("a", "", True).GetName(), "a"),
        (bool(made.GetDirectory("a/b") is deep), True),
    )
    deep.cd()
    ROOT.TH1D("h", "", 1, 0, 1)
    made.Write()
    made.Close()
    back = ROOT.TFile("deep.root")
    assert back.Get("a/b/h").GetNbinsX() == 1
    back.Close()


def test_get_object_fills_the_pointer_it_is_handed_as_the_translator_hands_one():
    """FitHistoInFile.C's ``f->GetObject("histo", histo)``: the pointer is set, or nulled."""

    class Pointer:
        value: object = "unset"

    out = ROOT.TFile("objects.root", "RECREATE")
    ROOT.TH1D("histo", "t", 2, 0, 1).Fill(0.5)
    out.Write()
    out.Close()
    back = ROOT.TFile("objects.root")
    found, missing = Pointer(), Pointer()
    assert back.GetObject("histo", found) is found.value
    assert (found.value.GetEntries(), back.GetObject("nothing", missing), missing.value) == (
        1,
        None,
        None,
    )
    assert back.GetObject("histo", "not a pointer").GetName() == "histo"


def test_a_file_is_copied_here_and_a_missing_one_is_said_so(tmp_path, capsys) -> None:
    source = tmp_path / "a.root"
    source.write_bytes(b"0123")
    assert ROOT.TFile.Cp(str(source), str(tmp_path / "b.root"))
    assert (tmp_path / "b.root").read_bytes() == b"0123"
    assert "[TFile::Cp] Total 0.00 MB" in capsys.readouterr().err
    assert ROOT.TFile.Cp(str(source), str(tmp_path / "c.root"), False)
    assert not ROOT.TFile.Cp(str(tmp_path / "none.root"), str(tmp_path / "d.root"))
    assert "Error in <TFile::Cp>: cannot open source file" in capsys.readouterr().err


def test_a_file_is_written_with_roots_compression_unless_told_another() -> None:
    plain = ROOT.TFile("plain.root", "RECREATE")
    assert (plain.GetCompressionSettings(), plain.GetCompressionAlgorithm()) == (101, 1)
    plain.SetCompressionLevel(6)
    assert plain.GetCompressionSettings() == 106
    plain.SetCompressionSettings(207)
    assert (plain.GetCompressionSettings(), plain.GetCompressionLevel()) == (207, 7)
    plain.Close()
    raw = ROOT.TFile("raw.root", "RECREATE", "", 0)
    assert raw.GetCompressionSettings() == 0
    raw.Close()
    assert ROOT.TFile("plain.root").GetCompressionSettings() == 207


def test_a_file_on_a_macros_stack_is_closed_as_its_scope_ends() -> None:
    from xrdroot.pycint import translate
    from xrdroot.pycint.execute import run_source

    source = 'void t() { { TFile f("stack.root", "RECREATE"); } TFile::Open("stack.root"); }'
    assert "f._destruct()" in translate(source, "t.C")
    run_source(source, "t.C", root=ROOT)
    assert ROOT.TFile("stack.root").IsOpen()


def test_the_read_ahead_size_is_one_setting_that_every_file_reads_in_batches_of():
    out = ROOT.TFile("out.root", "RECREATE")
    expect(
        (ROOT.TFile.GetReadaheadSize(), files.READAHEAD_SIZE),
        (files.READAHEAD_SIZE, 256000),
    )
    ROOT.TFile.SetReadaheadSize(0)
    expect(
        (ROOT.TFile.GetReadaheadSize(), 0),
        (out.GetReadaheadSize(), 0),
    )
    ROOT.TFile.SetReadaheadSize()
    expect((ROOT.TFile.GetReadaheadSize(), files.READAHEAD_SIZE))
    out.Close()


def test_a_file_hands_back_the_cache_it_reads_blocks_through_and_none_without_one():
    out = ROOT.TFile("out.root", "RECREATE")
    tree = ROOT.TTree("T", "a tree")
    other = ROOT.TTree("U", "another tree")
    default, for_tree, for_other = object(), object(), object()
    expect(
        (out.GetCacheRead(), None),
        (out.GetCacheRead(tree), None),
    )
    out.SetCacheRead(for_tree, tree)
    expect(
        (bool(out.GetCacheRead(tree) is for_tree), True),
        (bool(out.GetCacheRead(other) is for_tree), True),  # no cache of its own: the default
    )
    out.SetCacheRead(for_other, other)
    expect(
        (bool(out.GetCacheRead(other) is for_other), True),
        (bool(out.GetCacheRead(tree) is for_tree), True),
        (bool(out.GetCacheRead() is for_other), True),  # the last one set is the default
    )
    out.SetCacheRead(None, other)
    expect(
        (bool(out.GetCacheRead(other) is None), True),  # it was that tree's own, now taken away
        (bool(out.GetCacheRead() is for_tree), True),  # the one cache left, though not default
    )
    out.SetCacheRead(default)
    expect(
        (bool(out.GetCacheRead(tree) is for_tree), True),
        (bool(out.GetCacheRead(other) is default), True),
        (bool(out.GetCacheRead() is default), True),
    )
    out.SetCacheRead(None)
    out.SetCacheRead(None, tree)
    expect(
        (out.GetCacheRead(), None),
        (out.GetCacheRead(tree), None),
    )
    out.Close()


#: ROOT 6.40.04's ``WriteObjectAny`` of a ``std::complex<float>(1, 2)`` and a
#: ``std::complex<double>(3, 4)``, as the oracle's file holds each key's payload: the record,
#: the version a class of no ``ClassDef`` carries, and the checksum of its description.
COMPLEX_KEY_PAYLOADS = {
    "cFloat_0": "4000000e00002e5c46083f80000040000000",
    "cDouble_0": "400000160000db2371a140080000000000004010000000000000",
}


def test_writeobjectany_writes_a_complex_under_the_class_name_it_is_handed():
    """A ``std::complex`` names no class of its own in Python, so the class ``WriteObjectAny``
    is handed is the one ROOT's key and record carry - checked against ROOT 6.40.04's file,
    whose payloads are these byte for byte."""
    from xrdroot.cppstd.stdlib import complex_type
    from xrdroot.file import open_root

    out = ROOT.TFile("complex.root", "RECREATE")
    expect(
        (out.WriteObjectAny(complex_type("float")(1, 2), "complex<float>", "cFloat_0"), 1),
        (out.WriteObjectAny(complex_type("double")(3, 4), "complex<double>", "cDouble_0"), 1),
        (out.GetKey("cFloat_0").GetClassName(), "complex<float>"),
        (out.GetKey("cFloat_0").GetTitle(), ""),  # ROOT gives a value no title
        (out.GetKey("cDouble_0").GetClassName(), "complex<double>"),
    )
    out.ls()
    out.Close()
    data = pathlib.Path("complex.root").read_bytes()
    with open_root("complex.root") as held:
        for name, payload in COMPLEX_KEY_PAYLOADS.items():
            key = held.key(name)
            at = key.seek_key + key.keylen
            expect((data[at : at + key.nbytes - key.keylen].hex(), payload))


def test_a_complex_written_by_its_class_name_reads_back_as_that_complex():
    """``Get`` hands back the value a key holds: a ``complex<float>`` key read is the kit's
    complex of that width, its two parts what the file holds - not an object of a class
    nothing describes."""
    from xrdroot.cppstd.stdlib import complex_type

    out = ROOT.TFile("complex.root", "RECREATE")
    out.WriteObjectAny(complex_type("float")(1, 2), "complex<float>", "cFloat_0")
    out.WriteObjectAny(complex_type("double")(3, 4), "complex<double>", "cDouble_0")
    out.Close()
    held = ROOT.TFile("complex.root")
    single, double = held.Get("cFloat_0"), held.Get("cDouble_0")
    expect(
        (type(single), complex_type("float")),
        (type(double), complex_type("double")),
        ((single.real(), single.imag()), (1.0, 2.0)),
        ((double.real(), double.imag()), (3.0, 4.0)),
        (complex(double), complex(3, 4)),
    )


def test_a_class_name_with_no_complex_or_another_class_says_no_value_of_it_is_made():
    """``_value_of`` makes a value of a ``std::complex`` job and of nothing else: a pointer to
    one is no class of its own, a template of no argument is no type, and a record that is no
    mapping is not a complex' two numbers - each ``None``, so the read falls back to what an
    object of an undescribed class is."""
    expect(
        (files._value_of("complex<float>*", {"_real": 1.0}), None),
        (files._value_of("complex", {"_real": 1.0}), None),
        (files._value_of("complex<float>", 1.0), None),
        (files._value_of("vector<int>", {"_real": 1.0}), None),
    )


def test_writeobjectany_names_a_value_by_its_class_and_says_when_it_cannot_write(capsys):
    """ROOT takes the name a key is written under, and a directory opened for reading writes
    nothing: ``WriteObjectAny`` says so, as ``WriteTObject`` does, and answers 0."""
    from xrdroot.cppstd.stdlib import complex_type

    out = ROOT.TFile("complex.root", "RECREATE")
    expect((out.WriteObjectAny(complex_type("float")(1, 2), "complex<float>", ""), 1))
    expect((out.GetKey("complex<float>").GetClassName(), "complex<float>"))
    out.Close()
    held = ROOT.TFile("complex.root")
    expect((held.WriteObjectAny(complex_type("float")(1, 2), "complex<float>", "z"), 0))
    assert "Error in <TDirectoryFile::WriteObjectAny>: Directory complex.root is not writable" in (
        capsys.readouterr().err
    )


def test_make_project_writes_a_header_for_each_class_the_file_describes(tmp_path):
    """``MakeProject`` gives a reader, under the project's directory, one header per class the
    file's streamer infos describe and the project, LinkDef and ``G__`` source that put them
    in a library."""
    held = ROOT.TFile(str(DATA / "declared-objects-6.40.root"))
    place = tmp_path / "libproj"
    held.MakeProject(str(place))
    derived = (place / "Derived.h").read_text()
    expect(
        ("#ifndef R__LIBPROJ_DERIVED" in derived, True),
        ("class Derived : public Base1 {" in derived, True),
        ("   int arr2[2][3];" in derived, True),
        ("   vector<vector<float> > vvf;" in derived, True),
        ("   ns::Deep deep;" in derived, True),
        ("#include <vector>" in derived, True),
        ("using namespace std;" in derived, True),
    )
    deep = (place / "Deep.h").read_text()
    expect(
        ("namespace ns {" in deep and "} // namespace ns" in deep, True),
        ("class Deep {" in deep, True),
        ("   int x;" in deep, True),
    )
    project = (place / "libprojProject.h").read_text()
    linkdef = (place / "libprojLinkDef.h").read_text()
    expected = '#include "Derived.h"'
    expect(
        (expected in project, True),
        ('#pragma link C++ class Derived+;' in linkdef, True),
        ((place / "G__libproj.cxx").read_text().startswith('#include "libprojProject.h"'), True),
    )


def test_a_project_leaves_out_the_classes_the_namespace_already_has(tmp_path):
    """A ROOT class the namespace implements, and a standard container, are not declared again
    - a second ``TNamed`` of the same name would be nothing a reader could use."""
    place = tmp_path / "libproj"
    ROOT.TFile(str(DATA / "declared-objects-6.40.root")).MakeProject(str(place))
    written = {found.name for found in place.glob("*.h")}
    expect(
        ("TNamed.h" not in written, True),
        ("pair.h" not in written, True),
        ("Inner.h" in written, True),
    )


def test_make_project_writes_the_classes_the_pattern_names(tmp_path):
    """ROOT, which generates a project for a few classes at a time as well as for all of them,
    is asked by name; ``*`` is every one."""
    place = tmp_path / "libproj"
    ROOT.TFile(str(DATA / "declared-objects-6.40.root")).MakeProject(str(place), "Inner")
    expect(
        ((place / "Inner.h").exists(), True),
        ((place / "Derived.h").exists(), False),
    )


def test_make_project_recreating_the_directory_empties_it_first(tmp_path):
    """``recreate`` makes the directory anew, as ROOT does, so a header of a class the file no
    longer describes does not outlive it."""
    place = tmp_path / "libproj"
    place.mkdir()
    (place / "stale.h").write_text("a class the file no longer describes")
    ROOT.TFile(str(DATA / "declared-objects-6.40.root")).MakeProject(str(place), "*", "recreate")
    expect(
        ((place / "stale.h").exists(), False),
        ((place / "Inner.h").exists(), True),
    )


def test_a_project_of_a_file_being_written_declares_no_class(tmp_path):
    """A file this session only writes has no streamer info to describe a class with, so its
    project is the empty one ROOT writes too."""
    made = ROOT.TFile(str(tmp_path / "empty.root"), "RECREATE")
    made.MakeProject(str(tmp_path / "libproj"))
    made.Close()
    project = (tmp_path / "libproj" / "libprojProject.h").read_text()
    expect(
        ('#include "' in project, False),
        ((tmp_path / "libproj" / "G__libproj.cxx").exists(), True),
    )


def test_a_record_of_an_rntuple_declares_the_cxx_type_each_of_its_members_was_written_with(
    tmp_path,
):
    """An RNTuple stores a record's members as physical columns, so the record is declared from
    the C++ type each field says it was written with - the one part of it no physical layout
    can say: a jagged list of pairs is a ``std::unordered_multimap`` here and an ordered
    ``std::multimap`` in the next file."""
    from types import SimpleNamespace

    members = [
        SimpleNamespace(field=SimpleNamespace(type="std::bitset<16>")),
        SimpleNamespace(field=SimpleNamespace(type="std::unordered_multimap<std::string,std::vector<float>>")),
    ]
    record = SimpleNamespace(names=["foo", "strange"], members=members)
    declared = files._record(SimpleNamespace(node=record))
    header = files._header("libproj", "MySTLEvent", declared)
    expect(
        ("class MySTLEvent {" in header, True),
        ("   std::bitset<16> foo;" in header, True),
        ("   std::unordered_multimap<std::string,std::vector<float>> strange;" in header, True),
        ("#include <bitset>" in header, True),
        ("#include <unordered_map>" in header, True),
    )


def test_a_pointer_member_is_declared_with_the_star_the_file_may_not_spell():
    """A streamer element whose kind is a pointer but whose type names a class alone - ROOT's
    ``TStreamerObjectAnyPointer`` - is declared as the pointer it is."""
    star = files._member(("TStreamerObjectAnyPointer", "p", "", 0, 0, 0, 0, (0,) * 5, "Thing", ()))
    plain = files._member(("TStreamerObjectPointer", "q", "", 0, 0, 0, 0, (0,) * 5, "TList*", ()))
    expect((star.spelling, "Thing*"), (plain.spelling, "TList*"))


def test_a_files_description_that_cannot_be_read_describes_no_class(monkeypatch):
    """A project is what a file describes; a description that cannot be read leaves the project
    empty rather than refusing it, as ROOT's generator writes what it finds."""
    from xrdroot.errors import FormatError

    def broken(source):
        raise FormatError("no")

    monkeypatch.setattr("xrdroot.streamers.read_infos", broken)
    expect((files._described(type("Reading", (), {"_source": None})()), {}))


def test_a_projects_rntuple_record_is_declared_from_the_cxx_type_its_fields_name(tmp_path):
    """An RNTuple written from a declared struct leaves the project a record whose fields name
    the C++ type each was written with, which is what the record's header is declared from."""
    ROOT.gInterpreter.Declare("struct Ev { std::vector<int> v; std::bitset<8> b; };")
    made = ROOT.TFile("a.root", "RECREATE")
    model = ROOT.RNTupleModel.Create()
    model.MakeField["Ev"]("test")
    writer = ROOT.RNTupleWriter.Append(model, "events", made)
    writer.Fill()
    writer.reset()
    made.Close()
    held = ROOT.TFile("a.root")
    held.MakeProject("libproj")
    header = (tmp_path / "libproj" / "Ev.h").read_text()
    expect(
        ("class Ev {" in header, True),
        ("   std::bitset<8> b;" in header, True),
        ((tmp_path / "libproj" / "libprojLinkDef.h").read_text().count("Ev+;"), 1),
    )


def test_the_rntuples_a_file_holds_are_looked_for_below_its_directories():
    """ROOT's project is made of the whole file, so a directory's RNTuples are as much its as
    the top's are."""
    import xrdroot

    held = xrdroot.open_root(str(DATA / "dirs-6.14.00.root"))
    try:
        expect((files._ntuples(held), []))
    finally:
        held.close()


def test_a_file_written_before_it_closes_can_be_read_and_added_to_again(tmp_path):
    """ROOT's ``TFile::Write`` writes the keys, the streamer info and the header as it goes, so
    a program that writes a file and reads it back in the same run reads it (its own ``w.C``
    probe answers ``open=1 tree=1``). The file stays open and what is written next is appended."""
    made = ROOT.TFile(str(tmp_path / "twice.root"), "RECREATE")
    first = ROOT.TH1D("first", "first", 4, 0, 4)
    first.Fill(1.0)
    first.Write()
    assert made.Write() == 1
    with ROOT.TFile(str(tmp_path / "twice.root")) as read:
        expect(
            (read.IsOpen(), True),
            (read.Get("first") is not None, True),
            (read.Get("second") is None, True),
        )
    made.cd()  # reading a file leaves the current directory elsewhere, as ROOT's does
    second = ROOT.TH1D("second", "second", 4, 0, 4)
    second.Fill(2.0)
    second.Write()
    made.Write()
    made.Close()
    with ROOT.TFile(str(tmp_path / "twice.root")) as read:
        expect(
            (read.Get("first") is not None, True),
            (read.Get("second") is not None, True),
        )


def test_a_tree_written_before_the_file_is_assembled_reads_back_with_its_entries(tmp_path):
    """A tree written to a file that is then written is on disk at the ``Write`` - one key per
    write, as ROOT's own probe measures (cycles 2 and 1 for ``t.Write()`` then ``f.Write()``) -
    and the entries are the ones filled."""
    import numpy

    made = ROOT.TFile(str(tmp_path / "tree.root"), "RECREATE")
    tree = ROOT.TTree("t", "t")
    x = numpy.zeros(1)
    tree.Branch("x", x, "x/D")
    for value in (1.0, 2.0, 3.0):
        x[0] = value
        tree.Fill()
    tree.Write()
    made.Write()
    made.Close()
    with ROOT.TFile(str(tmp_path / "tree.root")) as read:
        keys = [key for key in read.GetListOfKeys() if key.GetName() == "t"]
        again = read.Get("t")
        expect(
            (len(keys), 2),
            (again is not None, True),
            (int(again.GetEntries()), 3),
        )


def test_an_ntuple_written_into_a_file_is_readable_when_the_file_is_written(tmp_path):
    """The RNTuple goes out with the rest of the file's bookkeeping, so a ``Write`` makes it
    readable too - its pages, header and footer are written once between a checkpoint and the
    close that follows."""

    made = ROOT.TFile(str(tmp_path / "ntuple.root"), "RECREATE")
    model = ROOT.RNTupleModel.Create()
    field = model.MakeField["int"]("n")
    writer = ROOT.RNTupleWriter.Append(ROOT.std.move(model), "events", made)
    for value in (1, 2, 3):
        field.value = value
        writer.Fill()
    made.Write()
    del writer
    made.Close()
    with ROOT.TFile(str(tmp_path / "ntuple.root")) as read:
        expect((read.Get("events") is not None, True))
