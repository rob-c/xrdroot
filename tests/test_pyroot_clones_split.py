"""A ``TClonesArray`` split into a branch per member, as ``TTree::Branch`` splits one.

The top branch counts the objects each entry holds (``fType`` 3) and each
member of every object is a branch of its own (``fType`` 31) - the objects'
``TObject`` part split into ``fUniqueID`` and ``fBits``, a ``TString`` member a
row of strings. ``TTree::Print`` of such a tree is ROOT 6.40's, line for line,
and ROOT reads the file back object for object (checked with the oracle when
this was written); here it is read back by xrdroot both ways.
"""

from __future__ import annotations

import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.wbranch import clones_members

#: ``TTree::Print`` of ROOT 6.40 for the tree :func:`_written` makes, its sizes and all.
PRINTED = """\
*Br    0 :N         : Int_t N_                                               *
*Entries :        3 : Total  Size=       3322 bytes  File Size  =         97 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.00     *
*............................................................................*
*Br    1 :N.fUniqueID : UInt_t fUniqueID[N_]                                 *
*Entries :        3 : Total  Size=        708 bytes  File Size  =        106 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.12     *
*............................................................................*
*Br    2 :N.fBits   : UInt_t fBits[N_]                                       *
*Entries :        3 : Total  Size=        688 bytes  File Size  =        102 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br    3 :N.fName   : TString fName[N_]                                      *
*Entries :        3 : Total  Size=        700 bytes  File Size  =        124 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.02     *
*............................................................................*
*Br    4 :N.fTitle  : TString fTitle[N_]                                     *
*Entries :        3 : Total  Size=        705 bytes  File Size  =        125 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.02     *
*............................................................................*
"""


def _written(path: str) -> None:
    """Three entries of one, two and three ``TNamed``, the branch split as ROOT splits it."""
    out = ROOT.TFile(path, "RECREATE")
    tree = ROOT.TTree("T", "T")
    held = ROOT.TClonesArray("TNamed")
    tree.Branch("N", held)
    for entry in range(3):
        held.Clear()
        for at in range(entry + 1):
            held.AddAt(ROOT.TNamed(f"N_{entry}_{at}", f"N_{entry}_{at}"), at)
        tree.Fill()
    tree.Write()
    tree.Print()
    out.Close()


def test_a_split_clones_array_is_printed_as_root_prints_it(tmp_path, capsys):
    _written(str(tmp_path / "clones.root"))
    assert PRINTED in capsys.readouterr().out


def test_each_member_reads_back_as_a_row_per_entry(tmp_path):
    path = tmp_path / "clones.root"
    _written(str(path))
    with xrdroot.open_root(path) as found:
        tree = found["T"]
        assert tree.keys() == ["N", "N.fUniqueID", "N.fBits", "N.fName", "N.fTitle"]
        assert tree.arrays(["N.fName"])["N.fName"][2] == ["N_2_0", "N_2_1", "N_2_2"]
        record = tree.branches["N.fBits"].record
        assert record.classname == "TObject"
        assert record.element[1:] == ("", 2417737773, 1, 1, 31, 15, 0)


def test_a_script_reads_the_objects_back_into_its_clones_array_or_a_new_one(tmp_path):
    path = str(tmp_path / "clones.root")
    _written(path)
    back = ROOT.TFile(path)
    tree = back.Get("T")
    mine = ROOT.TClonesArray("TNamed")
    tree.SetBranchAddress("N", mine)
    tree.GetEntry(1)
    assert [o.GetName() for o in mine] == ["N_1_0", "N_1_1"] and mine[1].GetTitle() == "N_1_1"
    back.Close()
    back = ROOT.TFile(path)
    tree = back.Get("T")
    cell = ROOT.MakeNullPointer(ROOT.TClonesArray) if hasattr(ROOT, "MakeNullPointer") else None
    holder = type("Cell", (), {"value": cell})()
    tree.SetBranchAddress("N", holder)
    tree.GetEntry(2)
    assert holder.value.GetClass().GetName() == "TNamed" and len(holder.value) == 3
    back.Close()


def test_long_names_go_out_with_four_bytes_of_length(tmp_path):
    path = str(tmp_path / "long.root")
    out = ROOT.TFile(path, "RECREATE")
    tree = ROOT.TTree("T", "T")
    held = ROOT.TClonesArray("TObjString")
    tree.Branch("S", held, 32000, 1)
    held.AddAt(ROOT.TObjString("x" * 300), 0)
    tree.Fill()
    tree.Write()
    out.Close()
    with xrdroot.open_root(path) as found:
        assert found["T"].arrays(["S.fString"])["S.fString"][0] == ["x" * 300]


def test_a_class_without_a_description_or_with_members_not_split_here_is_refused():
    with pytest.raises(UnsupportedFeatureError, match="carries a description of"):
        clones_members("MyMacroClass")
    with pytest.raises(UnsupportedFeatureError, match="TH1::fXaxis is a Object member"):
        clones_members("TH1F")
    names = [(m.owner, m.name) for m in clones_members("TLine")]
    assert ("TAttLine", "fLineColor") in names and ("TLine", "fX1") in names
