"""``gROOT`` and ``gDirectory``: the open files, where the session is, and ROOT's lookup order.

Every test here has a :class:`Session` of its own, so nothing one of them
opens or goes into is seen by another; the module's :data:`gROOT` is only
looked at to see that it is one, with :data:`gDirectory` following it.
"""

from __future__ import annotations

import gc
import pathlib

import pytest

import xrdroot
from xrdroot import Histogram, gDirectory, gROOT, open_root
from xrdroot.session import CurrentDirectory, Session, split_location

DATA = pathlib.Path(__file__).parent / "data"
DIRS = str(DATA / "dirs-6.14.00.root")
GRAPHS = str(DATA / "graphs.root")


@pytest.fixture
def session():
    made = Session()
    yield made
    made.close_all()


# -- where a location splits ---------------------------------------------------


def test_a_location_splits_at_the_last_dot_root_so_a_urls_colons_stay_in_the_file():
    assert split_location("root://host:1094//store/f.root:dir/h") == (
        "root://host:1094//store/f.root",
        "dir/h",
    )


def test_a_location_with_no_path_is_the_whole_file():
    assert split_location("s3://bucket/f.root") == ("s3://bucket/f.root", "")


def test_a_leading_slash_on_the_path_is_the_top_of_the_file():
    assert split_location("f.root:/dir/h") == ("f.root", "dir/h")


def test_a_path_inside_a_file_named_like_a_file_stays_in_the_path():
    assert split_location("a.root:dir.root/h") == ("a.root", "dir.root/h")


def test_a_bare_path_names_no_file():
    assert split_location("dir/h") == (None, "dir/h")


# -- the files -----------------------------------------------------------------


def test_opening_a_file_lists_it_and_goes_into_it(session):
    opened = session.open(DIRS)
    assert opened in session.files
    assert session.pwd() == f"{DIRS}:/"
    assert session.directory is opened


def test_a_file_open_root_opened_is_listed_while_something_holds_it(session):
    held = open_root(GRAPHS)
    assert held in session.files
    held.close()
    assert held not in session.files


def test_a_file_nobody_holds_is_no_longer_listed(session):
    opened = open_root(GRAPHS)
    marker = id(opened)
    opened.close()
    del opened
    gc.collect()
    assert marker not in {id(found) for found in session.files}


def test_a_file_whose_handle_its_owner_closed_is_not_listed(session):
    with open(GRAPHS, "rb") as handle:
        opened = open_root(handle)
        assert opened in session.files
    assert opened.closed
    assert opened not in session.files


def test_close_all_closes_the_files_the_session_opened_and_goes_to_the_top(session):
    opened = session.open(DIRS)
    elsewhere = open_root(GRAPHS)
    session.close_all()
    assert opened.closed
    assert not elsewhere.closed  # whoever opened it closes it
    assert session.pwd() == "Rint:/"
    elsewhere.close()


def test_a_closed_current_file_puts_the_session_back_at_the_top(session):
    opened = session.open(DIRS)
    opened.close()
    assert session.pwd() == "Rint:/"
    assert session.directory is session


def test_the_session_says_where_it_is_and_what_it_has_open(session):
    session.open(GRAPHS)
    assert repr(session).startswith(f"<Session at {GRAPHS}:/ with ")


# -- going places ----------------------------------------------------------------


def test_cd_walks_down_up_and_from_the_top(session):
    session.open(DIRS)
    session.cd("dir1/dir11")
    assert session.pwd() == f"{DIRS}:/dir1/dir11"
    session.cd("..")
    assert session.pwd() == f"{DIRS}:/dir1"
    session.cd("./dir11/../dir11")
    assert session.pwd() == f"{DIRS}:/dir1/dir11"
    session.cd("/dir2")
    assert session.pwd() == f"{DIRS}:/dir2"


def test_cd_hands_back_the_directory_it_went_into(session):
    session.open(DIRS)
    assert session.cd("dir1").keys() == ["dir11"]


def test_cd_names_an_open_file_by_its_name_its_base_name_or_its_path(session):
    session.open(DIRS)
    session.open(GRAPHS)
    session.cd("dirs-6.14.00.root:/dir1")
    assert session.pwd() == f"{DIRS}:/dir1"
    session.cd(f"{GRAPHS}:")
    assert session.pwd() == f"{GRAPHS}:/"


def test_cd_matches_a_local_path_however_it_is_spelled(session):
    session.open(DIRS)
    spelled = str(DATA / ".." / "data" / "dirs-6.14.00.root")
    session.cd(f"{spelled}:dir3")
    assert session.pwd() == f"{DIRS}:/dir3"


def test_cd_into_a_file_that_is_not_open_says_which_are(session):
    session.open(GRAPHS)
    with pytest.raises(KeyError, match=r"'other.root' is not an open file.*graphs.root"):
        session.cd("other.root:/dir")


def test_a_file_that_is_not_open_is_not_looked_for(session):
    assert session.get("nowhere-at-all.root:h") is None
    with pytest.raises(KeyError, match=r"nowhere-at-all\.root' is not an open file"):
        session["nowhere-at-all.root:h"]


def test_cd_into_a_url_matches_only_by_name(session):
    session.open(GRAPHS)
    with pytest.raises(KeyError):
        session.cd("root://host//graphs-elsewhere.root:")


def test_cd_to_a_path_at_the_top_asks_for_the_file_too(session):
    with pytest.raises(KeyError, match=r"not in one: gROOT.cd\('file.root:/dir1'\)"):
        session.cd("dir1")


def test_cd_into_something_that_is_not_a_directory_refuses_by_class(session):
    session.open(DIRS)
    with pytest.raises(NotADirectoryError, match="'dir1/dir11/h1' is a TH1F, not a directory"):
        session.cd("dir1/dir11/h1")


def test_cd_to_a_file_or_directory_object_goes_there(session):
    session.open(GRAPHS)  # a file the directory is not in, looked at first
    opened = session.open(DIRS)
    session.cd()
    assert session.cd(opened) is opened
    assert session.pwd() == f"{DIRS}:/"
    inner = opened["dir1/dir11"]
    session.cd(inner)
    assert session.pwd() == f"{DIRS}:/dir1/dir11"


def test_a_directory_whose_file_is_not_listed_is_its_own_top(session):
    with open_root(DIRS) as opened:
        inner = opened["dir1"]
    session.cd(inner)
    assert session.pwd() == "Directory:/"


def test_cd_with_nothing_goes_back_to_the_top(session):
    session.open(DIRS)
    assert session.cd() is session
    assert session.pwd() == "Rint:/"


# -- listing ---------------------------------------------------------------------


def test_ls_at_the_top_lists_memory_and_the_open_files(session):
    session.open(GRAPHS)
    session.cd()
    session.add(Histogram.book("h", (10, 0.0, 1.0), title="a title"))
    listing = session.ls()
    assert listing.splitlines()[0].startswith("Rint*")
    assert " OBJ: TH1D\th\ta title" in listing
    assert f" TFile*\t\t{GRAPHS}" in listing


def test_ls_in_a_file_lists_its_keys_as_root_does(session):
    session.open(GRAPHS)
    listing = session.ls().splitlines()
    assert listing[:2] == [f"TFile**\t\t{GRAPHS}", f" TFile*\t\t{GRAPHS}"]
    assert "  KEY: TGraph\ttg;1\tgraph without errors" in listing


def test_ls_in_a_directory_names_it_and_its_keys(session):
    session.open(DIRS)
    session.cd("dir1")
    listing = session.ls().splitlines()
    assert listing[0] == f"TDirectoryFile*\t\tdir1\t{DIRS}:/dir1"
    assert "  KEY: TDirectory\tdir11;1\tdir11" in listing


def test_ls_of_somewhere_else_lists_it_without_going_there(session):
    session.open(DIRS)
    assert "KEY: TH1F\th1;1" in session.ls("dir1/dir11")
    assert session.pwd() == f"{DIRS}:/"


# -- finding things by name ------------------------------------------------------------


def test_a_located_name_is_read_exactly_where_it_says(session):
    session.open(DIRS)
    session.open(GRAPHS)
    assert session.get("dirs-6.14.00.root:/dir1/dir11/h1").name == "h1"
    assert session["graphs.root"].name == GRAPHS


def test_a_leading_slash_reads_from_the_top_of_the_current_file(session):
    session.open(DIRS)
    session.cd("dir2")
    assert session["/dir1/dir11/h1"].name == "h1"


def test_a_leading_slash_at_the_top_looks_everywhere(session):
    session.add(Histogram.book("h", (10, 0.0, 1.0)))
    with pytest.raises(KeyError):
        session["/h"]


def test_a_bare_name_is_looked_for_here_then_in_memory_then_in_every_file(session):
    session.open(DIRS)
    session.open(GRAPHS)
    booked = session.add(Histogram.book("tg", (10, 0.0, 1.0)))
    assert session["tg"].classname == "TGraph"  # here first: the current file
    session.cd("dirs-6.14.00.root:/dir1")
    assert session["tg"] is booked  # then memory
    session.remove("tg")
    assert session["tg"].classname == "TGraph"  # then every file
    assert session["dir11/h1"].name == "h1"  # a path from here


def test_a_name_that_is_nowhere_says_where_it_looked(session):
    with pytest.raises(KeyError, match=r"'nothing' is not in Rint:/, in memory, or in any open"):
        session["nothing"]
    assert session.get("nothing") is None
    assert session.get("nothing", 7) == 7
    assert "nothing" not in session


def test_the_top_of_the_session_holds_what_was_put_in_memory(session):
    booked = Histogram.book("h", (10, 0.0, 1.0))
    assert session.add(booked) is booked
    assert session.add(booked, "again") is booked
    assert session.keys() == ["h", "again"]
    assert list(session) == ["h", "again"]
    assert len(session) == 2
    assert "h" in session
    assert session.objects == {"h": booked, "again": booked}


def test_something_with_no_name_is_put_in_memory_only_under_one_given(session):
    with pytest.raises(ValueError, match=r"a list has no name to be found by"):
        session.add([1, 2])
    assert session.add([1, 2], "pair") == [1, 2]


# -- macros ------------------------------------------------------------------------


def test_a_macro_is_run_with_roots_names_and_its_function_called(session, tmp_path):
    macro = tmp_path / "fill.py"
    macro.write_text(
        "def fill(n, scale=1):\n"
        "    h = Histogram.book('h', (4, 0.0, 4.0))\n"
        "    h.fill(np.arange(n) % 4)\n"
        "    return h.sum() * scale\n"
    )
    assert session.macro(macro, 8) == 8.0
    assert session.macro(str(macro), 8, 2) == 16.0


def test_a_macro_without_a_function_of_its_name_is_just_run(session, tmp_path):
    macro = tmp_path / "setup.py"
    macro.write_text("print(__name__, __file__.endswith('setup.py'), gROOT is not None)\n")
    assert session.macro(macro) is None


def test_arguments_for_a_macro_with_no_function_to_take_them_are_refused(session, tmp_path):
    macro = tmp_path / "setup.py"
    macro.write_text("x = 1\n")
    with pytest.raises(TypeError, match=r"setup.py defines no function setup\(\) to hand 1, 'a'"):
        session.macro(macro, 1, "a")


def test_running_source_hands_back_what_it_defined(session):
    names = session.run("y = np.arange(3).sum()\nz = TRandom3")
    assert names["y"] == 3
    assert names["z"] is xrdroot.TRandom3


def test_help_lists_the_prompts_commands(session):
    assert ".x macro.py" in session.help()


# -- the one session, and gDirectory ------------------------------------------------------


def test_groot_is_a_session_and_gdirectory_follows_it():
    assert isinstance(gROOT, Session)
    assert isinstance(gDirectory, CurrentDirectory)
    assert xrdroot.gROOT is gROOT


def test_gdirectory_is_whatever_directory_the_session_is_in(session):
    here = CurrentDirectory(session)
    session.open(DIRS)
    session.cd("dir1/dir11")
    assert here.keys() == ["h1"]
    assert here["h1"].name == "h1"
    assert list(here) == ["h1"]
    assert len(here) == 1
    assert "h1" in here
    assert repr(here) == f"<gDirectory {DIRS}:/dir1/dir11>"
    assert "KEY: TH1F" in here.ls()
    here.cd("..")
    assert here.pwd() == f"{DIRS}:/dir1"


def test_gdirectory_at_the_top_is_the_session(session):
    here = CurrentDirectory(session)
    session.add(Histogram.book("h", (10, 0.0, 1.0)))
    assert here.keys() == ["h"]
    assert here.cd() is session
