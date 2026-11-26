"""``std::filesystem``: the path as its parts, and the questions a program asks of one."""

from __future__ import annotations

import pytest

from xrdroot.cppstd import filesystem as fs


def test_a_path_is_the_parts_of_it_and_the_text_it_was_made_of():
    made = fs.path("/tmp/one/two.py")
    assert made.string() == "/tmp/one/two.py" and str(made) == made.string()
    assert made.c_str() == made.string() and made.filename().string() == "two.py"
    assert made.stem().string() == "two" and made.extension().string() == ".py"
    assert made.parent_path().string() == "/tmp/one"
    assert made.is_absolute() and not fs.path("two.py").is_absolute()
    assert not made.empty() and fs.path().empty()


def test_a_path_joined_with_another_keeps_the_left_on_the_front():
    """C++'s ``operator/``: a relative name is appended, an absolute one replaces the left."""
    base = fs.path("/tmp")
    assert (base / "name").string() == "/tmp/name"
    assert (base / fs.path("name")).string() == "/tmp/name"
    assert (base / "/elsewhere").string() == "/elsewhere"  # absolute: the left is dropped
    assert (fs.path() / "name").string() == "name"  # empty: what is added stands alone
    assert (fs.path("/tmp/") / "name").string() == "/tmp/name"  # no doubled separator


def test_a_file_can_be_removed_and_asked_about(tmp_path):
    made = tmp_path / "one.txt"
    made.write_text("x")
    assert fs.exists(made) and fs.is_regular_file(made) and not fs.is_directory(made)
    assert fs.file_size(made) == 1
    assert fs.remove(made) is True
    assert not fs.exists(made)
    assert fs.remove(made) is False  # C++ throws, a program asks first: answered, not raised


def test_a_directory_is_removed_only_when_it_is_empty(tmp_path):
    full = tmp_path / "full"
    full.mkdir()
    (full / "one.txt").write_text("x")
    assert fs.remove(full) is False and full.is_dir()  # a full directory is not a file to remove


def test_create_directories_makes_every_missing_one_and_says_so(tmp_path):
    wanted = fs.path(tmp_path / "a" / "b")
    assert fs.create_directories(wanted) is True and fs.is_directory(wanted)
    assert fs.create_directories(wanted) is False  # there was nothing left to make


def test_the_temporary_directory_and_the_current_one_are_where_this_machine_has_them(tmp_path):
    temporary = fs.temp_directory_path()
    assert isinstance(temporary, fs.path) and fs.is_directory(temporary)
    assert fs.current_path().string() == os_getcwd()


def os_getcwd() -> str:
    """The working directory, read afresh: another test may have changed it."""
    import os

    return os.getcwd()


def test_replacing_the_filename_keeps_the_directory_and_moving_a_path_reads_it():
    made = fs.path("/tmp/one/two.py")
    assert made.replace_filename("three.py").string() == "/tmp/one/three.py"
    assert made == fs.path("/tmp/one/two.py") and made != fs.path("/tmp/one/other.py")
    assert made == "/tmp/one/two.py"  # compared with its text, as a C++ path compares
    assert str(fs.path(made)) == made.string()  # made of a path: its text again
    import os

    assert os.fspath(made) == made.string()
    assert repr(made) == '<std::filesystem::path "/tmp/one/two.py">'


def test_a_path_hash_the_text_it_holds():
    assert len({fs.path("/tmp"), fs.path("/tmp")}) == 1


@pytest.mark.parametrize("name", ["with'quote", 'with"quote', "with\nnewline", "with\rcarriage"])
def test_a_name_is_a_path_and_nothing_to_escape(name):
    """``TPython::LoadMacro`` takes a path, so a quote or a newline in it is just its name."""
    made = fs.temp_directory_path() / f"tpython_loadmacro_{name}_test.py"
    assert made.filename().string() == f"tpython_loadmacro_{name}_test.py"
