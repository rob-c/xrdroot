"""``gROOT``, ``gSystem``, ``gDirectory``, and the clocks: the session a script runs in."""

from __future__ import annotations

import os
import pathlib
import time

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_stopwatch_prints_real_and_cpu_time_as_root_does(capsys):
    watch = ROOT.TStopwatch()
    watch.Stop()
    watch.Continue()
    watch.Stop()
    watch.Start(False)
    watch.Print()
    watch.Print("m")
    watch.Print("u")
    lines = capsys.readouterr().out.splitlines()
    expect(
        (bool(lines[0].startswith("Real time 0:00:0")), True),
        (bool(lines[0].endswith(", 2 slices")), True),
        (bool(", CP time " in lines[1]), True),
        (len(lines[2].split(",")[0].split(":")[-1]), 9),
        (bool(watch.RealTime() >= 0), True),
        (bool(watch.CpuTime() >= 0), True),
        (watch.Counter(), 2),
    )
    watch.Reset()
    watch.ResetCpuTime(1.5)
    assert watch.CpuTime() == 1.5


def test_a_stopwatch_never_started_does_not_continue():
    watch = ROOT.TStopwatch()
    watch.Reset()
    watch._state = watch.kUndefined
    watch.Continue()
    assert watch._state == watch.kUndefined


def test_a_benchmark_shows_each_stopwatch_by_name(capsys):
    ROOT.gBenchmark.Reset()
    ROOT.gBenchmark.Start("fill")
    ROOT.gBenchmark.Show("fill")
    ROOT.gBenchmark.Print("fill")
    ROOT.gBenchmark.Stop("none")
    real, cpu = ROOT.gBenchmark.Summary()
    out = capsys.readouterr().out.splitlines()
    expect(
        (bool(out[0].startswith("fill      : Real Time =   ")), True),
        (bool("seconds Cpu Time" in out[0]), True),
        (bool(out[-1].startswith("TOTAL     : Real Time =")), True),
        (bool(real >= 0), True),
        (bool(cpu >= 0), True),
        (ROOT.gBenchmark.GetRealTime("none"), 0.0),
        (ROOT.gBenchmark.GetCpuTime("none"), 0),
    )
    ROOT.gBenchmark.Start("other")
    ROOT.gBenchmark.Start("fill")  # started again, it keeps its place
    assert [ROOT.gBenchmark.GetBench(n) for n in ("fill", "other", "none")] == [0, 1, -1]


def test_a_datime_is_a_date_and_time_to_the_second(capsys):
    when = ROOT.TDatime(2026, 9, 25, 12, 30, 5)
    expect(
        ((when.GetDate(), when.GetTime()), (20260925, 123005)),
        ((when.GetYear(), when.GetMonth(), when.GetDay()), (2026, 9, 25)),
        ((when.GetHour(), when.GetMinute(), when.GetSecond()), (12, 30, 5)),
        (when.GetDayOfWeek(), 5),
        (when.AsSQLString(), "2026-09-25 12:30:05"),
    )
    same = ROOT.TDatime(20260925, 123005)
    expect(
        (same, when),
        (hash(same), hash(when)),
        (bool(same != "x"), True),
        (ROOT.TDatime(when.Convert()), when),
    )
    when.Print()
    expect(
        (capsys.readouterr().out, "Date/Time = Fri Sep 25 12:30:05 2026\n"),
        (bool(abs(ROOT.TDatime().Convert() - time.time()) < 5), True),
    )
    stamp = ROOT.TTimeStamp()
    assert stamp.AsDouble() == float(stamp.GetSec())


def test_gsystem_says_a_path_is_missing_the_way_root_does(tmp_path):
    (tmp_path / "there.txt").write_text("x")
    expect(
        (bool(ROOT.gSystem.AccessPathName("there.txt") is False), True),
        (bool(ROOT.gSystem.AccessPathName("missing.txt") is True), True),
        (bool(ROOT.gSystem.AccessPathName("there.txt", ROOT.kReadPermission) is False), True),
        (bool(ROOT.gSystem.kFileExists == ROOT.kFileExists == 0), True),
    )


def test_gsystem_loads_and_includes_without_complaint():
    ROOT.gSystem.Unload("libPhysics")  # as no earlier test in this process loaded it
    expect(
        (ROOT.gSystem.Load("libPhysics"), 0),
        (bool("libPhysics" in ROOT.gSystem.GetLibraries()), True),
    )
    ROOT.gSystem.Unload("libPhysics")
    ROOT.gSystem.Unload("libNothing")
    ROOT.gSystem.AddIncludePath("-I/a")
    ROOT.gSystem.AddIncludePath("-I/b")
    assert ROOT.gSystem.GetIncludePath() == "-I/a -I/b"
    ROOT.gSystem.SetIncludePath("-I/c")
    assert ROOT.gSystem.GetIncludePath() == "-I/c"
    ROOT.gSystem.AddLinkedLibs("-lm")
    ROOT.gSystem.SetDynamicPath("/x")
    ROOT.gSystem.AddDynamicPath("/y")
    expect(
        (ROOT.gSystem.GetDynamicPath(), "/x:/y"),
        (ROOT.gSystem.CompileMacro("no-such-macro.C"), 0),  # ACLiC has nothing to build
    )
    ROOT.gSystem.SetDynamicPath("")
    ROOT.gSystem.SetBuildDir("/tmp")
    assert (ROOT.gSystem.GetBuildDir(), ROOT.gSystem.IsBuildDirFlat()) == ("/tmp", False)
    ROOT.gSystem.SetBuildDir("")


def test_gsystem_reads_and_sets_the_environment(monkeypatch):
    monkeypatch.delenv("PYROOT_TEST", raising=False)
    assert ROOT.gSystem.Getenv("PYROOT_TEST") is None
    ROOT.gSystem.Setenv("PYROOT_TEST", 5)
    assert ROOT.gSystem.Getenv("PYROOT_TEST") == "5"
    ROOT.gSystem.Unsetenv("PYROOT_TEST")
    expect(
        (bool("PYROOT_TEST" not in os.environ), True),
        (bool(ROOT.gSystem.HostName()), True),
        (ROOT.gSystem.GetPid(), os.getpid()),
        (ROOT.gSystem.GetUid(), os.getuid()),
        (bool(ROOT.gSystem.GetUserInfo() is None), True),
        (bool(ROOT.gSystem.GetBuildArch()), True),
        (bool(ROOT.gSystem.Now() > 0), True),
    )


def test_gsystem_runs_commands_and_sleeps():
    expect(
        (ROOT.gSystem.Exec("true"), 0),
        (ROOT.gSystem.GetFromPipe("echo hi"), "hi"),
        (bool(ROOT.gSystem.ProcessEvents() is False), True),
    )
    ROOT.gSystem.Sleep(1)
    with pytest.raises(SystemExit):
        ROOT.gSystem.Exit(0)
    with pytest.raises(SystemExit):
        ROOT.gSystem.Abort()


def test_gsystem_walks_and_names_paths(tmp_path):
    here = os.getcwd()
    expect(
        (bool(ROOT.gSystem.pwd() == ROOT.gSystem.WorkingDirectory() == here), True),
        (ROOT.gSystem.mkdir("a/b", True), 0),
        (ROOT.gSystem.mkdir("a"), -1),
        (ROOT.gSystem.MakeDirectory("c"), 0),
        (bool(ROOT.gSystem.cd("a")), True),
        (bool(not ROOT.gSystem.ChangeDirectory("nowhere")), True),
    )
    os.chdir(here)
    expect(
        (ROOT.gSystem.BaseName("/x/y.root"), "y.root"),
        (ROOT.gSystem.BaseName("/"), "/"),
        (ROOT.gSystem.DirName("/x/y.root"), "/x"),
        (ROOT.gSystem.DirName("y"), "."),
        (ROOT.gSystem.DirName("/y"), "/"),
        (ROOT.gSystem.GetDirName("/x/y"), "/x"),
        (ROOT.gSystem.ConcatFileName("a", "b"), os.path.join("a", "b")),
    )
    name = ROOT.TString("f.root")
    expect(
        (ROOT.gSystem.PrependPathName("dir", name), "dir/f.root"),
        (name, "dir/f.root"),
        (bool(ROOT.gSystem.IsAbsoluteFileName("/x")), True),
        (ROOT.gSystem.UnixPathName("a"), "a"),
        (ROOT.gSystem.HomeDirectory(), os.path.expanduser("~")),
        (bool(os.path.isdir(ROOT.gSystem.TempDirectory())), True),
    )


def test_gsystem_expands_and_finds_files(tmp_path, monkeypatch):
    monkeypatch.setenv("PYROOT_DIR", "there")
    assert ROOT.gSystem.ExpandPathName("$PYROOT_DIR/x") == "there/x"
    text = ROOT.TString("$PYROOT_DIR")
    expect(
        (bool(ROOT.gSystem.ExpandPathName(text) is False), True),
        (text, "there"),
    )
    (tmp_path / "found.txt").write_text("x")
    expect(
        (ROOT.gSystem.Which(f"/nowhere:{tmp_path}", "found.txt"), str(tmp_path / "found.txt")),
        (bool(ROOT.gSystem.Which(".", "missing.txt") is None), True),
        (ROOT.gSystem.Which("", str(tmp_path / "found.txt")), str(tmp_path / "found.txt")),
        (bool(ROOT.gSystem.Which("", "/no/such/file") is None), True),
        (bool(ROOT.gSystem.Which("/nowhere", "sh", ROOT.kExecutePermission)), True),
        (bool(ROOT.gSystem.FindFile(str(tmp_path), "found.txt")), True),
        (bool(ROOT.gSystem.IsFileInIncludePath("found.txt")), True),
    )


def test_gsystem_moves_and_removes_files(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    expect(
        (ROOT.gSystem.CopyFile("a.txt", "b.txt"), 0),
        (ROOT.gSystem.CopyFile("a.txt", "b.txt"), -1),
        (ROOT.gSystem.CopyFile("a.txt", "b.txt", True), 0),
        (ROOT.gSystem.Rename("b.txt", "c.txt"), 0),
        (ROOT.gSystem.Rename("nope", "d"), -1),
        (ROOT.gSystem.Unlink("c.txt"), 0),
        (ROOT.gSystem.Unlink("c.txt"), -1),
    )
    os.mkdir("empty")
    assert ROOT.gSystem.Unlink("empty") == 0


def test_gsystem_lists_a_directory_an_entry_at_a_time(tmp_path):
    (tmp_path / "one").write_text("")
    handle = ROOT.gSystem.OpenDirectory(str(tmp_path))
    names = []
    entry = ROOT.gSystem.GetDirEntry(handle)
    while entry is not None:
        names.append(entry)
        entry = ROOT.gSystem.GetDirEntry(handle)
    ROOT.gSystem.FreeDirectory(handle)
    expect(
        (names, [".", "..", "one"]),
        (bool(ROOT.gSystem.OpenDirectory("nowhere") is None), True),
        (bool(ROOT.gSystem.GetDirEntry(None) is None), True),
    )


def test_gsystem_makes_a_temporary_file_named_into_the_string_it_was_given(tmp_path):
    name = ROOT.TString(str(tmp_path / "root_test_"))
    made = ROOT.gSystem.TempFileName(name, None, ".txt")
    made.write("text")
    made.stream.close()
    assert str(name).startswith(str(tmp_path / "root_test_")) and str(name).endswith(".txt")
    assert pathlib.Path(str(name)).read_text() == "text"
    other = ROOT.gSystem.TempFileName("plain_", str(tmp_path))
    assert other.name.startswith(str(tmp_path / "plain_"))
    other.stream.close()
    elsewhere = ROOT.gSystem.TempFileName("xrd_")
    assert elsewhere.name.startswith(ROOT.gSystem.TempDirectory())
    elsewhere.stream.close()
    os.remove(elsewhere.name)
    assert ROOT.gSystem.CleanCompiledMacros() is None


def test_a_uuid_is_version_1_made_now_or_version_4_at_random_and_orders_by_its_bits():
    now, random = ROOT.TUUID(), ROOT.TUUID.UUIDv4()
    assert now.AsString()[14] == "1" and random.AsString()[14] == "4"
    assert random.AsString()[19] in "89ab" and now.GetUUIDNumber() == 0
    again = ROOT.TUUID(random.AsString())
    assert again == random and hash(again) == hash(random) and again.Hash() == random.Hash()
    assert again.Compare(random) == 0 and (now < random) == (now.Compare(random) == -1)
    assert len({ROOT.TUUID.UUIDv4() for _ in range(100)}) == 100 and now != "x"
    assert repr(again) == f"<TUUID {random.AsString()}>"


def test_the_derivatives_of_samples_are_central_differences_with_one_sided_ends():
    squares = [float(x * x) for x in range(6)]
    assert list(ROOT.TMath.Gradient(6, squares, 1.0)) == [1.0, 2.0, 4.0, 6.0, 8.0, 9.0]
    assert list(ROOT.TMath.Laplacian(6, squares, 1.0)) == [2.0] * 6
    assert ROOT.TMath.Gradient(1, squares, 1.0) is None and ROOT.TMath.Laplacian(2, squares) is None
    assert ROOT.TMath.Gradient(6, squares, 0) is None


def test_gsystem_says_what_a_path_is_and_internal_gives_secure_random_bytes(tmp_path):
    import ctypes

    path = tmp_path / "f.txt"
    path.write_text("12345")
    stat = ROOT.FileStat_t()
    assert ROOT.gSystem.GetPathInfo(str(path), stat) == 0 and stat.fSize == 5
    assert not stat.fIsLink and stat.fMtime > 0 and stat.fUrl == ""
    ident, size, flags, modified = (ctypes.c_long(), ctypes.c_long(), ctypes.c_long(),
                                    ctypes.c_long())
    assert ROOT.gSystem.GetPathInfo(str(tmp_path), ident, size, flags, modified) == 0
    assert flags.value & 2 and modified.value > 0
    assert ROOT.gSystem.GetPathInfo(str(tmp_path / "none"), stat) == 1
    buffer = bytearray(16)
    assert ROOT.Internal.GetCryptoRandom(buffer, 16) == 16 and any(buffer)
    with pytest.raises(AttributeError, match=r"ROOT has Internal\.Other"):
        ROOT.Internal.__getattr__("Other")
