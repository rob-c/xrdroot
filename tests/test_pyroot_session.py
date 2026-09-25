"""``gROOT``, ``gSystem``, ``gDirectory``, and the clocks: the session a script runs in."""

from __future__ import annotations

import os
import time

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh


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
    assert lines[0].startswith("Real time 0:00:0") and lines[0].endswith(", 2 slices")
    assert ", CP time " in lines[1] and len(lines[2].split(",")[0].split(":")[-1]) == 9
    assert watch.RealTime() >= 0 and watch.CpuTime() >= 0 and watch.Counter() == 2
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
    assert out[0].startswith("fill      : Real Time =   ") and "seconds Cpu Time" in out[0]
    assert out[-1].startswith("TOTAL     : Real Time =") and real >= 0 and cpu >= 0
    assert ROOT.gBenchmark.GetRealTime("none") == 0.0 and ROOT.gBenchmark.GetCpuTime("none") == 0


def test_a_datime_is_a_date_and_time_to_the_second(capsys):
    when = ROOT.TDatime(2026, 9, 25, 12, 30, 5)
    assert (when.GetDate(), when.GetTime()) == (20260925, 123005)
    assert (when.GetYear(), when.GetMonth(), when.GetDay()) == (2026, 9, 25)
    assert (when.GetHour(), when.GetMinute(), when.GetSecond()) == (12, 30, 5)
    assert when.GetDayOfWeek() == 5 and when.AsSQLString() == "2026-09-25 12:30:05"
    same = ROOT.TDatime(20260925, 123005)
    assert same == when and hash(same) == hash(when) and same != "x"
    assert ROOT.TDatime(when.Convert()) == when
    when.Print()
    assert capsys.readouterr().out == "Date/Time = Fri Sep 25 12:30:05 2026\n"
    assert abs(ROOT.TDatime().Convert() - time.time()) < 5
    stamp = ROOT.TTimeStamp()
    assert stamp.AsDouble() == float(stamp.GetSec())


def test_gsystem_says_a_path_is_missing_the_way_root_does(tmp_path):
    (tmp_path / "there.txt").write_text("x")
    assert ROOT.gSystem.AccessPathName("there.txt") is False
    assert ROOT.gSystem.AccessPathName("missing.txt") is True
    assert ROOT.gSystem.AccessPathName("there.txt", ROOT.kReadPermission) is False
    assert ROOT.gSystem.kFileExists == ROOT.kFileExists == 0


def test_gsystem_loads_and_includes_without_complaint():
    assert ROOT.gSystem.Load("libPhysics") == 0 and "libPhysics" in ROOT.gSystem.GetLibraries()
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
    assert ROOT.gSystem.GetDynamicPath() == "/x:/y"
    assert ROOT.gSystem.CompileMacro("m.C") == 1
    ROOT.gSystem.SetBuildDir("/tmp")


def test_gsystem_reads_and_sets_the_environment(monkeypatch):
    monkeypatch.delenv("PYROOT_TEST", raising=False)
    assert ROOT.gSystem.Getenv("PYROOT_TEST") is None
    ROOT.gSystem.Setenv("PYROOT_TEST", 5)
    assert ROOT.gSystem.Getenv("PYROOT_TEST") == "5"
    ROOT.gSystem.Unsetenv("PYROOT_TEST")
    assert "PYROOT_TEST" not in os.environ
    assert ROOT.gSystem.HostName() and ROOT.gSystem.GetPid() == os.getpid()
    assert ROOT.gSystem.GetUid() == os.getuid() and ROOT.gSystem.GetUserInfo() is None
    assert ROOT.gSystem.GetBuildArch() and ROOT.gSystem.Now() > 0


def test_gsystem_runs_commands_and_sleeps():
    assert ROOT.gSystem.Exec("true") == 0
    assert ROOT.gSystem.GetFromPipe("echo hi") == "hi"
    assert ROOT.gSystem.ProcessEvents() is False
    ROOT.gSystem.Sleep(1)
    with pytest.raises(SystemExit):
        ROOT.gSystem.Exit(0)
    with pytest.raises(SystemExit):
        ROOT.gSystem.Abort()


def test_gsystem_walks_and_names_paths(tmp_path):
    here = os.getcwd()
    assert ROOT.gSystem.pwd() == ROOT.gSystem.WorkingDirectory() == here
    assert ROOT.gSystem.mkdir("a/b", True) == 0 and ROOT.gSystem.mkdir("a") == -1
    assert ROOT.gSystem.MakeDirectory("c") == 0
    assert ROOT.gSystem.cd("a") and not ROOT.gSystem.ChangeDirectory("nowhere")
    os.chdir(here)
    assert ROOT.gSystem.BaseName("/x/y.root") == "y.root" and ROOT.gSystem.BaseName("/") == "/"
    assert ROOT.gSystem.DirName("/x/y.root") == "/x" and ROOT.gSystem.DirName("y") == "."
    assert ROOT.gSystem.DirName("/y") == "/" and ROOT.gSystem.GetDirName("/x/y") == "/x"
    assert ROOT.gSystem.ConcatFileName("a", "b") == os.path.join("a", "b")
    name = ROOT.TString("f.root")
    assert ROOT.gSystem.PrependPathName("dir", name) == "dir/f.root" and name == "dir/f.root"
    assert ROOT.gSystem.IsAbsoluteFileName("/x") and ROOT.gSystem.UnixPathName("a") == "a"
    assert ROOT.gSystem.HomeDirectory() == os.path.expanduser("~")
    assert os.path.isdir(ROOT.gSystem.TempDirectory())


def test_gsystem_expands_and_finds_files(tmp_path, monkeypatch):
    monkeypatch.setenv("PYROOT_DIR", "there")
    assert ROOT.gSystem.ExpandPathName("$PYROOT_DIR/x") == "there/x"
    text = ROOT.TString("$PYROOT_DIR")
    assert ROOT.gSystem.ExpandPathName(text) is False and text == "there"
    (tmp_path / "found.txt").write_text("x")
    assert ROOT.gSystem.Which(f"/nowhere:{tmp_path}", "found.txt") == str(tmp_path / "found.txt")
    assert ROOT.gSystem.Which(".", "missing.txt") is None
    assert ROOT.gSystem.Which("", str(tmp_path / "found.txt")) == str(tmp_path / "found.txt")
    assert ROOT.gSystem.Which("", "/no/such/file") is None
    assert ROOT.gSystem.Which("/nowhere", "sh", ROOT.kExecutePermission)
    assert ROOT.gSystem.FindFile(str(tmp_path), "found.txt")
    assert ROOT.gSystem.IsFileInIncludePath("found.txt")


def test_gsystem_moves_and_removes_files(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    assert (
        ROOT.gSystem.CopyFile("a.txt", "b.txt") == 0
        and ROOT.gSystem.CopyFile("a.txt", "b.txt") == -1
    )
    assert ROOT.gSystem.CopyFile("a.txt", "b.txt", True) == 0
    assert ROOT.gSystem.Rename("b.txt", "c.txt") == 0 and ROOT.gSystem.Rename("nope", "d") == -1
    assert ROOT.gSystem.Unlink("c.txt") == 0 and ROOT.gSystem.Unlink("c.txt") == -1
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
    assert names == [".", "..", "one"]
    assert ROOT.gSystem.OpenDirectory("nowhere") is None and ROOT.gSystem.GetDirEntry(None) is None
