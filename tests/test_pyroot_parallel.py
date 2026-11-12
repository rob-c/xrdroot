"""``TThreadExecutor``, ``TProcessExecutor`` and the objects ROOT runs work on."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT


def negate(value: int) -> int:
    """A free function: what a process pool needs, unlike a lambda."""
    return -value


def test_a_sequence_is_what_root_hands_an_executor():
    # ROOT::TSeqI is the range ROOT walks; the executors take it as they take any iterable
    assert list(ROOT.TSeqI(4)) == [0, 1, 2, 3]
    assert list(ROOT.TSeqI(1, 7, 2)) == [1, 3, 5]
    assert ROOT.TThreadExecutor(1).Map(abs, ROOT.TSeqI(3)) == [0, 1, 2]
    # TSeqI(seeds): another sequence is walked as it is, as the C++ copy does
    assert list(ROOT.TSeqI(ROOT.TSeqI(3))) == [0, 1, 2]
    # a float bound is converted the way C++ converts it
    assert list(ROOT.TSeqI(2.9)) == [0, 1]


def test_a_thread_pool_maps_in_argument_order():
    with ROOT.TThreadExecutor(3) as pool:
        assert pool.Map(negate, ROOT.TSeqI(4)) == [0, -1, -2, -3]
    # one worker and a single argument are the two short roads through Map
    assert ROOT.TThreadExecutor(1).Map(negate, [2]) == [-2]
    assert ROOT.TThreadExecutor(2).Map(negate, 3) == [-3]
    # a string is one argument, not a sequence to walk
    assert ROOT.TThreadExecutor(2).Map(len, "xyz") == [3]


def test_a_mapping_over_tuples_hands_them_to_the_callable_as_arguments():
    # max takes two arguments: a tuple element is passed as them
    assert ROOT.TThreadExecutor(2).Map(max, [(1, 2), (3, 1)]) == [2, 3]
    # a one-argument callable receives the tuple itself
    assert ROOT.TThreadExecutor(1).Map(tuple, [(1, 2)]) == [(1, 2)]
    # a callable of unknown signature is handed the element whole
    assert ROOT.TThreadExecutor(1).Map(len, [(1, 2, 3)]) == [3]


def test_map_reduce_folds_the_answers_of_the_pool():
    pool = ROOT.TThreadExecutor(2)
    assert pool.MapReduce(negate, ROOT.TSeqI(3), lambda a, b: a + b) == -3
    # the two-stage form folds the per-worker partials: one worker, one partial
    assert pool.MapReduce(negate, ROOT.TSeqI(3), lambda a, b: a + b, lambda a, b: a * 2) == -3
    assert pool.MapReduce(negate, [], lambda a, b: a + b) is None
    assert pool.Foreach(negate, ROOT.TSeqI(2)) is None


def test_submit_hands_back_a_handle_whose_get_is_the_answer():
    pool = ROOT.TThreadExecutor(2)
    handle = pool.Submit(negate, 5)
    assert handle.get() == -5
    assert handle.wait(5) is True
    assert "submitted" in repr(handle)
    one = ROOT.TThreadExecutor(1)
    assert one.Submit(negate, 6).get() == -6


def test_a_process_pool_runs_free_functions_in_processes():
    pool = ROOT.TProcessExecutor(2)
    assert pool.Map(negate, ROOT.TSeqI(4)) == [0, -1, -2, -3]
    # a lambda cannot be pickled to a worker: the answers still come back
    assert pool.Map(lambda value: value + 1, [1, 2]) == [2, 3]
    assert ROOT.TProcessExecutor(1).Map(negate, [5]) == [-5]
    assert ROOT.TProcessExecutor(2).Map(negate, 7) == [-7]
    pool.Close()
    pool.Close()  # closing twice is closing a closed pool


def test_reducers_fold_histograms_numbers_and_collections():
    histogram = ROOT.TH1F("h", "h", 4, 0, 1)
    histogram.Fill(0.5)
    reducer = ROOT.ExecutorUtils.ReduceObjects["TH1F*"]()
    assert "TH1F" in repr(reducer)
    merged = reducer(histogram, histogram)
    assert merged.GetEntries() == 2.0
    assert reducer(None, histogram) is histogram
    assert reducer(histogram, None) is histogram
    assert ROOT.ExecutorUtils.ReduceObjects()()(2, 3) == 5
    assert "default" in repr(ROOT.ExecutorUtils.ReduceObjects()())
    assert ROOT.ExecutorUtils.ReduceVectors()([1], [2, 3]) == [1, 2, 3]


def test_a_threaded_object_gives_each_thread_its_own_and_merges_them():
    threaded = ROOT.TThreadedObject["TH1F"]("h", "t", 10, 0, 1)
    mine = threaded.Get()
    mine.Fill(0.5)
    assert threaded.Get() is mine
    snapshot = threaded.SnapshotMerge()
    assert snapshot.GetEntries() == 1.0
    assert threaded.Merge().GetEntries() == 1.0
    assert "1 thread" in repr(threaded)


def test_a_threaded_object_of_nothing_says_what_it_needs():
    with pytest.raises(AttributeError, match="TThreadedObject needs the class it holds"):
        ROOT.TThreadedObject().Get()
    with pytest.raises(AttributeError, match="ROOT has no class"):
        ROOT.TThreadedObject["NotAClass"]().Get()


def test_a_threaded_object_without_a_clone_merges_with_plus():
    threaded = ROOT.TThreadedObject["TH1F"]("h", "t", 10, 0, 1)
    threaded._per_thread = {1: 2, 2: 3}
    assert threaded.Merge() == 5
    threaded._per_thread = {}
    # nothing filled yet: ROOT still answers with an object, and it is empty
    assert threaded.Merge().GetEntries() == 0.0


def test_a_task_runs_itself_and_the_tasks_added_to_it():
    ran: list[str] = []

    class Recording(ROOT.TTask):
        def ExecuteTask(self, task, option=""):
            ran.append(task.GetName())

    parent = Recording("geant3", "Geant3 simulation main program")
    child = Recording("uginit", "Initialisation manager")
    parent.Add(child)
    assert child.GetParent() is parent
    assert parent.GetTitle() == "Geant3 simulation main program"
    parent.SetName("g3")
    parent.SetTitle("renamed")
    assert (parent.GetName(), parent.GetTitle()) == ("g3", "renamed")
    assert parent.GetOption() == ""
    parent.Run()
    assert ran == ["g3", "uginit"]
    assert [task.GetName() for task in parent.GetListOfTasks()] == ["uginit"]
    parent.CleanTasks()
    assert parent.GetListOfTasks() == []
    assert "TTask" in repr(parent)
    assert ROOT.TTask().Run() is None


def test_a_pool_task_base_gives_the_signature_a_tutorial_overrides():
    base = ROOT.TThreadPoolTaskImp["TTestTask", "int"]
    assert base().Process(1) is None
    assert base().GetTaskName() == "TThreadPoolTaskImp[TTestTask, int]"

    class Mine(base):
        def Process(self, argument):
            return argument * 2

    assert Mine().Process(21) == 42
    plain = ROOT.TThreadPoolTaskImp["single"]
    assert plain().Process() is None


def test_the_window_system_is_batch_and_says_so():
    system = ROOT.gVirtualX
    assert system.InheritsFrom("TVirtualX")
    assert not system.InheritsFrom("TGCocoa")
    assert system.IsBatch() and system.GetName() == "TVirtualX"
    assert "batch" in system.GetTitle().lower()
    # nothing is shown, so every call that would touch a window does nothing
    assert system.SetDoubleBuffer() is None
    assert system.MapWindow(1) is None
    assert system.Flush() is None
    assert "TVirtualX" in repr(system)


def test_pyconfig_keeps_the_switches_a_script_sets():
    config = ROOT.PyConfig
    assert config.IgnoreCommandLineOptions is False
    config.IgnoreCommandLineOptions = True
    assert config.IgnoreCommandLineOptions is True
    assert config.DisableRootLogon is False
    assert config.EnableTPython is False
    assert "PyConfig" in repr(config)
    config.IgnoreCommandLineOptions = False


def test_a_varargs_callable_is_handed_the_element_whole():
    def collect(*values):
        return values

    # a tuple element is the call's arguments, so varargs receives them unpacked
    assert ROOT.TThreadExecutor(1).Map(collect, [(1, 2)]) == [(1, 2)]


def test_one_workers_piece_of_work_is_the_callable_and_its_argument():
    # the piece a process pool sends is executed the way a worker executes it,
    # which is checked here as well because a child's work is its own process
    from xrdroot.pyroot.core.parallel import _spawn_call

    assert _spawn_call((negate, 4)) == -4
    assert _spawn_call((max, (1, 2))) == 2


def test_a_threaded_object_of_a_class_name_that_is_not_a_string_keeps_the_base():
    typed = ROOT.TThreadedObject["TH1F"]("h", "t", 1, 0, 1)
    assert ROOT.TThreadedObject[42] is type(typed).__mro__[1]


def test_a_root_thread_runs_its_work_and_join_waits_for_it():
    done: list[int] = []
    thread = ROOT.TThread("h1", lambda value: done.append(value), 1)
    assert thread.GetName() == "h1"
    assert thread.Join() == 0
    assert done == [1]
    assert thread.IsRunning() is False
    assert thread.GetId() > 0
    assert "h1" in repr(thread)


def test_run_starts_the_work_and_a_second_run_leaves_it_alone():
    done: list[int] = []

    def work(value: int) -> None:
        done.append(value)

    thread = ROOT.TThread("h1", work, 1)   # the constructor starts it, as ROOT's does
    thread.Run()                            # so Run only starts a thread that has not started
    thread.Join()
    assert done == [1]
    ROOT.TThread.Printf("thread %ld here", 3)
    started = ROOT.TThread("late")
    started._work = work
    started._arguments = (2,)
    started.Run()
    started.Join()
    assert done == [1, 2]


def test_a_thread_that_raised_raises_it_from_join():
    def explode() -> None:
        raise RuntimeError("the work failed")

    thread = ROOT.TThread("bad", explode)
    with pytest.raises(RuntimeError, match="the work failed"):
        thread.Join()
    # the error is reported once, as ROOT's own thread reports it once
    assert thread.Join() == 0


def test_a_thread_with_no_work_is_never_running_and_can_be_killed():
    idle = ROOT.TThread("idle")
    assert idle.IsRunning() is False
    assert idle.GetId() == 0
    assert idle.Join() == 0          # a thread that never started has nothing to wait for
    assert idle.Kill() == 0
    assert ROOT.TThread.SelfId() > 0
    ROOT.TThread.Sleep(1)            # one number: microseconds
    ROOT.TThread.Sleep(0, 1)         # a range of them
    ROOT.TThread.Sleep(5, 1)         # nothing to wait for
    ROOT.TThread.Ps()


def test_a_thread_pool_processes_what_was_pushed_onto_it():
    processed: list[int] = []

    class Task:
        def Process(self, argument: int) -> None:
            processed.append(argument)

    pool = ROOT.TThreadPool["Task", "int"](2, True)
    assert pool.GetPoolSize() == 2
    assert "TThreadPool" in repr(pool)
    pool.Drain()  # nothing queued: nothing to do
    pool.PushTask(Task(), 1)
    pool.PushTask(Task(), 2)
    pool.Drain()
    assert sorted(processed) == [1, 2]
    pool.PushTask(Task(), 3)
    pool.Drain()                     # the pool exists now: the same threads run it
    assert sorted(processed) == [1, 2, 3]
    pool.Drain()                     # queued nothing, and the pool is already there
    pool.Stop(wait=False)            # stop without joining the workers
    pool.Stop()
    assert ROOT.TThreadPool().GetPoolSize() >= 1
    assert ROOT.TThreadPool["Task"]().GetPoolSize() >= 1
    assert ROOT.TThreadPool["Task", "int"].__name__ == "TThreadPool[Task, int]"
