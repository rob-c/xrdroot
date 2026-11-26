---
title: "A file nobody closed is a `ResourceWarning` Python 3.13 attributes to your test"
kind: "gotcha"
summary: "A dropped `ROOTFile` leaves an open handle whose `ResourceWarning` Python 3.13 attributes to whatever test is running; `Source.__del__` closes the handle it owns (not a forked or caller's one) and a test pins it."
tags: [file-io, resourcewarning, pytest, python-313, ci]
created: "2026-11-26T17:42:43.423Z"
updated: "2026-11-26T17:42:43.423Z"
---
A `ROOTFile` nobody closes leaves a raw `FileIO` for Python's finaliser, and Python's finaliser of an open file is a `ResourceWarning` — which this suite turns into an error, in whichever test happens to be running when the collector answers. Python 3.13 finalises in a different order, so CI's 3.13 job failed on another test entirely while 3.10–3.12 stayed green.

## Where it bites

A `pyunittest`/`distrdf` test builds a frame, and the frame reads the files it was built from **long after the test that built it has ended**. Nothing closes them. On `d3b39e1c`'s CI:

```text
FAILED tests/test_pyroot_distrdf_graph.py::test_a_dataset_spec_is_cut_like_a_chain_of_its_samples
  - ResourceWarning: unclosed file <_io.FileIO name='.../a.root' mode='rb'>
```

The test itself is innocent, and locally (Python 3.11) it passes under `pytest -W error::ResourceWarning`, so it cannot be reproduced by reading that test.

## The fix, and the caveat it must keep

`src/xrdroot/file.py::Source.__del__` closes the handle **it owns**:

```python
    def __del__(self) -> None:
        handle = getattr(self, "handle", None)
        if self.owned and handle is not None and handle is not self._inherited:
            handle.close()
```

`owned` is False for a file object a caller handed in (the caller closes it), and `_inherited` is the handle a **fork** left behind, which is the parent's — `_adopt` says so in as many words, and closing it would write a `kXR_endsess` the parent never asked for.

`tests/test_root_directories.py::test_a_file_nobody_closed_lets_go_of_its_handle_quietly` pins it: drop a `ROOTFile`, `gc.collect()`, assert `handle.closed` and no `ResourceWarning` recorded.

## How to check your own case

```bash
export PYTHONPATH=$PWD/src
.venv/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py -W error::ResourceWarning
```

A warning that only appears in a full run is a *collection-order* effect: `xrdroot.file.OPENED` is a `WeakValueDictionary`, so a `ROOTFile` can be collected before any fixture sweeps it, and the sweep in `tests/test_pyroot_distrdf_graph.py::_close_the_files_the_frames_opened` only sees what is still alive.
