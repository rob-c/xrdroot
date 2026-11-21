

def _window_reads(source, window, requests):
    """What each request returns, and every ``read`` the source was asked for on the way."""
    reads, answers = [], []
    real_read = type(source).read

    def counting(self, offset, size):
        reads.append((offset, size))
        return real_read(self, offset, size)

    type(source).read = counting
    try:
        for offset, size in requests:
            answers.append(source.window(offset, size, window))
    finally:
        type(source).read = real_read
    return answers, reads


def test_a_read_ahead_window_is_shared_and_read_once(tmp_path):
    """Every branch of a tree reads a window of its own today, and a tree's flushes interleave
    the branches' baskets - so forty branches read the same megabytes forty times: 41 MB out of
    a 3 MB file, measured. One window kept per source is read once and shared.

    The window is four times the run it serves, so a request never straddles two of them; one
    that would is read as it asks, because two windows is more than the read is worth.
    """
    from xrdroot.file import WINDOWS_KEPT, Source

    data = bytes(index % 251 for index in range(10_000))     # a period no window divides
    path = tmp_path / "w.root"
    path.write_bytes(data)
    with path.open("rb") as handle:
        source = Source(handle, str(path), owned=False)
        answers, reads = _window_reads(source, 1000, [(500, 10), (600, 10), (1500, 10)])
        assert answers == [data[500:510], data[600:610], data[1500:1510]]
        assert reads == [(0, 1000), (1000, 1000)]            # one read a window, shared
        more, extra = _window_reads(source, 1000, [(950, 100)])
        assert more == [data[950:1050]] and extra == [(950, 100)]   # straddles: read as asked
        _window_reads(source, 1000, [(start, 1) for start in range(0, 9000, 1000)])
        assert len(source._windows) == WINDOWS_KEPT          # only a few are kept
        assert len(source._window_order) == WINDOWS_KEPT


