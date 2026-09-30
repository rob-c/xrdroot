"""``std::mt19937`` and its distributions number for number as libc++'s, and the macro corners
ROOT 7's histogram tutorials reach: a thread made in a vector, a char chosen by ``?:``."""

from __future__ import annotations

from typing import Any

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from refmachine import roots
from xrdroot.cint.execute import run_source


def test_the_twister_gives_the_standards_ten_thousandth_word() -> None:
    engine = ROOT.std.mt19937()
    engine.discard(9999)
    seeded = ROOT.std.mt19937(1)
    reseeded = ROOT.std.mt19937(7)
    reseeded.seed(1)
    expect((engine(), 4123659995), (seeded(), reseeded()), (seeded(), 4282876139),
           ((ROOT.std.mt19937.min(), ROOT.std.mt19937.max()), (0, 4294967295)))  # fmt: skip


def test_the_distributions_draw_what_libcxx_draws() -> None:
    """The values ROOT 6.40 printed, with ``%.17g``, drawing the same from a default twister.

    The twister's words and the uniform doubles made of them are exact
    everywhere; a normal value goes through ``log``, which another C library
    may round an ulp otherwise, and so is held to 1e-15 of itself off ROOT's
    machine - a few ulps, where a wrong draw is wrong in the first digit.
    """
    engine = ROOT.std.mt19937()
    normal = ROOT.std.normal_distribution["double"](5.0, 2.0)
    drawn = [normal(engine) for _ in range(3)]
    uniform = ROOT.std.uniform_real_distribution(0.0, 1.0)
    expect((drawn, roots([4.7072364376205549, 5.2690593169446558, 1.2572313791787946], rel=1e-15)),
           (uniform(engine), 0.1883819760471811),
           ((normal.mean(), normal.stddev(), uniform.a(), uniform.b()), (5.0, 2.0, 0.0, 1.0)),
           (ROOT.std.uniform_real_distribution["double"] is ROOT.std.uniform_real_distribution,
            True))  # fmt: skip
    normal.reset()
    uniform.reset()


def test_a_vector_of_threads_starts_each_from_its_function(capsys: Any) -> None:
    macro = """
    void t() {
       std::vector<std::thread> threads;
       for (int i = 0; i < 2; i++) threads.emplace_back([i] { printf("%d\\n", i * 0); });
       for (auto &&t : threads) t.join();
       int v = 3;
       std::cout << (v > 2 ? '*' : ' ') << (v > 5 ? 'x' : '-') << (v > 2 ? 1 : 2.5) << "\\n";
    }
    """
    run_source(macro, "t.C")
    assert capsys.readouterr().out == "0\n0\n*-1\n"
    words = ROOT.std.vector["std::string"]()
    words.emplace_back("a")
    assert list(words) == ["a"]
