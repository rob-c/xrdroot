"""GoogleTest programs run with the translator alone: GoogleTest is C++, not any library's."""

from __future__ import annotations

from pathlib import Path

import pytest

from xrdroot.pycxx import Unsupported, run_program
from xrdroot.pycxx.runtime.gtest import provide, runner, testing
from xrdroot.pycxx.runtime.gtest.exceptions import CAUGHT, REFUSAL, adopt, caught, internal

#: A unit test of a class of its own, with a fixture, a parameterised test and a failure.
TESTS = r"""
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

class Counter {
public:
   void add(int n) { if (n < 0) throw std::invalid_argument("negative"); fTotal += n; }
   int total() const { return fTotal; }
private:
   int fTotal = 0;
};

class CounterTest : public ::testing::Test {
protected:
   void SetUp() override { counter.add(2); }
   Counter counter;
};

TEST(Counter, StartsAtZero) { Counter c; EXPECT_EQ(c.total(), 0); }
TEST_F(CounterTest, AddsWhatItIsGiven) { counter.add(3); ASSERT_EQ(counter.total(), 5); }
TEST(Counter, RefusesANegative) { Counter c; EXPECT_THROW(c.add(-1), std::invalid_argument); }

class Halves : public ::testing::TestWithParam<int> {};
TEST_P(Halves, AreSmaller) { EXPECT_LT(GetParam() / 2, GetParam()); }
INSTANTIATE_TEST_SUITE_P(Some, Halves, ::testing::Values(2, 4, 8));
"""


def _written(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_a_googletest_program_runs_its_tests_and_passes(tmp_path, capsys) -> None:
    path = _written(tmp_path, "counter_test.cxx", TESTS)
    assert run_program([path], use_cache=False) == 0
    out = capsys.readouterr().out
    assert "[==========] 6 tests from 3 test suites ran." in out
    assert "[  PASSED  ] 6 tests." in out


def test_a_failing_expectation_fails_the_program(tmp_path, capsys) -> None:
    source = "#include <gtest/gtest.h>\nTEST(Sums, AreWrong) { EXPECT_EQ(1 + 1, 3); }\n"
    assert run_program([_written(tmp_path, "wrong.cxx", source)], use_cache=False) == 1
    assert "[  FAILED  ] Sums.AreWrong" in capsys.readouterr().out


def test_a_part_of_googletest_the_runtime_lacks_is_refused_by_name() -> None:
    REFUSAL.clear()  # as no host has adopted a refusal of its own; the fixture puts it back
    with pytest.raises(Unsupported, match=r"^testing::Mock is a part of GoogleTest"):
        testing.Mock  # noqa: B018
    assert internal(Unsupported("gap")) and not internal(ValueError("v"))


def test_a_host_adopts_its_own_exceptions_once_each() -> None:
    class Gap(Exception):
        pass

    class Failure(Exception):
        pass

    REFUSAL.clear()  # the fixture puts back what a host adopted before
    for _ in range(2):
        adopt(gaps=(Gap,), errors=(Failure,), caught={"lib::Error": "library"}, refused=Gap)
    assert REFUSAL == [Gap] and CAUGHT["lib::Error"] == "library"
    assert internal(Gap()) and caught(Failure(), "const lib::Error &")
    assert caught(RuntimeError(), "std::runtime_error") and not caught(Gap(), "std::runtime_error")
    with pytest.raises(Gap):
        testing.Mock  # noqa: B018
    adopt()


def test_a_host_adds_its_own_checks_and_what_ends_with_each_test(tmp_path, capsys) -> None:
    ended = []
    provide({"Remember": ended.append})
    runner.AFTER_TEST.append(lambda: ended.append("ended"))
    source = ("#include <gtest/gtest.h>\n"
              "TEST(Host, Checks) { ::testing::internal::Remember(1); }\n")  # fmt: skip
    assert run_program([_written(tmp_path, "host.cxx", source)], use_cache=False) == 0
    assert ended == [1, "ended"]


#: A typed test: one body, run for each type of its suite's list, its fixture a template.
TYPED = r"""
#include <gtest/gtest.h>
#include <cstddef>

#ifndef TYPED_TEST_SUITE
#define TYPED_TEST_SUITE TYPED_TEST_CASE
#endif

template <typename T>
class Sized : public testing::Test {
protected:
   T fZero = T();
   static constexpr std::size_t kBytes = sizeof(T);
};

using SizedTypes = testing::Types<char, short, double>;
TYPED_TEST_SUITE(Sized, SizedTypes);

TYPED_TEST(Sized, HalvesLikeItsType)
{
   TypeParam a = 7;
   EXPECT_EQ(a / 2, TypeParam(7) / TypeParam(2));
   EXPECT_EQ(this->fZero, 0);
}

TYPED_TEST(Sized, KnowsItsSize)
{
   static constexpr std::size_t kSeen = sizeof(TypeParam);
   EXPECT_EQ(kSeen, TestFixture::kBytes);
   EXPECT_LE(sizeof(TypeParam), 2u);
}
"""


def test_a_typed_test_runs_once_for_each_type_of_its_list(tmp_path, capsys) -> None:
    assert run_program([_written(tmp_path, "typed_test.cxx", TYPED)], use_cache=False) == 1
    out = capsys.readouterr().out
    assert "[==========] 6 tests from 3 test suites ran." in out
    assert "[       OK ] Sized/1.KnowsItsSize" in out
    assert "[  FAILED  ] Sized/2.KnowsItsSize" in out and "[  PASSED  ] 5 tests." in out


def test_a_typed_test_suite_runs_for_the_types_it_is_instantiated_with(tmp_path, capsys) -> None:
    source = r"""
    #include <gtest/gtest.h>
    template <typename T> class Pair_Of : public testing::Test {};
    TYPED_TEST_SUITE_P(Pair_Of);
    TYPED_TEST_P(Pair_Of, Adds_Up) { TypeParam a = 3; EXPECT_EQ(a + a, TypeParam(6)); }
    TYPED_TEST_P(Pair_Of, Halves) { EXPECT_EQ(TypeParam(3) / 2, TypeParam(1.5)); }
    REGISTER_TYPED_TEST_SUITE_P(Pair_Of, Adds_Up, Halves);
    typedef testing::Types<int, double> Numbers;
    INSTANTIATE_TYPED_TEST_SUITE_P(My, Pair_Of, Numbers);
    """
    assert run_program([_written(tmp_path, "suite_test.cxx", source)], use_cache=False) == 0
    out = capsys.readouterr().out
    assert "[       OK ] My/Pair_Of/0.Adds_Up" in out and "[       OK ] My/Pair_Of/1.Halves" in out
    assert "[  PASSED  ] 4 tests." in out
