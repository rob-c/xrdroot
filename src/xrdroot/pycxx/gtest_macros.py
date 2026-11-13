"""GoogleTest's macros, as C++ the translator already reads: what ``gtest/gtest.h`` defines.

A unit test is a file of ``TEST(Suite, Name) { ... }`` blocks with
``EXPECT_EQ(a, b) << "why";`` inside them. Those are macros in GoogleTest
too, and they are defined here the way GoogleTest defines them - a test is
a class deriving from its fixture, registered by a static initialiser,
whose ``TestBody`` is the block that follows - so that nothing past the
preprocessor needs to know GoogleTest exists. The macros call into
``::testing::internal``, which :mod:`xrdroot.pycxx.runtime.gtest` provides.

An assertion is ``if (check) ; else <failure>``, as in GoogleTest, so that
the ``<< message`` written after it streams onto the failure it reports,
and an ``ASSERT_`` returns from the function it is in, as GoogleTest's
does. A dialect whose library has test macros of its own adds them to these,
as a prelude of its headers (:class:`~xrdroot.pycxx.dialect.Prelude`).

The typed tests, which instantiate one body for every type of a list, are
refused by name: the Python a body becomes has no type to vary.
"""

from __future__ import annotations

__all__ = ["HEADERS", "PRELUDE", "REFUSED"]

#: The headers that make a file a GoogleTest program: including any defines the macros.
HEADERS = frozenset({"gtest/gtest.h", "gmock/gmock.h"})

#: The comparisons of two values, each an ``EXPECT_`` and an ``ASSERT_`` macro.
COMPARISONS = (
    "EQ", "NE", "LT", "LE", "GT", "GE",
    "STREQ", "STRNE", "STRCASEEQ", "STRCASENE", "FLOAT_EQ", "DOUBLE_EQ",
)  # fmt: skip

#: Each kind of assertion: its arguments, and the check of ``::testing::internal`` it makes.
CHECKS = {
    "TRUE": ("(c)", "IsTrue(c, #c, __FILE__, __LINE__, {fatal})"),
    "FALSE": ("(c)", "IsFalse(c, #c, __FILE__, __LINE__, {fatal})"),
    "NEAR": ("(a, b, e)", "Near(a, b, e, #a, #b, #e, __FILE__, __LINE__, {fatal})"),
    "THROW": (
        "(s, x)",
        "Throws([&]() {{ s; }}, #x, #s, __FILE__, __LINE__, {fatal})",
    ),
    "NO_THROW": ("(s)", "NoThrow([&]() {{ s; }}, #s, __FILE__, __LINE__, {fatal})"),
    "ANY_THROW": ("(s)", "AnyThrow([&]() {{ s; }}, #s, __FILE__, __LINE__, {fatal})"),
    "THAT": ("(v, m)", "That(v, m, #v, #m, __FILE__, __LINE__, {fatal})"),
    **{
        kind: ("(a, b)", f'Compare("{kind}", a, b, #a, #b, __FILE__, __LINE__, {{fatal}})')
        for kind in COMPARISONS
    },
}

#: The macros that are neither a test nor an assertion of two values.
FIXED = r"""
#define GTEST_NONFATAL_(check) if (check) ; else ::testing::internal::Pending()
#define GTEST_FATAL_(check) if (check) ; else return ::testing::internal::Pending()
#define GTEST_TEST_(suite, name, parent, registration) \
   class suite##_##name##_Test : public parent { public: void TestBody() override; }; \
   static int suite##_##name##_registered = \
      ::testing::internal::registration(#suite, #name, __FILE__, __LINE__); \
   void suite##_##name##_Test::TestBody()
#define TEST(suite, name) GTEST_TEST_(suite, name, ::testing::Test, Register)
#define GTEST_TEST(suite, name) TEST(suite, name)
#define TEST_F(fixture, name) GTEST_TEST_(fixture, name, fixture, Register)
#define TEST_P(fixture, name) GTEST_TEST_(fixture, name, fixture, RegisterPattern)
#define INSTANTIATE_TEST_SUITE_P(prefix, suite, ...) \
   static int gtest_##prefix##_##suite##_instantiated = \
      ::testing::internal::Instantiate(#prefix, #suite, __FILE__, __LINE__, __VA_ARGS__)
#define INSTANTIATE_TEST_CASE_P(prefix, suite, ...) \
   INSTANTIATE_TEST_SUITE_P(prefix, suite, __VA_ARGS__)
#define GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(suite)
#define FRIEND_TEST(suite, name) friend class suite##_##name##_Test
#define RUN_ALL_TESTS() ::testing::internal::RunAllTests()
#define FAIL() GTEST_FATAL_(::testing::internal::Fail(__FILE__, __LINE__, true))
#define GTEST_FAIL() FAIL()
#define ADD_FAILURE() GTEST_NONFATAL_(::testing::internal::Fail(__FILE__, __LINE__, false))
#define ADD_FAILURE_AT(file, line) GTEST_NONFATAL_(::testing::internal::Fail(file, line, false))
#define SUCCEED() GTEST_NONFATAL_(true)
#define GTEST_SUCCEED() SUCCEED()
#define GTEST_SKIP() GTEST_FATAL_(::testing::internal::Skip(__FILE__, __LINE__))
#define SCOPED_TRACE(message) ::testing::internal::Trace(__FILE__, __LINE__, message)
#define GTEST_DEATH_(s) GTEST_FATAL_(::testing::internal::Death(#s, __FILE__, __LINE__))
#define EXPECT_DEATH(s, regex) GTEST_DEATH_(s)
#define ASSERT_DEATH(s, regex) GTEST_DEATH_(s)
#define EXPECT_DEATH_IF_SUPPORTED(s, regex) GTEST_DEATH_(s)
#define ASSERT_DEATH_IF_SUPPORTED(s, regex) GTEST_DEATH_(s)
#define EXPECT_DEBUG_DEATH(s, regex) GTEST_DEATH_(s)
#define ASSERT_DEBUG_DEATH(s, regex) GTEST_DEATH_(s)
#define EXPECT_EXIT(s, exited, regex) GTEST_DEATH_(s)
#define ASSERT_EXIT(s, exited, regex) GTEST_DEATH_(s)
#define GTEST_MATCHER_(name, description, params, arguments, values) \
   static bool name##_gmock_matches(const auto &arg GTEST_EXPAND_ params, \
                                    ::testing::internal::Listener *result_listener); \
   ::testing::internal::Matcher name(GTEST_STRIP_ params) { \
      return ::testing::internal::MakeMatcher( \
         [=](const auto &arg, ::testing::internal::Listener *result_listener) { \
            return name##_gmock_matches(arg GTEST_EXPAND_ arguments, result_listener); }, \
         #name, description GTEST_EXPAND_ values); } \
   static bool name##_gmock_matches(const auto &arg GTEST_EXPAND_ params, \
                                    ::testing::internal::Listener *result_listener)
#define GTEST_EXPAND_(...) __VA_ARGS__
#define GTEST_STRIP_(first, ...) __VA_ARGS__
#define MATCHER(name, description) GTEST_MATCHER_(name, description, (), (), ())
#define MATCHER_P(name, p0, description) \
   GTEST_MATCHER_(name, description, (, const auto &p0), (, p0), (, p0))
#define MATCHER_P2(name, p0, p1, description) \
   GTEST_MATCHER_(name, description, (, const auto &p0, const auto &p1), (, p0, p1), (, p0, p1))
"""


def _assertions() -> list[str]:
    """``EXPECT_EQ(a, b)``, ``ASSERT_EQ(a, b)`` and the rest: a ``#define`` for each."""
    lines = []
    for kind, (params, check) in CHECKS.items():
        for prefix, fatal in (("EXPECT", "false"), ("ASSERT", "true")):
            kept = "FATAL" if fatal == "true" else "NONFATAL"
            body = f"GTEST_{kept}_(::testing::internal::{check.format(fatal=fatal)})"
            lines.append(f"#define {prefix}_{kind}{params} {body}")
    return lines


#: Everything ``gtest/gtest.h`` defines, as ``#define`` lines.
PRELUDE = FIXED + "\n".join(_assertions()) + "\n"

#: The typed tests: one body, run once for each type of a ``::testing::Types`` list.
_TYPED = (
    "{name}, a GoogleTest typed test, instantiates its body once for each type of a "
    "::testing::Types list, and its Python translation has no types to vary: it is "
    "not translated"
)

#: The macros refused wherever they are used, each with the sentence that says why.
REFUSED = {
    name: _TYPED.format(name=name)
    for name in (
        "TYPED_TEST",
        "TYPED_TEST_SUITE",
        "TYPED_TEST_CASE",
        "TYPED_TEST_P",
        "TYPED_TEST_SUITE_P",
        "TYPED_TEST_CASE_P",
        "REGISTER_TYPED_TEST_SUITE_P",
        "REGISTER_TYPED_TEST_CASE_P",
        "INSTANTIATE_TYPED_TEST_SUITE_P",
        "INSTANTIATE_TYPED_TEST_CASE_P",
    )
}
