"""Templates are instantiated for each set of arguments, as C++ instantiates them.

Each program's expected output is what ROOT 6.40.04's ``root.exe`` printed for
the same C++ on macOS ARM64.
"""

from __future__ import annotations

import pytest

from xrdroot.pycxx import Refusal, run_source, translate
from xrdroot.pycxx.instantiate import mangled
from xrdroot.pycxx.runtime import uninstantiated

#: A class template, its partial and explicit specialisations, a value parameter with a
#: default, and static members that each instance has its own of.
CLASSES = r"""
#include <cstdio>
#include <limits>
#include <typeinfo>

template <typename T> struct Box {
   T value;
   static int count;
   Box() : value() { ++count; }
   Box(T v) : value(v) { ++count; }
   T half() const { return value / 2; }
   double eps() const { return std::numeric_limits<T>::epsilon(); }
   unsigned long bytes() const { return sizeof(T); }
   T made() const { T x; x = 3; return x * 2; }
};
template <typename T> int Box<T>::count = 0;

template <typename T> struct Box<T *> {
   T *ptr;
   Box(T *p) : ptr(p) {}
   int kind() const { return 1; }
};

template <> struct Box<char> {
   int kind() const { return 2; }
};

template <typename T, int N = 3> struct Grid {
   T cells[N];
   int size() const { return N; }
   T sum() const { T s = T(); for (int i = 0; i < N; ++i) s += cells[i]; return s; }
};

struct Derived : Box<int> {
   Derived() : Box<int>(9) {}
};

int classes() {
   Box<int> a(7);
   Box<double> b(7);
   Box<int> c;
   printf("%d %g %d %d\n", a.half(), b.half(), Box<int>::count, Box<double>::count);
   printf("%g %g %lu %lu\n", a.eps(), b.eps(), a.bytes(), b.bytes());
   printf("%d %g %d\n", a.made(), b.made(), c.value);
   int x = 5;
   Box<int *> p(&x);
   Box<char> q;
   printf("%d %d %d\n", p.kind(), q.kind(), *p.ptr);
   Grid<double> g;
   for (int i = 0; i < 3; ++i) g.cells[i] = i + 0.5;
   Grid<int, 4> h;
   for (int i = 0; i < 4; ++i) h.cells[i] = i;
   printf("%d %g %d %d\n", g.size(), g.sum(), h.size(), h.sum());
   Derived d;
   printf("%d %d %lu\n", d.half(), Box<int>::count, sizeof(Grid<double>));
   printf("%s\n", typeid(a).name());
   return 0;
}
"""


def test_each_class_template_instance_is_a_class_of_its_own(capsys) -> None:
    run_source(CLASSES, "classes.C")
    assert capsys.readouterr().out == (
        "3 3.5 2 1\n0 2.22045e-16 4 8\n6 6 0\n1 2 5\n3 4.5 4 6\n4 3 24\n3BoxIiE\n"
    )
    python = translate(CLASSES, "classes.C")
    assert "class Box__int:" in python and "class Box__double:" in python
    assert "_cxx_layout_ = ('Grid<double, 3>', (), (('cells', 'double', '', (3,)),))" in python
    assert "return idiv(self.value, 2)" in python and "return self.value / 2" in python


#: Function templates, called with their arguments given, deduced, or defaulted.
FUNCTIONS = r"""
#include <cstdio>
#include <limits>
#include <vector>
#include <type_traits>

template <typename T> T half(T x) { return x / 2; }
template <typename T> T relative(const T &x, const T &y) {
   T diff = x > y ? x - y : y - x;
   if (diff < std::numeric_limits<T>::epsilon()) return diff;
   return diff / (x + y);
}
template <typename T> unsigned long width() { return sizeof(T) * 8; }
template <typename T, int N> T times(T x) { return N * x; }
template <unsigned N> unsigned fill(short (&out)[N]) {
   for (unsigned i = 0; i < N; ++i) out[i] = 10 + i;
   return N;
}
template <typename T> T total(const std::vector<T> &v) {
   T sum = T();
   for (auto x : v) sum += x;
   return sum;
}
template <typename T> const char *kind() {
   if constexpr (std::is_same_v<T, int>) return "int";
   else if constexpr (std::is_floating_point<T>::value) return "floating";
   else return "other";
}
template <typename T> struct Pair2 { T a, b; };
template <typename T> T first(const Pair2<T> &p) { return p.a; }
template <typename T = long> unsigned long bytes() { return sizeof(T); }
template <typename T> const char *name(T) { return "any"; }
template <> const char *name<double>(double) { return "double"; }

int functions() {
   printf("%d %g %g\n", half(7), half(7.0), relative(1.0f, 3.0f));
   printf("%lu %lu %lu\n", width<char>(), width<double>(), width<long double>());
   printf("%d %g\n", times<int, 3>(2), times<double, 2>(1.25));
   short a[3];
   unsigned n = fill(a);
   printf("%u %d %d\n", n, a[0], a[2]);
   std::vector<int> vi{1, 2, 3};
   std::vector<double> vd{0.5, 0.25};
   printf("%d %g\n", total(vi), total(vd));
   printf("%s %s %s\n", kind<int>(), kind<float>(), kind<char>());
   Pair2<int> p{4, 5};
   printf("%d %lu %s %s\n", first(p), bytes(), name(1), name(2.0));
   return 0;
}
"""


def test_each_function_template_instance_is_a_function_of_its_own(capsys) -> None:
    run_source(FUNCTIONS, "functions.C")
    assert capsys.readouterr().out == (
        "3 3.5 0.5\n8 64 64\n6 2.5\n3 10 12\n6 0.75\nint floating other\n4 8 any double\n"
    )
    python = translate(FUNCTIONS, "functions.C")
    assert "half__int(7)" in python and "half__double(7.0)" in python
    assert "def kind__int():\n    return 'int'\n" in python


def test_an_instance_needed_while_another_is_written_is_written_before_it(capsys) -> None:
    source = r"""
    template <typename T> struct Holder { T held; };
    template <typename T> T unwrap(T x) { Holder<T> h{x}; return h.held; }
    template <typename T> T twice(T x) { return unwrap(x) * 2; }
    int g = twice(4);
    void chained() { printf("%d %g\n", g, twice(1.5)); }
    """
    run_source(source, "chained.C")
    assert capsys.readouterr().out == "8 3\n"
    python = translate(source, "chained.C")
    assert python.index("class Holder__int") < python.index("def unwrap__int")
    assert python.index("def twice__int") < python.index("g = twice__int(4)")


def test_a_template_whose_arguments_are_not_known_is_refused_when_it_is_used() -> None:
    source = """
    template <typename T> T zero() { T x; return x; }
    template <typename T> struct Bag { T x; static T made() { T y; return y; } };
    int unknown(int n) { return zero(); }
    """
    python = translate(source, "unknown.C")
    assert "uninstantiated('the function template zero', " in python
    assert "uninstantiated('the class template Bag', " in python
    names = run_source(source, "unknown.C", call=False)
    with pytest.raises(Refusal, match=r"template arguments are not known: .*template parameter"):
        names["unknown"](1)
    with pytest.raises(Refusal, match="the class template Bag"):
        names["Bag"]()
    with pytest.raises(Refusal, match="the class template X, used where"):
        uninstantiated("the class template X", "why")


def test_template_specialisations_are_chosen_as_cpp_chooses_them(capsys) -> None:
    source = r"""
    template <typename T, typename U> struct P { int k() { return 0; } };
    template <typename T> struct P<T, int> { int k() { return 1; } };
    template <typename T> struct P<T*, T> { int k() { return 2; } };
    template <typename T> struct P<const T*, double> { int k() { return 3; } };
    template <> struct P<char, char> { int k() { return 4; } };
    template <int N, bool B> struct V { int k() { return N; } };
    template <int N> struct V<N, true> { int k() { return -N; } };
    void spec() {
      P<double, double> a; P<char, int> b; P<long*, long> c; P<const int*, double> d;
      P<char, char> e; V<3, false> f; V<4, true> g; V<-2, true> h;
      printf("%d %d %d %d %d %d %d %d\n", a.k(), b.k(), c.k(), d.k(), e.k(), f.k(), g.k(), h.k());
    }
    """
    run_source(source, "spec.C")
    assert capsys.readouterr().out == "0 1 2 3 4 3 -4 2\n"


def test_an_alias_template_names_the_type_with_its_arguments_put_in(capsys) -> None:
    source = r"""
    #include <vector>
    template <typename T> struct Wrapper { std::vector<T> items; };
    template <typename T, typename U = T> using Pair = std::pair<T, U>;
    struct Holder {
       template <typename T> using Vec = std::vector<T>;
       template <typename T> using Wrapped = Wrapper<T>;
       Vec<float> v;
       Wrapped<int> w;
    };
    void aliases() {
      Holder h; h.v.push_back(1.5); h.w.items.push_back(3);
      Pair<int> p{1, 2};
      printf("%g %d %d %lu\n", h.v[0], h.w.items[0], p.second, sizeof(Pair<char, double>));
    }
    """
    run_source(source, "aliases.C")
    assert capsys.readouterr().out == "1.5 3 2 16\n"


def test_nested_classes_static_members_and_out_of_class_members_belong_to_each_instance(
    capsys,
) -> None:
    source = r"""
    template <class T> class List {
    public:
      struct Node { T value; Node *next; };
      List() : head(nullptr) {}
      void push(T v);
      T front() const { return head->value; }
      static const char *tag;
    private:
      Node *head;
    };
    template <class T> void List<T>::push(T v) { head = new Node{v, head}; }
    template <class T> const char *List<T>::tag = "list";
    void nested() {
      List<int> a; a.push(3); a.push(4);
      List<double> b; b.push(0.5);
      printf("%d %g %s %lu\n", a.front(), b.front(), List<int>::tag, sizeof(List<int>::Node));
    }
    """
    run_source(source, "nested.C")
    assert capsys.readouterr().out == "4 0.5 list 16\n"
    assert "class List__int__Node:" in translate(source, "nested.C")


def test_type_traits_of_known_types_are_constants(capsys) -> None:
    source = r"""
    #include <type_traits>
    enum Colour { kCrimson };
    struct Base {}; struct Child : Base {};
    template <class T> int traits() {
      return std::is_integral_v<T> + 2 * std::is_signed<T>::value + 4 * std::is_pointer_v<T>
           + 8 * std::is_class_v<T> + 16 * std::is_enum_v<T> + 32 * std::is_arithmetic_v<T>
           + 64 * std::is_unsigned_v<T> + 128 * std::is_const_v<T>;
    }
    void known() {
      printf("%d %d %d %d %d %d %d\n", traits<int>(), traits<unsigned>(), traits<double *>(),
             traits<Child>(), traits<Colour>(), traits<float>(), traits<const char>());
      printf("%d %d %d\n", std::is_base_of_v<Base, Child>, std::is_base_of_v<Child, Base>,
             std::is_same<int, long>::value);
    }
    """
    run_source(source, "known.C")
    assert capsys.readouterr().out == "35 97 4 8 16 34 163\n1 0 0\n"


def test_a_trait_of_a_library_class_is_left_to_the_library() -> None:
    source = "void t() { bool c = std::is_class_v<Gadget>; bool e = std::is_enum_v<Gadget>; }"
    python = translate(source, "t.C")
    assert "CXX.std.is_class_v['Gadget']" in python
    assert "CXX.std.is_enum_v['Gadget']" in python
    assert "c = False" in translate("void t() { bool c = std::is_class_v<int*>; }", "t.C")


def test_if_constexpr_keeps_only_the_branch_its_constant_chooses() -> None:
    source = """
    template <class T> int pick(T x) {
      if constexpr (sizeof(T) == 8 && !std::is_integral_v<T>) return 1;
      else if constexpr (-sizeof(T) < 0 || false) { return x.missing(); }
      return 0;
    }
    int a = pick(1.0);
    """
    python = translate(source, "t.C")
    assert "def pick__double(x):\n    return 1\n    return 0\n" in python
    chosen = "void t() { if constexpr (int k = 2; false) { f(k); } }"
    assert "def t():\n    k = 2\n\n" in translate(chosen, "t.C")
    generic = translate("template <class T> int f() { if constexpr (sizeof(T) > 2) {} }", "")
    assert "if 0 > 2:" not in generic and "uninstantiated" in generic


def test_a_template_argument_that_is_not_a_constant_leaves_the_template_generic() -> None:
    source = """
    template <int N> struct Fixed { int a[N]; };
    template <int N> int count() { return N; }
    void t(int n) { Fixed<n> f; int c = count<n>(); Fixed<-1> g; }
    """
    python = translate(source, "t.C")
    assert "f = Fixed()" in python and "count(N=n)" in python
    assert "class Fixed__minus_1:" in python


def test_an_instance_is_named_by_a_python_identifier_c_plus_plus_reserves() -> None:
    assert mangled("Box<int*>") == "Box__int_p"
    assert mangled("Grid<double, 3>") == "Grid__double_3"
    assert mangled("Holder<>") == "Holder__0"


def test_instances_named_alike_are_told_apart() -> None:
    source = "template <class T> struct B {}; struct B__int_p {}; B<int*> b; B__int_p c;"
    assert "class B__int_p_:" in translate(source, "t.C")


def test_a_definition_whose_template_arguments_cannot_be_read_is_read_as_a_name() -> None:
    source = "template <auto V> struct Box { int f(); }; template <> int Box<[]>::f() { return 1; }"
    assert "def f(" in translate(source, "t.C")


def test_a_specialisation_for_arguments_it_cannot_work_out_is_refused() -> None:
    with pytest.raises(Refusal, match="a specialisation of B for arguments"):
        translate("template <int N> struct B {}; template <> struct B<n> {};", "t.C")


#: Corners of instantiation: a destructor, a static local, a call made twice, a template's
#: own name with other arguments, a standard container's member types, and ``T::member``.
CORNERS = r"""
#include <cstdio>
#include <map>
#include <vector>
template <typename T> struct Counted {
   static int alive;
   T v;
   Counted(T x) : v(x) { ++alive; }
   ~Counted() { --alive; }
};
template <typename T> int Counted<T>::alive = 0;
template <typename T> int next() { static int n = 0; return ++n + int(sizeof(T)); }
template <typename C> typename C::value_type head(const C &c) {
   typename C::size_type n = c.size();
   return c[n - 2];
}
template <typename M> typename M::mapped_type second(const M &m, typename M::key_type k) {
   return m.at(k);
}
template <typename T> int alive() { return T::alive; }
template <typename T> T *first(T *p) { return p; }
template <typename T> int third(T (&a)[3]) { return a[2]; }
struct Plain { int a; };
int corners() {
   Counted<int> *c = new Counted<int>(3);
   Counted<double> d(c->v);
   printf("%d %d %g\n", Counted<int>::alive, alive<Counted<double>>(), d.v);
   delete c;
   printf("%d %d %d\n", Counted<int>::alive, next<char>(), next<char>());
   std::vector<double> v{2.5, 1};
   std::map<int, long> m{{1, 7}};
   int arr[3] = {4, 5, 6};
   Plain plain[2];
   plain[0].a = 1;
   printf("%g %ld %d %d %d\n", head(v), second(m, 1), *first(arr), third(arr), plain[0].a);
   return 0;
}
"""


def test_instances_have_destructors_static_locals_and_member_types_of_their_own(capsys) -> None:
    run_source(CORNERS, "corners.C")
    assert capsys.readouterr().out == "1 1 3\n0 2 3\n2.5 7 4 6 1\n"


def test_a_template_given_too_few_arguments_or_a_value_not_known_is_left_generic() -> None:
    source = """
    template <class T, class U> struct Two { T t; };
    template <class T> struct Two<T *, char> { T t; };
    template <class T> struct Two<T, T> { T t; };
    template <class T> T *pick(T (&a)[2]) { return a; }
    template <class T> int take(T a, T b) { return 0; }
    template <class T> struct Two<T *> { T t; };
    template <class T> struct Holder { T h; };
    template <int N> int count(int (&a)[N]) { return N; }
    template <class T> struct Outer { struct In { static int n; }; };
    template <class T> int Outer<T>::In::n = 0;
    template <class T, int N> struct W { int k() { return 0; } };
    template <class T> struct W<T, 2> { int k() { return 1; } };
    template <class T> struct List { struct Node { T v; }; };
    template <class T> int size() { typename List<T>::Node node; return sizeof(node); }
    const int k = n;
    const int kTwo = 2;
    void t(int *p) {
      Two<int> a; Two<int*, int> b; Two<int, double> c;
      Holder<Two<int, int>> h; Holder<int[k]> g;
      int q[3]; pick(q); pick(p); take(1, 2.0); int v[2]; pick(v); int u[k]; count(u);
      W<int, kTwo> w2; W<int, 3> w3;
      bool same = std::is_same_v<T, 3>;
    }
    """
    python = translate(source, "t.C")
    assert "a = Two()" in python and "b = Two__int_p_int()" in python
    assert "pick(q)" in python and "pick(p)" in python and "take(1, 2.0)" in python
    assert "pick__int(v)" in python and "g = Holder()" in python and "count(u)" in python
    assert "w2 = W__int_2()" in python and "class W__int_2:\n" in python


def test_a_static_member_defined_for_one_instance_is_refused() -> None:
    source = "template <class T> struct B { static int n; }; template <> int B<int>::n = 5;"
    with pytest.raises(Refusal, match="a static member B::n defined for one instance"):
        translate(source, "t.C")


def test_specialisations_inside_a_class_and_of_the_standards_templates(capsys) -> None:
    source = r"""
    #include <functional>
    struct Key { int k; };
    template <> struct std::hash<Key> {
      std::size_t operator()(const Key &key) const { return key.k * 7; }
    };
    struct Outer {
      template <class T> struct Kind { static int k() { return 0; } };
      template <class T> struct Kind<T *> { static int k() { return 1; } };
      template <> struct Kind<int> { static int k() { return 2; } };
    };
    void nested() {
      Key key{3};
      printf("%zu %d %d %d\n", std::hash<Key>{}(key), Outer::Kind<char>::k(),
             Outer::Kind<char *>::k(), Outer::Kind<int>::k());
    }
    """
    run_source(source, "nested.C")
    assert capsys.readouterr().out == "21 0 1 2\n"


def test_an_explicit_instantiation_declared_elsewhere_is_passed_over() -> None:
    source = "template <class T> struct B {}; extern template class B<int>; B<int> b;"
    assert "b = B__int()" in translate(source, "t.C")


def test_an_alias_template_short_of_arguments_leaves_the_rest_as_written() -> None:
    source = "template <class T, class U> using M = std::map<T, U>; M<int> m;"
    assert "CXX.std.map['int', 'U']" in translate(source, "t.C")


def test_a_generic_template_constructs_its_own_template_generically() -> None:
    source = """
    template <class T> struct Box { T v; Box(T x) : v(x) {} };
    template <class T> Box<T> boxed(T x) { return Box<T>(x); }
    void t(Gadget g) { boxed(g); }
    """
    assert "return Box(x)" in translate(source, "t.C")
