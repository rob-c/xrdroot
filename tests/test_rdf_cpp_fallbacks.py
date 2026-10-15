"""Batch recognition leaves lifetimes, mutations and other numeric types to C++ translation."""
from __future__ import annotations

import pytest

from xrdroot.pyroot.rdf.batched import compiled


@pytest.mark.parametrize('parameters,used,body', [
    ('double x', ['x'], 'std::array<double,2> v{x?x:1.,x}; return v;'),
    ('double x', ['x'], 'std::array<double,2> v{M_PI,x}; return v;'),
    ('double x', ['x'], 'std::array<double,2> v{true,x}; return v;'),
    ('double x', ['x'], 'std::array<double,2> v{x/2.,x}; return v;'),
    ('double x', ['x'], 'std::array<double,2> v{!x,x}; return v;'),
    ('double x', ['x'], 'std::array<double,2> v{x++,x}; return v;'),
    ('double x', ['x'], 'std::array<double,2> v{std::abs(x),x}; return v;'),
    ('double x', ['x'], 'std::array<double,2> v{sqrt(),x}; return v;'),
    ('double x', ['x'], 'std::array<double,2> v{std::vector<double>{x}.size(),x}; return v;'),
    ('double x', ['x'], 'x++; return x;'),
    ('double x', ['x'], 'std::array<int,2> v{1,2}; return v;'),
    ('double x', ['x'], 'std::array<double> v{x}; return v;'),
    ('double x', ['x'], 'std::array<1,2> v{x}; return v;'),
    ('double x', ['x'], 'std::array<double,x> v{x}; return v;'),
    ('double x', ['x'], 'std::array<double,2.> v{x}; return v;'),
    ('double x', ['x'], 'std::array<double,-1> v{x}; return v;'),
    ('double x', ['x'], 'static std::array<double,2> v{x}; return v;'),
    ('double x', ['x'], 'std::array<double,2> v={x}; return v;'),
    ('double x', ['x'], 'std::array<double,1> v{x,x}; return v;'),
    ('double x', ['x'], 'std::array<double,2> v{x}; return x;'),
    ('float x', ['x'], 'std::array<double,2> v{x}; return v;'),
    ('double x', ['x'], 'double r=0.; r+=x; return r;'),
    ('std::vector<double> p', ['p'], 'double r=0.,s=0.; for(auto w:p) r+=w; return r;'),
    ('std::vector<double> p', ['p'], 'int r=0; for(auto w:p) r+=w; return r;'),
    ('std::vector<double> p', ['p'], 'static double r=0.; for(auto w:p) r+=w; return r;'),
    ('std::vector<double> p', ['p'], 'const double r=0.; for(auto w:p) r+=w; return r;'),
    ('std::vector<double> p', ['p'], 'double r=1.+1.; for(auto w:p) r+=w; return r;'),
    ('std::vector<double> p', ['p'], 'double r=0.; for(auto w:(p+1)) r+=w; return r;'),
    ('std::vector<double> p', [], 'double r=0.; for(auto w:p) r+=w; return r;'),
    ('std::vector<double> p', ['p'], 'double r=0.; for(int w:p) r+=w; return r;'),
    ('std::vector<double> p', ['p'], 'double r=0.; for(auto* w:p) r+=*w; return r;'),
    ('std::vector<double> p', ['p'], 'double r=0.; for(auto r:p) r+=r; return r;'),
    ('std::vector<double> p', ['p'], 'double r=0.; for(auto w:p) {r+=w; r+=1.;} return r;'),
    ('std::vector<double> p', ['p'], 'double r=0.; for(auto w:p) ++r; return r;'),
    ('std::vector<double> p', ['p'], 'double r=0.; for(auto w:p) r-=w; return r;'),
    ('std::vector<double> p', ['p'], 'double r=0.; for(auto w:p) w+=r; return r;'),
    ('std::vector<float> p', ['p'], 'double r=0.; for(auto w:p) r+=w; return r;'),
    ('Sequence p', ['p'], 'double r=0.; for(auto w:p) r+=w; return r;'),
    ('std::vector p', ['p'], 'double r=0.; for(auto w:p) r+=w; return r;'),
])
def test_bodies_outside_the_pure_double_subset_retain_the_translator(parameters, used, body):
    source = f'auto f({parameters}) {{ {body} }}'
    assert compiled(source, used, lambda: None) is None


@pytest.mark.parametrize('body', [
    'return sqrt(2.);',
    'return RooRandom::randomGenerator()->Uniform();',
    'return otherRandom->Uniform();',
    'return gRandom->Integer(5);',
    'return gRandom->Uniform(M_PI);',
    'return gRandom->Uniform(-(1.+2.));',
])
def test_other_calls_and_nonliteral_random_bounds_keep_the_original_function(body):
    assert compiled(f'auto f() {{ {body} }}', [], lambda: None) is None
