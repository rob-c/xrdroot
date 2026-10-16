"""Other tree-loop bodies retain their ordinary C++ statement semantics."""
import pytest

from xrdroot.cint.parse_decl import parse
from xrdroot.cint.tokens import tokenize
from xrdroot.cint.tree_loops import ntuple_fill


def body(text):
    function = parse(tokenize('void fill() {' + text + '}')).decls[0]
    return function.body


def test_one_getentry_and_one_getargs_fill_form_a_read_loop():
    plan = ntuple_fill(body('t->GetEntry(i); h->Fill(t->GetArgs()[3]);'), 'i')
    assert plan.tree.parts == ['t'] and plan.histogram.parts == ['h'] and plan.column == 3


@pytest.mark.parametrize('text', [
    '', 't->GetEntry(i);', 't->GetEntry(i); h->Fill(t->GetArgs()[0]); extra();',
    'int x=1; h->Fill(t->GetArgs()[0]);', 't->Read(i); h->Fill(t->GetArgs()[0]);',
    'Read(i); h->Fill(t->GetArgs()[0]);', 't->GetEntry(i); Fill(t->GetArgs()[0]);',
    'get()->GetEntry(i); h->Fill(t->GetArgs()[0]);',
    't->GetEntry(i, 1); h->Fill(t->GetArgs()[0]);',
    't->GetEntry(i); h->Fill(t->GetArgs()[0], 2);',
    't->GetEntry(i+1); h->Fill(t->GetArgs()[0]);',
    't->GetEntry(j); h->Fill(t->GetArgs()[0]);',
    't->GetEntry(i); h->Fill(x);', 't->GetEntry(i); h->Fill(t->GetArgs()[i]);',
    't->GetEntry(i); h->Fill(t->GetArgs()[-1]);',
    "t->GetEntry(i); h->Fill(t->GetArgs()['a']);",
    't->GetEntry(i); h->Fill(t->GetArgs(1)[0]);',
    't->GetEntry(i); h->Fill(t->Values()[0]);',
    't->GetEntry(i); h->Fill(GetArgs()[0]);',
    't->GetEntry(i); h->Fill(other->GetArgs()[0]);',
    't->GetEntry(i); h->Fill(get()->GetArgs()[0]);',
    't->GetEntry(i); h->Fill(values[0]);',
    't->GetEntry(i); h->Fill(t->GetArgs()[0.5]);',
])
def test_other_bodies_retain_the_original_loop(text):
    assert ntuple_fill(body(text), 'i') is None
