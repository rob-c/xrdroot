"""RooFit's decays and resolution models, held to the numbers ROOT 6.40.04 printed.

Every reference below was printed by ROOT itself through PyROOT - values
with ``%.12g`` or ``repr``, generated events with ``repr`` after
``RooRandom::randomGenerator()->SetSeed(4357)`` - for the same models built
here with the engine's classes. The Faddeeva function and the generated
events agree to the last bit; values and integrals to the twelve digits
printed; fits to the ten printed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.cerf import eval_cerf, eval_cerf_approx, faddeeva_fast
from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.functions import RooFormulaVar
from xrdroot.roofit.pdfs.addmodel import RooAddModel
from xrdroot.roofit.pdfs.anaconv import RooAbsAnaConvPdf, decay_type
from xrdroot.roofit.pdfs.basic import ref
from xrdroot.roofit.pdfs.bdecays import RooBCPEffDecay, RooBCPGenDecay, RooBDecay
from xrdroot.roofit.pdfs.decays import RooBMixDecay, RooDecay
from xrdroot.roofit.pdfs.gaussmodel import RooGaussModel
from xrdroot.roofit.pdfs.resolution import RooTruthModel
from xrdroot.roofit.pdfs.shapes import RooLandau
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar

#: ``RooMath::faddeeva_fast`` at these points, real and imaginary parts.
FADDEEVA = [
    ((0.1, 0.2), (0.8025666876062311, 0.08002860366885166)),
    ((3.14, 1e-4), (5.9222639732486064e-05, 0.1907921593134494)),
    ((-3.14, 1e-4), (5.9222639732486064e-05, -0.1907921593134494)),
    ((0.0, 0.0), (1.0, 0.0)),
    ((1.5, -0.3), (-0.030490856189214988, 0.5723397933024678)),
    ((-2, -1), (-0.20532558046177218, -0.14685548524505887)),
    ((10, 3), (0.015721778420853833, 0.05191987664557936)),
    ((-9, -0.5), (-0.003537804730597072, -0.06288174451925028)),
    ((0.2, -5), (-57579363738.729546, 125813205070.12158)),
    ((6.28, 0.001), (1.4887268732736755e-05, 0.09102441006440055)),
    ((25, 0.01), (9.048783908317857e-06, 0.022585677284165415)),
    ((3, 8), (0.06161254160922123, 0.022796642169821912)),
    ((-7, 7.5), (0.04036512264016637, -0.03731841604298533)),
    ((0.5, 1e-3), (0.7781517156462114, 0.4781471774701084)),
    ((6.2832, -0.001), (-1.4871852741276413e-05, 0.0909767948734399)),
]


def test_the_faddeeva_function_is_roots_to_the_last_bit() -> None:
    z = np.array([complex(*point) for point, _ in FADDEEVA])
    found = faddeeva_fast(z)
    for w, (_, (re, im)) in zip(found, FADDEEVA):
        assert (float(w.real), float(w.imag)) == (re, im)
    assert faddeeva_fast(complex(0.1, 0.2)) == complex(*FADDEEVA[0][1])


def test_the_complex_error_function_takes_its_safe_form_far_below_the_axis() -> None:
    assert eval_cerf(0.3, -6.0, 0.5) == eval_cerf_approx(0.3, -6.0, 0.5)
    assert eval_cerf(0.0, -6.0, 0.5) == eval_cerf_approx(0.0, -6.0, 0.5)
    on_axis = eval_cerf(0.0, 0.4, 0.5)
    assert on_axis == pytest.approx(math.exp(0.5 * (0.5 + 0.8)) * math.erfc(0.9), rel=1e-15)
    both = eval_cerf(np.array([0.0, 0.3]), np.array([0.4, -6.0]), 0.5)
    assert both[1] == eval_cerf_approx(0.3, -6.0, 0.5)


def test_a_decay_type_is_a_number_or_its_name() -> None:
    assert decay_type("Flipped") == RooAbsAnaConvPdf.Flipped == 2
    assert decay_type(RooDecay.SingleSided) == 0
    with pytest.raises(ValueError, match="is not a decay type"):
        decay_type("Sideways")


#: Per model, lifetime, frequency and basis: the convolution at t = -6.5, -0.4, 0, 0.3, 5.9, then
#: its#: integral over [-8, 8] and over the range "win", [-1.5, 2.5].
CONVOLUTIONS = """
truth 1.3 0.6 exp(-@0/@1) 0 0 1 0.793922657818 0.0106898398167 1.29723712062 1.10999647581
truth 1.3 0.6 exp(@0/@1) 0.00673794699909 0.735141480592 1 0 0 1.29723712062 0.889952342969
truth 1.3 0.6 exp(-abs(@0)/@1) 0.00673794699909 0.735141480592 1 0.793922657818 0.0106898398167 2.59447424124 1.99994881878
truth 1.3 0.6 exp(-@0/@1)*sin(@0*@2) 0 0 0 0.142135634762 -0.00414713316955 0.632034144697 0.506086166032
truth 1.3 0.6 exp(@0/@1)*sin(@0*@2) 0.00463413192835 -0.174745060732 0 0 0 -0.632034144697 -0.307127991407
truth 1.3 0.6 exp(-abs(@0)/@1)*sin(@0*@2) 0.00463413192835 -0.174745060732 0 0.142135634762 -0.00414713316955 0 0.198958174625
truth 1.3 0.6 exp(-@0/@1)*cos(@0*@2) 0 0 1 0.781095799456 -0.00985261192685 0.806771618 0.891812472887
truth 1.3 0.6 exp(@0/@1)*cos(@0*@2) -0.00489129339062 0.714070836988 1 0 0 0.806771618 0.805550455626
truth 1.3 0.6 exp(-abs(@0)/@1)*cos(@0*@2) -0.00489129339062 0.714070836988 1 0.781095799456 -0.00985261192685 1.613543236 1.69736292851
truth 1.3 0.6 (@0/@1)*exp(-@0/@1) 0 0 0 0.183212921035 0.0485154268605 1.28023478597 0.744605083128
truth 1.3 0.6 (@0/@1)*(@0/@1)*exp(-@0/@1) 0 0 0 0.0422799048542 0.220185398828 2.45583982027 0.786534411105
truth 1.3 0.6 exp(-@0/@1)*cosh(@0*@2/2) 0 0 1 0.797140215554 0.0322896571933 1.50814581647 1.17121218674
truth 1.3 0.6 exp(@0/@1)*cosh(@0*@2/2) 0.0241587797724 0.740440853924 1 0 0 1.50814581647 0.912033057132
truth 1.3 0.6 exp(-abs(@0)/@1)*cosh(@0*@2/2) 0.0241587797724 0.740440853924 1 0.797140215554 0.0322896571933 3.01629163294 2.08324524388
truth 1.3 0.6 exp(-@0/@1)*sinh(@0*@2/2) 0 0 0 0.071549539881 0.0304688248273 0.608697716513 0.669070959096
truth 1.3 0.6 exp(@0/@1)*sinh(@0*@2/2) -0.0232001446187 -0.0884288509086 0 0 0 -0.608697716513 -0.614962818421
truth 1.3 0.6 exp(-abs(@0)/@1)*sinh(@0*@2/2) -0.0232001446187 -0.0884288509086 0 0.071549539881 0.0304688248273 0 0.0541081406752
truth 0.0 0.6 exp(-@0/@1) 0 0 nan 0 0 1 1
truth 0.0 0.6 exp(@0/@1) 0 0 nan 0 0 1 1
truth 0.0 0.6 exp(-abs(@0)/@1) 0 0 nan 0 0 1 1
truth 0.0 0.6 exp(-@0/@1)*sin(@0*@2) 0 0 nan 0 -0 0 0
truth 0.0 0.6 exp(@0/@1)*sin(@0*@2) 0 -0 nan 0 0 0 0
truth 0.0 0.6 exp(-abs(@0)/@1)*sin(@0*@2) 0 -0 nan 0 -0 0 0
truth 0.0 0.6 exp(-@0/@1)*cos(@0*@2) 0 0 nan 0 -0 1 1
truth 0.0 0.6 exp(@0/@1)*cos(@0*@2) -0 0 nan 0 0 1 1
truth 0.0 0.6 exp(-abs(@0)/@1)*cos(@0*@2) -0 0 nan 0 -0 1 1
truth 0.0 0.6 (@0/@1)*exp(-@0/@1) 0 0 nan nan nan 0 0
truth 0.0 0.6 (@0/@1)*(@0/@1)*exp(-@0/@1) 0 0 nan nan nan 1 1
truth 0.0 0.6 exp(-@0/@1)*cosh(@0*@2/2) 0 0 nan 0 0 1 1
truth 0.0 0.6 exp(@0/@1)*cosh(@0*@2/2) 0 0 nan 0 0 1 1
truth 0.0 0.6 exp(-abs(@0)/@1)*cosh(@0*@2/2) 0 0 nan 0 0 1 1
truth 0.0 0.6 exp(-@0/@1)*sinh(@0*@2/2) 0 0 nan 0 0 0 0
truth 0.0 0.6 exp(@0/@1)*sinh(@0*@2/2) -0 -0 nan 0 0 0 0
truth 0.0 0.6 exp(-abs(@0)/@1)*sinh(@0*@2/2) -0 -0 nan 0 0 0 0
truth 1.3 0.0 exp(-@0/@1) 0 0 1 0.793922657818 0.0106898398167 1.29723712062 1.10999647581
truth 1.3 0.0 exp(@0/@1) 0.00673794699909 0.735141480592 1 0 0 1.29723712062 0.889952342969
truth 1.3 0.0 exp(-abs(@0)/@1) 0.00673794699909 0.735141480592 1 0.793922657818 0.0106898398167 2.59447424124 1.99994881878
truth 1.3 0.0 exp(-@0/@1)*sin(@0*@2) 0 0 0 0 0 0 0
truth 1.3 0.0 exp(@0/@1)*sin(@0*@2) -0 -0 0 0 0 0 0
truth 1.3 0.0 exp(-abs(@0)/@1)*sin(@0*@2) -0 -0 0 0 0 0 0
truth 1.3 0.0 exp(-@0/@1)*cos(@0*@2) 0 0 1 0.793922657818 0.0106898398167 1.29723712062 1.10999647581
truth 1.3 0.0 exp(@0/@1)*cos(@0*@2) 0.00673794699909 0.735141480592 1 0 0 1.29723712062 0.889952342969
truth 1.3 0.0 exp(-abs(@0)/@1)*cos(@0*@2) 0.00673794699909 0.735141480592 1 0.793922657818 0.0106898398167 2.59447424124 1.99994881878
truth 1.3 0.0 (@0/@1)*exp(-@0/@1) 0 0 0 0.183212921035 0.0485154268605 1.28023478597 0.744605083128
truth 1.3 0.0 (@0/@1)*(@0/@1)*exp(-@0/@1) 0 0 0 0.0422799048542 0.220185398828 2.45583982027 0.786534411105
truth 1.3 0.0 exp(-@0/@1)*cosh(@0*@2/2) 0 0 1 0.793922657818 0.0106898398167 1.29723712062 1.10999647581
truth 1.3 0.0 exp(@0/@1)*cosh(@0*@2/2) 0.00673794699909 0.735141480592 1 0 0 1.29723712062 0.889952342969
truth 1.3 0.0 exp(-abs(@0)/@1)*cosh(@0*@2/2) 0.00673794699909 0.735141480592 1 0.793922657818 0.0106898398167 2.59447424124 1.99994881878
truth 1.3 0.0 exp(-@0/@1)*sinh(@0*@2/2) 0 0 0 0 0 0 0
truth 1.3 0.0 exp(@0/@1)*sinh(@0*@2/2) -0 -0 0 0 0 0 0
truth 1.3 0.0 exp(-abs(@0)/@1)*sinh(@0*@2/2) -0 -0 0 0 0 0 0
gauss 1.3 0.6 exp(-@0/@1) 3.00933878242e-15 0.378288802278 0.581200511102 0.714327214432 0.0288903032259 2.59253305715 2.07383742423
gauss 1.3 0.6 exp(@0/@1) 0.0151401804044 0.808071412827 0.690678626625 0.551257985329 5.4522843948e-12 2.5937918045 1.69973237676
gauss 1.3 0.6 exp(-abs(@0)/@1) 0.0151401804044 1.1863602151 1.27187913773 1.26558519976 0.0288903032314 5.18632486164 3.77356980099
gauss 1.3 0.6 exp(-@0/@1)*sin(@0*@2) 1.72730313472e-16 0.0872659471978 0.158481447808 0.222740634036 -1.89766793074e-05 1.26574721547 0.908722560281
gauss 1.3 0.6 exp(@0/@1)*sin(@0*@2) 0.00644829598567 -0.295547147304 -0.209550619127 -0.146464964732 -3.51785924329e-13 -1.26467137671 -0.638183919388
gauss 1.3 0.6 exp(-abs(@0)/@1)*sin(@0*@2) 0.00644829598567 -0.208281200106 -0.0510691713191 0.0762756693033 -1.89766796592e-05 0.00107583875437 0.270538640893
gauss 1.3 0.6 exp(-@0/@1)*cos(@0*@2) 2.99951411159e-15 0.359809568427 0.542233644789 0.652620882821 -0.0254444522046 1.61472365623 1.67283393766
gauss 1.3 0.6 exp(@0/@1)*cos(@0*@2) -0.0116715310015 0.713664350807 0.634022851833 0.516068689445 5.42984224253e-12 1.61446007786 1.47100398591
gauss 1.3 0.6 exp(-abs(@0)/@1)*cos(@0*@2) -0.0116715310015 1.07347391923 1.17625649662 1.16868957227 -0.0254444521991 3.2291837341 3.14383792357
gauss 1.3 0.6 (@0/@1)*exp(-@0/@1) 2.22157045504e-16 0.116401346656 0.21401306109 0.304518941932 0.116388603783 2.55038945494 1.35786191645
gauss 1.3 0.6 (@0/@1)*(@0/@1)*exp(-@0/@1) 3.23975739409e-17 0.0627813481343 0.133551140354 0.213265177212 0.480949803514 4.85980186248 1.4807546724
gauss 1.3 0.6 exp(-@0/@1)*cosh(@0*@2/2) 3.01180454248e-15 0.38310322328 0.591464775126 0.730752950612 0.0748551887948 3.00900664926 2.18984796561
gauss 1.3 0.6 exp(@0/@1)*cosh(@0*@2/2) 0.0496440347159 0.833578254406 0.70572559318 0.56051006664 5.45792227454e-12 3.01515607092 1.76308211198
gauss 1.3 0.6 exp(-abs(@0)/@1)*cosh(@0*@2/2) 0.0496440347159 1.21668147769 1.29719036831 1.29126301725 0.0748551888003 6.02416272019 3.95293007759
gauss 1.3 0.6 exp(-@0/@1)*sinh(@0*@2/2) 8.67105340336e-17 0.0458533073695 0.0845684416616 0.120708127361 0.068658043304 1.13911676058 0.550679731699
gauss 1.3 0.6 exp(@0/@1)*sinh(@0*@2/2) -0.0471198075753 -0.163869161711 -0.113174105171 -0.0779492155127 -1.76775468715e-13 -1.14512727898 -0.364095562484
gauss 1.3 0.6 exp(-abs(@0)/@1)*sinh(@0*@2/2) -0.0471198075753 -0.118015854342 -0.0286056635092 0.0427589118479 0.0686580433039 -0.00601051840264 0.186584169215
gauss 0.0 0.6 exp(-@0/@1) 1.54790273489e-14 0.392116957241 0.470109690393 0.464151472937 2.48456611429e-11 1 0.970804689437
gauss 0.0 0.6 exp(@0/@1) 1.54790273489e-14 0.392116957241 0.470109690393 0.464151472937 2.48456611429e-11 1 0.970804689437
gauss 0.0 0.6 exp(-abs(@0)/@1) 3.09580546979e-14 0.784233914482 0.940219380786 0.928302945875 4.96913222857e-11 2 1.94160937887
gauss 0.0 0.6 exp(-@0/@1)*sin(@0*@2) 0 0 0 0 0 0 0
gauss 0.0 0.6 exp(@0/@1)*sin(@0*@2) 0 0 0 0 0 0 0
gauss 0.0 0.6 exp(-abs(@0)/@1)*sin(@0*@2) 0 0 0 0 0 0 0
gauss 0.0 0.6 exp(-@0/@1)*cos(@0*@2) 1.54790273489e-14 0.392116957241 0.470109690393 0.464151472937 2.48456611429e-11 1 0.970804689437
gauss 0.0 0.6 exp(@0/@1)*cos(@0*@2) 1.54790273489e-14 0.392116957241 0.470109690393 0.464151472937 2.48456611429e-11 1 0.970804689437
gauss 0.0 0.6 exp(-abs(@0)/@1)*cos(@0*@2) 3.09580546979e-14 0.784233914482 0.940219380786 0.928302945875 4.96913222857e-11 2 1.94160937887
gauss 0.0 0.6 (@0/@1)*exp(-@0/@1) 0 0 0 0 0 0 0
gauss 0.0 0.6 (@0/@1)*(@0/@1)*exp(-@0/@1) 0 0 0 0 0 0 0
gauss 0.0 0.6 exp(-@0/@1)*cosh(@0*@2/2) 0 0 0 0 0 0 0
gauss 0.0 0.6 exp(@0/@1)*cosh(@0*@2/2) 0 0 0 0 0 0 0
gauss 0.0 0.6 exp(-abs(@0)/@1)*cosh(@0*@2/2) 0 0 0 0 0 0 0
gauss 0.0 0.6 exp(-@0/@1)*sinh(@0*@2/2) 0 0 0 0 0 0 0
gauss 0.0 0.6 exp(@0/@1)*sinh(@0*@2/2) 0 0 0 0 0 0 0
gauss 0.0 0.6 exp(-abs(@0)/@1)*sinh(@0*@2/2) 0 0 0 0 0 0 0
gauss 1.3 0.0 exp(-@0/@1) 3.00933878242e-15 0.378288802278 0.581200511102 0.714327214432 0.0288903032259 2.59253305715 2.07383742423
gauss 1.3 0.0 exp(@0/@1) 0.0151401804044 0.808071412827 0.690678626625 0.551257985329 5.4522843948e-12 2.5937918045 1.69973237676
gauss 1.3 0.0 exp(-abs(@0)/@1) 0.0151401804044 1.1863602151 1.27187913773 1.26558519976 0.0288903032314 5.18632486164 3.77356980099
gauss 1.3 0.0 exp(-@0/@1)*sin(@0*@2) 0 0 0 0 0 0 0
gauss 1.3 0.0 exp(@0/@1)*sin(@0*@2) 0 0 0 0 0 0 0
gauss 1.3 0.0 exp(-abs(@0)/@1)*sin(@0*@2) 0 0 0 0 0 0 0
gauss 1.3 0.0 exp(-@0/@1)*cos(@0*@2) 3.00933878242e-15 0.378288802278 0.581200511102 0.714327214432 0.0288903032259 2.59253305715 2.07383742423
gauss 1.3 0.0 exp(@0/@1)*cos(@0*@2) 0.0151401804044 0.808071412827 0.690678626625 0.551257985329 5.4522843948e-12 2.5937918045 1.69973237676
gauss 1.3 0.0 exp(-abs(@0)/@1)*cos(@0*@2) 0.0151401804044 1.1863602151 1.27187913773 1.26558519976 0.0288903032314 5.18632486164 3.77356980099
gauss 1.3 0.0 (@0/@1)*exp(-@0/@1) 2.22157045504e-16 0.116401346656 0.21401306109 0.304518941932 0.116388603783 2.55038945494 1.35786191645
gauss 1.3 0.0 (@0/@1)*(@0/@1)*exp(-@0/@1) 3.23975739409e-17 0.0627813481343 0.133551140354 0.213265177212 0.480949803514 4.85980186248 1.4807546724
gauss 1.3 0.0 exp(-@0/@1)*cosh(@0*@2/2) 3.00933878242e-15 0.378288802278 0.581200511102 0.714327214432 0.0288903032259 2.59253305715 2.07383742423
gauss 1.3 0.0 exp(@0/@1)*cosh(@0*@2/2) 0.0151401804044 0.808071412827 0.690678626625 0.551257985329 5.4522843948e-12 2.5937918045 1.69973237676
gauss 1.3 0.0 exp(-abs(@0)/@1)*cosh(@0*@2/2) 0.0151401804044 1.1863602151 1.27187913773 1.26558519976 0.0288903032314 5.18632486164 3.77356980099
gauss 1.3 0.0 exp(-@0/@1)*sinh(@0*@2/2) 0 0 0 0 0 0 0
gauss 1.3 0.0 exp(@0/@1)*sinh(@0*@2/2) 0 0 0 0 0 0 0
gauss 1.3 0.0 exp(-abs(@0)/@1)*sinh(@0*@2/2) 0 0 0 0 0 0 0
narrow 1.3 0.6 exp(-@0/@1) 0 1.51824235678e-23 0.0448555644573 1.71601275977 0.0231062615363 2.59402799157 2.18930431156
narrow 1.3 0.6 exp(@0/@1) 0.0124873861597 1.36243214014 1.80713039597 6.27973833999e-05 0 2.59487957656 1.84006178174
narrow 1.3 0.6 exp(-abs(@0)/@1) 0.0124873861597 1.36243214014 1.85198596043 1.71607555715 0.0231062615363 5.18890756813 4.0293660933
narrow 1.3 0.6 exp(-@0/@1)*sin(@0*@2) 0 4.45135522394e-26 0.000496320874288 0.203378234506 -0.00764234267861 1.26451384012 0.981677330628
narrow 1.3 0.6 exp(@0/@1)*sin(@0*@2) 0.00910257096037 -0.400943670287 -0.109470110731 -4.2165217951e-07 0 -1.2636662137 -0.662699769383
narrow 1.3 0.6 exp(-abs(@0)/@1)*sin(@0*@2) 0.00910257096037 -0.400943670287 -0.108973789857 0.203377812854 -0.00764234267861 0.000847626421459 0.318977561245
narrow 1.3 0.6 exp(-@0/@1)*cos(@0*@2) 0 1.51822942079e-23 0.0448505572421 1.70314112231 -0.0217948073481 1.61352129175 1.78028174104
narrow 1.3 0.6 exp(@0/@1)*cos(@0*@2) -0.00854035929074 1.30145905125 1.80309550012 6.27946680178e-05 0 1.61359344482 1.64673281103
narrow 1.3 0.6 exp(-abs(@0)/@1)*cos(@0*@2) -0.00854035929074 1.30145905125 1.84794605736 1.70320391697 -0.0217948073481 3.22711473658 3.42701455207
narrow 1.3 0.6 (@0/@1)*exp(-@0/@1) 0 5.70691341668e-26 0.000636368849776 0.26147377726 0.103055293686 2.55774539002 1.43170442475
narrow 1.3 0.6 (@0/@1)*(@0/@1)*exp(-@0/@1) 0 4.25117677882e-28 1.6461561271e-05 0.0423784198484 0.459666888647 4.89504903961 1.46527607652
narrow 1.3 0.6 exp(-@0/@1)*cosh(@0*@2/2) 0 1.5182455898e-23 0.0448568163847 1.71923688209 0.0678208000054 3.01385801303 2.30316606501
narrow 1.3 0.6 exp(@0/@1)*cosh(@0*@2/2) 0.0460631237118 1.37782514257 1.80813991885 6.27980622589e-05 0 3.01858520536 1.89094700017
narrow 1.3 0.6 exp(-abs(@0)/@1)*cosh(@0*@2/2) 0.0460631237118 1.37782514257 1.85299673523 1.71929968015 0.0678208000054 6.03244321839 4.19411306518
narrow 1.3 0.6 exp(-@0/@1)*sinh(@0*@2/2) 0 2.225700887e-26 0.00024818970554 0.102046277027 0.0637623827216 1.14375632433 0.576579677182
narrow 1.3 0.6 exp(@0/@1)*sinh(@0*@2/2) -0.0443378107681 -0.204360197206 -0.0548050760692 -2.10836614302e-07 0 -1.14840613827 -0.358976822023
narrow 1.3 0.6 exp(-abs(@0)/@1)*sinh(@0*@2/2) -0.0443378107681 -0.204360197206 -0.0545568863636 0.10204606619 0.0637623827216 -0.00464981394241 0.217602855159
narrow 0.0 0.6 exp(-@0/@1) 0 1.53891972534e-21 1.07981933026 0.0026766045153 0 1 1
narrow 0.0 0.6 exp(@0/@1) 0 1.53891972534e-21 1.07981933026 0.0026766045153 0 1 1
narrow 0.0 0.6 exp(-abs(@0)/@1) 0 3.07783945068e-21 2.15963866053 0.0053532090306 0 2 2
narrow 0.0 0.6 exp(-@0/@1)*sin(@0*@2) 0 0 0 0 0 0 0
narrow 0.0 0.6 exp(@0/@1)*sin(@0*@2) 0 0 0 0 0 0 0
narrow 0.0 0.6 exp(-abs(@0)/@1)*sin(@0*@2) 0 0 0 0 0 0 0
narrow 0.0 0.6 exp(-@0/@1)*cos(@0*@2) 0 1.53891972534e-21 1.07981933026 0.0026766045153 0 1 1
narrow 0.0 0.6 exp(@0/@1)*cos(@0*@2) 0 1.53891972534e-21 1.07981933026 0.0026766045153 0 1 1
narrow 0.0 0.6 exp(-abs(@0)/@1)*cos(@0*@2) 0 3.07783945068e-21 2.15963866053 0.0053532090306 0 2 2
narrow 0.0 0.6 (@0/@1)*exp(-@0/@1) 0 0 0 0 0 0 0
narrow 0.0 0.6 (@0/@1)*(@0/@1)*exp(-@0/@1) 0 0 0 0 0 0 0
narrow 0.0 0.6 exp(-@0/@1)*cosh(@0*@2/2) 0 0 0 0 0 0 0
narrow 0.0 0.6 exp(@0/@1)*cosh(@0*@2/2) 0 0 0 0 0 0 0
narrow 0.0 0.6 exp(-abs(@0)/@1)*cosh(@0*@2/2) 0 0 0 0 0 0 0
narrow 0.0 0.6 exp(-@0/@1)*sinh(@0*@2/2) 0 0 0 0 0 0 0
narrow 0.0 0.6 exp(@0/@1)*sinh(@0*@2/2) 0 0 0 0 0 0 0
narrow 0.0 0.6 exp(-abs(@0)/@1)*sinh(@0*@2/2) 0 0 0 0 0 0 0
narrow 1.3 0.0 exp(-@0/@1) 0 1.51824235678e-23 0.0448555644573 1.71601275977 0.0231062615363 2.59402799157 2.18930431156
narrow 1.3 0.0 exp(@0/@1) 0.0124873861597 1.36243214014 1.80713039597 6.27973833999e-05 0 2.59487957656 1.84006178174
narrow 1.3 0.0 exp(-abs(@0)/@1) 0.0124873861597 1.36243214014 1.85198596043 1.71607555715 0.0231062615363 5.18890756813 4.0293660933
narrow 1.3 0.0 exp(-@0/@1)*sin(@0*@2) 0 0 0 0 0 0 0
narrow 1.3 0.0 exp(@0/@1)*sin(@0*@2) 0 0 0 0 0 0 0
narrow 1.3 0.0 exp(-abs(@0)/@1)*sin(@0*@2) 0 0 0 0 0 0 0
narrow 1.3 0.0 exp(-@0/@1)*cos(@0*@2) 0 1.51824235678e-23 0.0448555644573 1.71601275977 0.0231062615363 2.59402799157 2.18930431156
narrow 1.3 0.0 exp(@0/@1)*cos(@0*@2) 0.0124873861597 1.36243214014 1.80713039597 6.27973833999e-05 0 2.59487957656 1.84006178174
narrow 1.3 0.0 exp(-abs(@0)/@1)*cos(@0*@2) 0.0124873861597 1.36243214014 1.85198596043 1.71607555715 0.0231062615363 5.18890756813 4.0293660933
narrow 1.3 0.0 (@0/@1)*exp(-@0/@1) 0 5.70691341668e-26 0.000636368849776 0.26147377726 0.103055293686 2.55774539002 1.43170442475
narrow 1.3 0.0 (@0/@1)*(@0/@1)*exp(-@0/@1) 0 4.25117677882e-28 1.6461561271e-05 0.0423784198484 0.459666888647 4.89504903961 1.46527607652
narrow 1.3 0.0 exp(-@0/@1)*cosh(@0*@2/2) 0 1.51824235678e-23 0.0448555644573 1.71601275977 0.0231062615363 2.59402799157 2.18930431156
narrow 1.3 0.0 exp(@0/@1)*cosh(@0*@2/2) 0.0124873861597 1.36243214014 1.80713039597 6.27973833999e-05 0 2.59487957656 1.84006178174
narrow 1.3 0.0 exp(-abs(@0)/@1)*cosh(@0*@2/2) 0.0124873861597 1.36243214014 1.85198596043 1.71607555715 0.0231062615363 5.18890756813 4.0293660933
narrow 1.3 0.0 exp(-@0/@1)*sinh(@0*@2/2) 0 0 0 0 0 0 0
narrow 1.3 0.0 exp(@0/@1)*sinh(@0*@2/2) 0 0 0 0 0 0 0
narrow 1.3 0.0 exp(-abs(@0)/@1)*sinh(@0*@2/2) 0 0 0 0 0 0 0
"""  # noqa: E501

#: ROOT's curves: the B0 slice at t = 1; the mixing asymmetry at t = -2, 0.5, 3.
REFERENCE_SLICE = 16.1083409
REFERENCE_ASYMMETRY = [0.5685874975, 0.7376203286, 0.3162526033]
#: The conditional product's first event, its fit, and its projection at t = 0.5.
REFERENCE_PRODUCT_EVENT = (-1.6002829215094645, 0.8274411848780711)
REFERENCE_PRODUCT_FIT = (-0.1319442732, 0.8961762944)
REFERENCE_PRODUCT_CURVE = 10.25779408


def _convolution_models() -> tuple[dict[str, Any], Any, Any, Any]:
    t = RooRealVar("t", "t", -8, 8)
    t.setRange("win", -1.5, 2.5)
    tau, freq = RooRealVar("tau", "tau", 1.3), RooRealVar("freq", "freq", 0.6)
    mean, sigma = RooRealVar("mean", "mean", 0.1), RooRealVar("sigma", "sigma", 0.7)
    models = {
        "truth": RooTruthModel("truth", "truth", t),
        "gauss": RooGaussModel("gauss", "gauss", t, mean, sigma, RooRealVar("scale", "scale", 1.2)),
        "narrow": RooGaussModel("narrow", "narrow", t, mean, ref(0.05)),
    }
    return models, t, tau, freq


def test_each_basis_convolved_with_each_model_has_roots_values_and_integrals() -> None:
    models, t, tau, freq = _convolution_models()
    owner = RooRealVar("owner", "owner", 0)
    for line in CONVOLUTIONS.strip().splitlines():
        name, tv, fv, expr, *numbers = line.split()
        tau.setVal(float(tv))
        freq.setVal(float(fv))
        basis = RooFormulaVar(expr + "_b", expr, [t, tau, freq])
        conv = models[name].convolution(basis, owner)
        found = []
        for x in (-6.5, -0.4, 0.0, 0.3, 5.9):
            t.setVal(x)
            found.append(conv.getVal())
        found.append(conv.createIntegral([t]).getVal())
        found.append(conv.createIntegral([t], "win").getVal())
        expected = [float(one) for one in numbers]
        assert found == pytest.approx(expected, rel=1e-10, abs=1e-300, nan_ok=True), line


def test_a_convolution_is_named_for_its_basis_and_owner() -> None:
    models, t, tau, _ = _convolution_models()
    decay = RooDecay("decay", "decay", t, tau, models["gauss"], "DoubleSided")
    (conv,) = decay.convs
    assert conv.GetName() == "gauss_conv_exp(-abs(@0)/@1)_t_tau_[decay]"
    assert conv.GetTitle() == "gauss convoluted with basis function exp(-abs(@0)/@1)_t_tau"
    assert decay.normalized_name([t]) == f"{conv.GetName()}_over_{conv.GetName()}_Int[t]"
    assert decay.coefficient(0) == 1.0
    assert decay.printArgs() == "[ t=t tau=tau ]"
    assert "--- RooAbsAnaConvPdf ---" in decay.printMultiline(0, False, "")
    assert models["truth"].basisCode("exp(-@0*@0/@1)") == 100
    assert models["gauss"].basisCode("exp(-@0*@0/@1)") == 0


def test_a_basis_the_model_cannot_convolve_is_refused_with_roots_error(capsys: Any) -> None:
    models, t, tau, _ = _convolution_models()
    dm, w = RooRealVar("dm", "dm", 0.5), RooRealVar("w", "w", 0.1)
    tag = RooCategory("tag", "tag", {"B0": 1, "B0bar": -1})
    bcp = RooBCPGenDecay(
        "bg",
        "bg",
        t,
        tag,
        tau,
        dm,
        w,
        ref(0.2),
        ref(0.6),
        ref(0.0),
        ref(0.0),
        models["gauss"],
        "Flipped",
    )
    assert (
        "resolution model gauss doesn't support basis function exp(@0)/@1)"
        in capsys.readouterr().out
    )
    assert len(bcp.convs) == 2


#: Values - raw, and normalised over the time - at t = -3.3, -0.2, 0, 0.7, 4.1; for the mixing
#: decays also the value normalised over the time and both categories, mixed.
DECAY_VALUES = """
d_tm_SingleSided 0/0 0/0 1/0.647007318641685 0.636229213217603/0.41164495728543 0.070751693144826/0.0457768632709931
d_gm1_SingleSided 0.000825350989684408/0.000267101196974984 0.557347261302542/0.180369470062138 0.638534375694039/0.206643353178466 0.817606232493401/0.26459482823992 0.174286780826378/0.0564029223414908
d_gmsum_SingleSided 0.0782766146862089/0.0261862961098167 0.391434790207492/0.130948781639931 0.433327560525246/0.144963394980057 0.526138847673295/0.176012053092506 0.191136871341005/0.0639420436167806
d_tm_DoubleSided 0.118625602179468/0.0383758163941963 0.878799096649792/0.284294723574058 1/0.323503659320842 0.636229213217603/0.205822478642715 0.070751693144826/0.0228884316354966
d_gm1_DoubleSided 0.291960944308682/0.0472423966709085 1.26716486103651/0.205040798022735 1.27706875138808/0.206643353178466 1.16288151889391/0.188166640325601 0.174322910610109/0.0282073073553392
d_gmsum_DoubleSided 0.335000854724929/0.0560348171291971 0.861545452346662/0.144108712529635 0.866655121050493/0.144963394980057 0.807637005231628/0.135091571429233 0.25785708559078/0.0431311575261837
d_tm_Flipped 0.118625602179468/0.0767516327883927 0.878799096649792/0.568589447148116 1/0.647007318641685 0/0 0/0
d_gm1_Flipped 0.291135593318997/0.094217692144842 0.709817599733967/0.229712125983332 0.638534375694039/0.206643353178466 0.345275286400508/0.111738452411282 3.61297837304012e-05/1.16923691877167e-05
d_gmsum_Flipped 0.25672424003872/0.0858833381485774 0.470110662139171/0.157268643419338 0.433327560525246/0.144963394980057 0.281498157558333/0.094171089765959 0.0667202142497751/0.0223202714355868
bmix_tm 0.111442020647748/0.084189072631843 0.13495005698586/0.101948260478562 0.15/0.113317766685987 0.122963943366788/0.0928933296348439 0.0873863890680939/0.0660162029863278 0.00706745415959081
bmix_gm1 0.342531778572226/0.0376478768983269 2.1493615043846/0.236237635130871 2.1681192695738/0.238299312554421 1.95187791854427/0.214532093647546 0.158178952393167/0.0173855452256923 0.00699971033793794
bd_tm 0.162872932108608/0.0444876415763272 1.07939566970351/0.294829638363658 1.2/0.327771896781437 0.697618614987819/0.190549813887164 0.0669656344894849/0.0182912108631591
bds_tm 0/0 0/0 1.2/0.727206079856604 0.697618614987819/0.422760415200238 0.0669656344894849/0.0405815137851738
bd_gm1 0.398420757002074/0.0544401870009809 1.53252669045329/0.209404349914021 1.52459535530231/0.208320612781359 1.32110892537958/0.180516239885451 0.165007769037116/0.0225466511097174
bds_gm1 0.000962822561700445/0.000292557825615953 0.623742173622679/0.189526774006769 0.710487852676831/0.215884826108126 0.886167950815121/0.269265988494437 0.164963438641825/0.0501248587589685
"""  # noqa: E501


class Model:
    """The variables and models of ROOT's B-physics tutorials, built afresh for each test."""

    def __init__(self) -> None:
        self.dt = RooRealVar("dt", "dt", -10, 10)
        self.tau = RooRealVar("tau", "tau", 1.548)
        self.tm = RooTruthModel("tm", "truth model", self.dt)
        self.gm1 = RooGaussModel(
            "gm1",
            "gauss model 1",
            self.dt,
            RooRealVar("bias1", "bias1", 0),
            RooRealVar("sigma1", "sigma1", 1),
        )
        self.gm2 = RooGaussModel(
            "gm2",
            "gauss model 2",
            self.dt,
            RooRealVar("bias2", "bias2", 0),
            RooRealVar("sigma2", "sigma2", 5),
        )
        self.frac = RooRealVar("gm1frac", "fraction of gm1", 0.5)
        self.gmsum = RooAddModel("gmsum", "sum of gm1 and gm2", [self.gm1, self.gm2], [self.frac])
        self.dm = RooRealVar("dm", "dm", 0.472)
        self.w, self.dw = RooRealVar("w", "w", 0.1), RooRealVar("dw", "dw", 0.05)
        self.mix = RooCategory("mixState", "B0/B0bar mixing state", {"mixed": -1, "unmixed": 1})
        self.tag = RooCategory("tagFlav", "Flavour of the tagged B0", {"B0": 1, "B0bar": -1})

    def bmix(self, model: Any, kind: Any = "DoubleSided", name: str = "bmix") -> RooBMixDecay:
        return RooBMixDecay(
            name,
            "decay",
            self.dt,
            self.mix,
            self.tag,
            self.tau,
            self.dm,
            self.w,
            self.dw,
            model,
            kind,
        )

    def bdecay(self, model: Any, kind: Any, name: str = "bd") -> RooBDecay:
        f0, f1, f2, f3 = (
            RooRealVar(n, n, v) for n, v in (("f0", 1.0), ("f1", 0.3), ("f2", 0.2), ("f3", -0.4))
        )
        dg = RooRealVar("DG", "DG", 0.3)
        return RooBDecay(name, "bd", self.dt, self.tau, dg, f0, f1, f2, f3, self.dm, model, kind)


def _decay_pdfs(m: Model) -> list[Any]:
    pdfs: list[Any] = [
        RooDecay(f"d_{model.GetName()}_{kind}", "decay", m.dt, m.tau, model, kind)
        for kind in ("SingleSided", "DoubleSided", "Flipped")
        for model in (m.tm, m.gm1, m.gmsum)
    ]
    pdfs += [m.bmix(model, name=f"bmix_{model.GetName()}") for model in (m.tm, m.gm1)]
    for model in (m.tm, m.gm1):
        pdfs += [
            m.bdecay(model, 1, f"bd_{model.GetName()}"),
            m.bdecay(model, 0, f"bds_{model.GetName()}"),
        ]
    return pdfs


def _values(m: Model, pdf: Any) -> list[float]:
    """The raw and normalised values at ROOT's times - and, for a mixing decay, one more."""
    found = []
    for x in (-3.3, -0.2, 0.0, 0.7, 4.1):
        m.dt.setVal(x)
        found += [pdf.getVal(), pdf.getVal([m.dt])]
    if pdf.GetName().startswith("bmix"):
        m.mix.setIndex(-1)
        found.append(pdf.getVal([m.dt, m.mix, m.tag]))
        m.mix.setIndex(1)
    return found


def test_decays_have_roots_values_raw_and_normalised() -> None:
    m = Model()
    expected = {line.split()[0]: line.split()[1:] for line in DECAY_VALUES.strip().splitlines()}
    for pdf in _decay_pdfs(m):
        wanted = [float(v) for one in expected[pdf.GetName()] for v in one.split("/")]
        assert _values(m, pdf) == pytest.approx(wanted, rel=1e-13, abs=1e-300), pdf.GetName()


def test_the_coefficients_integrate_over_categories_in_closed_form_or_state_by_state() -> None:
    m = Model()
    dterr = RooRealVar("dterr", "per-event error on dt", 0.01, 10)
    dterr.setVal(0.8)
    # the bias and scale factor as a fit of ROOT's had left them, to the digits it printed
    gm = RooGaussModel(
        "gm1",
        "gauss model",
        m.dt,
        RooRealVar("bias", "bias", 0.04738592669, -10, 10),
        RooRealVar("sigma", "sigma", 1.051435324, 0.1, 10),
        dterr,
    )
    m.dt.setRange("part", -2, 3)
    m.dt.setVal(0.7)
    bmix = m.bmix(gm, RooBMixDecay.SingleSided)
    expected = {
        ("dt",): (1.32233496263, 0.746888691023),
        ("dt", "mixState"): (5.87146585823, 4.86952549679),
        ("dt", "tagFlav"): (2.95369444411, 1.75006819767),
        ("dt", "mixState", "tagFlav"): (12.3609807542, 10.2516326248),
    }
    for names, (full, part) in expected.items():
        over = [bmix.variable(name) for name in names]
        assert bmix.createIntegral(over).getVal() == pytest.approx(full, rel=1e-9)
        assert bmix.createIntegral(over, "part").getVal() == pytest.approx(part, rel=1e-9)
    S, C, mu = RooRealVar("S", "S", 0.6), RooRealVar("C", "C", 0.2), RooRealVar("mu", "mu", 0.05)
    bg = RooBCPGenDecay("bg", "bg", m.dt, m.tag, m.tau, m.dm, m.w, C, S, m.dw, mu, gm, 0)
    be = RooBCPEffDecay(
        "be",
        "be",
        m.dt,
        m.tag,
        m.tau,
        m.dm,
        m.w,
        RooRealVar("CP", "CP", -1),
        RooRealVar("al", "al", 0.9),
        RooRealVar("arg", "arg", 0.7),
        RooRealVar("eff", "eff", 1),
        m.dw,
        gm,
        0,
    )
    for pdf, numbers in (
        (bg, (0.145517562466, 6.22887934979, 5.1490674698)),
        (be, (0.15450546211, 5.59334379126, 4.63886376273)),
    ):
        found = (
            pdf.getVal([m.dt, m.tag]),
            pdf.createIntegral([m.dt, m.tag]).getVal(),
            pdf.createIntegral([m.dt, m.tag], "part").getVal(),
        )
        assert found == pytest.approx(numbers, rel=1e-9)


#: Per model: the first and last events generated, the sum of the times, and the next ``Rndm()``.
GENERATED = {
    "decay_tm_double": (
        [(-0.0007997519395403083,), (1.325508847123752,)],
        2.684131512175503,
        0.07614040211774409,
    ),
    "decay_gm_single": (
        [(3.807868722661886,), (0.6804017163586186,)],
        278.5859315570673,
        0.1595886864233762,
    ),
    "decay_sum_flipped": (
        [(-1.388262682904946,), (-2.3086361173317536,)],
        -314.4959853765157,
        0.6758617367595434,
    ),
    "bmix_tm_all": (
        [(1.1909572719408799, 1, 1), (-1.9101063495530066, 1, 1)],
        -4.1098439711974635,
        0.29502106085419655,
    ),
    "bmix_gm_mix": (
        [(3.3311122467298233, -1), (4.03438446039273, 1)],
        330.9672469499396,
        0.5358399641700089,
    ),
    "bmix_gm_tag": (
        [(3.3311122467298233, 1), (3.5375960811092986, 1)],
        533.0264402023781,
        0.8354308204725385,
    ),
    "bmix_tm_time": (
        [(-5.008408399054868,), (0.4569335357029741,)],
        12.896346987507524,
        0.3692947735544294,
    ),
    "bcp_aliased": (
        [(-0.1727564239079816, -1), (0.6274658521758895, 1)],
        -5.5632667069867,
        0.8015677570365369,
    ),
    "bcpgen_gm_flipped": (
        [(-0.014321849924444852, 1), (-0.5762109980634773, 1)],
        -314.30260638914666,
        0.02609453210607171,
    ),
    "bdecay_double": (
        [(-5.315163773286805, -1), (-4.041207033134564, -1)],
        160.2831227942124,
        0.10197304654866457,
    ),
    "bdecay_single": (
        [(0.0005330990963044686,), (1.159878574245063,)],
        595.5929339193758,
        0.07625561906024814,
    ),
    "bdecay_flipped": (
        [(-0.0005330990963044686,), (-0.12020357064825912,)],
        -341.16292983020423,
        0.5674598715268075,
    ),
}


def _generated(
    pdf: Any, variables: list[Any], n: int
) -> tuple[list[tuple[Any, ...]], float, float]:
    generator().SetSeed(4357)
    data = pdf.generate(variables, n)
    rows = []
    for i in (0, n - 1):
        row = data.get(i)
        rows.append(
            tuple(
                row.getRealValue("dt") if one.GetName() == "dt" else row.getCatIndex(one.GetName())
                for one in variables
            )
        )
    return rows, float(np.sum(data.column("dt"))), generator().Rndm()


def _generation_cases(m: Model) -> dict[str, tuple[Any, list[Any], int]]:
    gm2 = RooGaussModel("gm2", "gauss model 2", m.dt, ref(0), ref(3))
    gmsum = RooAddModel("gmsum", "sum", [m.gm1, gm2], [RooRealVar("frac", "frac", 0.5)])
    bmt, bmg = m.bmix(m.tm, "DoubleSided", "bmt"), m.bmix(m.gm1, "SingleSided", "bmg")
    cp, eff = RooRealVar("CPeigen", "CP eigen value", -1), RooRealVar("effR", "eff", 1)
    abs_lambda = RooRealVar("absLambda", "|lambda|", 1, 0, 2)
    arg_lambda = RooRealVar(
        "absLambda", "|lambda|", 0.7, -1, 1
    )  # rf708's name, the same as |lambda|'s
    bcp = RooBCPEffDecay(
        "bcp",
        "bcp",
        m.dt,
        m.tag,
        m.tau,
        m.dm,
        m.w,
        cp,
        abs_lambda,
        arg_lambda,
        eff,
        m.dw,
        m.tm,
        "DoubleSided",
    )
    bg = RooBCPGenDecay(
        "bg",
        "bg",
        m.dt,
        m.tag,
        m.tau,
        m.dm,
        m.w,
        RooRealVar("C", "C", 0.2),
        RooRealVar("S", "S", 0.6),
        m.dw,
        RooRealVar("mu", "mu", 0.05),
        m.gm1,
        "Flipped",
    )
    dg = RooFormulaVar(
        "DG", "Delta Gamma", "@1/@0", [m.tau, RooRealVar("DGbG", "DGamma/GammaAvg", 0.5, -1, 1)]
    )
    fsin = RooFormulaVar(
        "fsin", "fsin", "@0*@1*(1-2*@2)", [RooRealVar("Amix", "Amix", 0.7), m.tag, m.w]
    )
    fcos = RooFormulaVar(
        "fcos", "fcos", "@0*@1*(1-2*@2)", [RooRealVar("Adir", "Adir", 0), m.tag, m.w]
    )
    fsinh = RooRealVar("fsinh", "fsinh", 0.7)

    def bd(kind: str) -> RooBDecay:
        return RooBDecay(
            "bd" + kind, "bd", m.dt, m.tau, dg, ref(1), fsinh, fcos, fsin, m.dm, m.tm, kind
        )

    return {
        "decay_tm_double": (RooDecay("d1", "d", m.dt, m.tau, m.tm, "DoubleSided"), [m.dt], 200),
        "decay_gm_single": (RooDecay("d2", "d", m.dt, m.tau, m.gm1, "SingleSided"), [m.dt], 200),
        "decay_sum_flipped": (RooDecay("d3", "d", m.dt, m.tau, gmsum, "Flipped"), [m.dt], 200),
        "bmix_tm_all": (bmt, [m.dt, m.mix, m.tag], 300),
        "bmix_gm_mix": (bmg, [m.dt, m.mix], 200),
        "bmix_gm_tag": (bmg, [m.dt, m.tag], 200),
        "bmix_tm_time": (bmt, [m.dt], 200),
        "bcp_aliased": (bcp, [m.dt, m.tag], 200),
        "bcpgen_gm_flipped": (bg, [m.dt, m.tag], 200),
        "bdecay_double": (bd("DoubleSided"), [m.dt, m.tag], 300),
        "bdecay_single": (bd("SingleSided"), [m.dt], 300),
        "bdecay_flipped": (bd("Flipped"), [m.dt], 300),
    }


@pytest.mark.parametrize("case", sorted(GENERATED))
def test_decays_generate_roots_events_taking_roots_random_numbers(case: str) -> None:
    pdf, variables, n = _generation_cases(Model())[case]
    rows, total, following = _generated(pdf, variables, n)
    expected_rows, expected_total, expected_following = GENERATED[case]
    assert rows == expected_rows
    assert total == pytest.approx(expected_total, rel=1e-13)
    assert following == expected_following


def _per_event_errors() -> tuple[Any, ...]:
    dt = RooRealVar("dt", "dt", -10, 10)
    dterr = RooRealVar("dterr", "per-event error on dt", 0.01, 10)
    bias = RooRealVar("bias", "bias", 0, -10, 10)
    sigma = RooRealVar("sigma", "per-event error scale factor", 1, 0.1, 10)
    gm = RooGaussModel("gm1", "gauss model", dt, bias, sigma, dterr)
    decay = RooDecay(
        "decay_gm", "decay", dt, RooRealVar("tau", "tau", 1.548), gm, RooDecay.DoubleSided
    )
    landau = RooLandau("pdfDtErr", "pdfDtErr", dterr, ref(1.0), ref(0.25))
    return dt, dterr, bias, sigma, decay, landau


def _prototype_generated() -> tuple[Any, ...]:
    dt, dterr, bias, sigma, decay, landau = _per_event_errors()
    generator().SetSeed(4357)
    proto = landau.generate([dterr], 400)
    data = decay.generate([dt], RooCmdArg("ProtoData", proto))
    return dt, dterr, bias, sigma, decay, proto, data


def test_a_decay_with_per_event_errors_is_generated_from_prototype_data_as_root_does() -> None:
    *_, data = _prototype_generated()
    assert [one.GetName() for one in data.get()] == ["dt", "dterr"]
    assert data.numEntries() == 400
    first, last = data.get(0).getRealValue("dt"), data.get(399).getRealValue("dt")
    assert (first, last, data.get(399).getRealValue("dterr")) == (
        4.15910020411669,
        2.522741547072659,
        2.5295921846159857,
    )


def test_a_decay_with_per_event_errors_is_fitted_and_averaged_over_them_as_root_does(
    capsys: Any,
) -> None:
    from xrdroot.roofit.plot.frame import make_frame

    dt, dterr, bias, sigma, decay, proto, data = _prototype_generated()
    decay.fitTo(data, ConditionalObservables=[dterr], PrintLevel=-1)
    assert (bias.getVal(), sigma.getVal()) == pytest.approx((0.04738592669, 1.051435324), rel=1e-9)
    name = "gm1_conv_exp(-abs(@0)/@1)_dt_tau_[decay_gm]"
    assert f"RooAbsPdf::fitTo({name}_over_{name}_Int[dt]) fixing" in capsys.readouterr().out
    frame = make_frame(dt, (), {})
    data.plotOn(frame)
    decay.plotOn(frame, ProjWData=(proto, True))
    decay.plotOn(frame, ProjWData=(proto, False))
    assert "plot on dt averages using data variables (dterr)" in capsys.readouterr().out
    found = [frame.getObject(i).interpolate(x) for i in (1, 2) for x in (-3.0, 0.2, 4.5)]
    expected = [5.007218795, 13.90434818, 2.518193987, 5.006274935, 13.90868903, 2.517485952]
    assert found == pytest.approx(expected, rel=1e-9)


def test_the_gaussian_integrates_over_a_flat_scale_factor_and_asymptotically_when_told_to() -> None:
    dt = RooRealVar("dt", "dt", -10, 10)
    tau = RooRealVar("tau", "tau", 1.548)
    bias, sigma = (
        RooRealVar("bias", "bias", 0.04738592669),
        RooRealVar("sigma", "sigma", 1.051435324),
    )
    ssf = RooRealVar("ssf", "ssf", 1.2, 0.5, 2.0)
    flat = RooGaussModel("g2", "g2", dt, bias, sigma, ref(1.0), ssf)
    flat.advertiseFlatScaleFactorIntegral(True)
    basis = RooFormulaVar("exp(-abs(@0)/@1)_b", "exp(-abs(@0)/@1)", [dt, tau])
    conv = flat.convolution(basis, dt)
    assert conv.createIntegral([dt, ssf]).getVal() == pytest.approx(9.26772953183, rel=1e-9)
    assert flat.createIntegral([dt, ssf]).getVal() == pytest.approx(1.5, rel=1e-12)
    assert flat.integral_code(frozenset(["dt", "ssf"])) == 2
    asymptotic = RooGaussModel("g3", "g3", dt, bias, sigma)
    asymptotic.advertiseAymptoticIntegral(True)
    assert asymptotic.convolution(basis, dt).createIntegral([dt]).getVal() == pytest.approx(
        6.192, rel=1e-12
    )
    assert asymptotic.createIntegral([dt]).getVal() == 1.0
    other = RooGaussModel("g4", "g4", dt, ref(0.5), ref(2.0))
    summed = RooAddModel("am", "am", [asymptotic, other], [RooRealVar("frac", "frac", 0.3)])
    dt.setVal(0.7)
    assert summed.getVal([dt]) == pytest.approx(0.232817590232, rel=1e-9)
    assert summed.printMetaArgs() == "frac * g3 + g4 "


def test_the_truth_model_evaluates_a_basis_it_has_no_closed_form_for() -> None:
    dt, tau = RooRealVar("dt", "dt", -10, 10), RooRealVar("tau", "tau", 1.548)
    dt.setVal(0.7)
    truth = RooTruthModel("tm", "tm", dt)
    conv = truth.convolution(RooFormulaVar("gen_b", "exp(-@0*@0/@1)", [dt, tau]), dt)
    assert conv.getVal() == pytest.approx(0.728667709545, rel=1e-11)
    assert conv.createIntegral([dt]).getVal() == pytest.approx(2.20526312019, rel=1e-10)
    plain = RooTruthModel("tm0", "tm0", dt)
    assert plain.getVal() == 0.0
    assert plain.createIntegral([dt]).getVal() == 1.0
    assert plain.generate_event(1, generator()) == {"dt": 0.0}


def test_a_slice_holds_categories_and_the_rest_are_integrated_out_as_root_says(capsys: Any) -> None:
    m = Model()
    bmix = m.bmix(m.tm)
    generator().SetSeed(4357)
    data = bmix.generate([m.dt, m.mix, m.tag], 1000)
    frame = m.dt.frame()
    data.plotOn(frame, Cut="tagFlav==tagFlav::B0")
    bmix.plotOn(frame, Slice=(m.tag, "B0"))
    bmix.plotOn(frame, Slice={m.tag: "B0bar", m.mix: "mixed"})
    out = capsys.readouterr().out
    assert "plotting 475 events out of 1000 total events" in out
    assert "RooAbsReal::plotOn(bmix) plot on dt represents a slice in (tagFlav)" in out
    assert "RooAbsReal::plotOn(bmix) plot on dt integrates over variables (mixState)" in out
    assert "represents a slice in (mixState,tagFlav)" in out
    assert (m.tag.getLabel(), m.mix.getLabel()) == ("B0bar", "mixed")
    curve = frame.getObject(1)
    assert curve.interpolate(1.0) == pytest.approx(REFERENCE_SLICE, rel=1e-6)


def test_an_asymmetry_curve_is_red_and_projects_the_other_categories(capsys: Any) -> None:
    m = Model()
    bmix = m.bmix(m.gm1)
    generator().SetSeed(4357)
    data = bmix.generate([m.dt, m.tag, m.mix], 500)
    frame = m.dt.frame()
    data.plotOn(frame)
    bmix.plotOn(frame, Asymmetry=m.mix)
    out = capsys.readouterr().out
    assert "RooAbsReal::plotAsymOn(bmix) plot on dt projects variables (tagFlav)" in out
    curve = frame.getObject(1)
    assert curve.GetName() == "bmix_Asym[mixState]"
    assert curve._core["TAttLine"]["fLineColor"] == 2
    assert [curve.interpolate(x) for x in (-2.0, 0.5, 3.0)] == pytest.approx(
        REFERENCE_ASYMMETRY, rel=1e-6
    )


def test_an_asymmetry_needs_a_category_of_signs_the_density_depends_on(capsys: Any) -> None:
    m = Model()
    decay = RooDecay("decay", "decay", m.dt, m.tau, m.tm, "DoubleSided")
    frame = m.dt.frame()
    decay.plotOn(frame, Asymmetry=m.mix)
    assert "function doesn't depend on asymmetry category mixState" in capsys.readouterr().out
    three = RooCategory("three", "three", {"a": 0, "b": 1, "c": 2})
    bmix = RooBMixDecay("bmix", "decay", m.dt, three, m.tag, m.tau, m.dm, m.w, m.dw, m.tm)
    bmix.plotOn(frame, Asymmetry=three)
    assert (
        "asymmetry category must have 2 or 3 states with index values -1,0,1"
        in capsys.readouterr().out
    )
    assert frame.numItems() == 0


def test_a_conditional_product_is_generated_fitted_and_projected_with_roots_integrals(
    capsys: Any,
) -> None:
    from xrdroot.roofit.pdfs.histpdf import RooHistPdf
    from xrdroot.roofit.pdfs.prodpdf import RooProdPdf

    dt, dterr, bias, sigma, decay, landau = _per_event_errors()
    generator().SetSeed(4357)
    errors = landau.generate([dterr], 2000)
    pdf_err = RooHistPdf("pdfErr", "pdfErr", [dterr], errors.binnedClone())
    model = RooProdPdf("model", "model", [pdf_err], Conditional=([decay], [dt]))
    data = model.generate([dt, dterr], 300)
    assert (
        data.get(0).getRealValue("dt"),
        data.get(0).getRealValue("dterr"),
    ) == REFERENCE_PRODUCT_EVENT
    model.fitTo(data, PrintLevel=-1)
    assert (bias.getVal(), sigma.getVal()) == pytest.approx(REFERENCE_PRODUCT_FIT, rel=1e-6)
    frame = dt.frame()
    data.plotOn(frame)
    model.plotOn(frame)
    out = capsys.readouterr().out
    assert "RooAbsPdf::fitTo(model) fixing normalization set" in out
    assert (
        "RooRealIntegral::init(SPECINT[pdfErr_NORM[dterr]_X_decay_gm_NORM[dt]]_Int[dterr]) using "
        "numeric integrator RooIntegrator1D to calculate Int(dterr)" in out
    )
    assert "model_Int" not in out
    assert frame.getObject(1).interpolate(0.5) == pytest.approx(REFERENCE_PRODUCT_CURVE, rel=1e-3)


def test_a_resolution_with_more_observables_leaves_the_decay_to_numerical_sampling() -> None:
    from xrdroot.roofit.generation.contexts import NumericContext
    from xrdroot.roofit.generation.convolution import context_for_convolution

    dt, dterr, _, _, decay, _ = _per_event_errors()
    generator().SetSeed(4357)
    assert isinstance(context_for_convolution(decay, frozenset(["dt", "dterr"])), NumericContext)
    data = decay.generate([dt, dterr], 5)
    assert data.numEntries() == 5


def test_a_decay_whose_envelope_fails_refuses_to_draw() -> None:
    m = Model()
    bad = RooBDecay(
        "bad",
        "bad",
        m.dt,
        m.tau,
        ref(0.0),
        ref(1.0),
        ref(0.0),
        ref(-3.0),
        ref(0.0),
        m.dm,
        m.tm,
        "SingleSided",
    )
    generator().SetSeed(4357)
    with pytest.raises(RuntimeError, match="below zero or above its envelope"):
        bad.generate([m.dt], 10)


def test_an_add_model_needs_as_many_fractions_as_models_or_one_fewer() -> None:
    m = Model()
    with pytest.raises(ValueError, match="inconsistent"):
        RooAddModel("bad", "bad", [m.gm1], [m.frac, m.frac])


def test_set_bin_puts_a_variable_at_a_bin_centre_or_says_the_bin_is_out_of_range(
    capsys: Any,
) -> None:
    dterr = RooRealVar("dterr", "dterr", 0.01, 10)
    dterr.setBin(20)
    assert dterr.getVal() == pytest.approx(0.01 + 20.5 * (10 - 0.01) / 100, rel=1e-15)
    dterr.setBin(100)
    assert "ERROR: bin index 100 is out of range (0,99)" in capsys.readouterr().out


class _Draws:
    """A generator that hands out the numbers it is given, then a Gaussian's mean."""

    def __init__(self, *numbers: float) -> None:
        self.numbers = list(numbers)

    def Rndm(self) -> float:
        return self.numbers.pop(0)

    def Gaus(self, mean: float, sigma: float) -> float:
        return self.numbers.pop(0) if self.numbers else mean


def test_a_sum_of_resolutions_draws_its_component_and_draws_again_on_a_threshold() -> None:
    m = Model()
    assert m.gmsum.generate_event(1, _Draws(0.5, 0.25, 3.0)) == {"dt": 3.0}
    assert m.gmsum.generate_event(1, _Draws(0.75, 20.0, -2.0), (-10.0, 10.0)) == {"dt": -2.0}
    both = RooAddModel(
        "both", "both", [m.gm1, m.gm2], [RooRealVar("a", "a", 0.2), RooRealVar("b", "b", 0.8)]
    )
    assert both.fractions({}) == [0.2, 0.8]
    assert m.gmsum.selfNormalized()
    m.dt.setVal(0.3)
    root2pi = math.sqrt(2 * math.pi)
    expected = math.exp(-0.045) / root2pi * 0.5 + math.exp(-0.0018) / (5 * root2pi) * 0.5
    assert m.gmsum.getVal() == pytest.approx(expected, rel=1e-15)


def test_a_decay_answers_roots_questions_about_itself() -> None:
    m = Model()
    decay = RooDecay("decay", "decay", m.dt, m.tau, m.gm1, "DoubleSided")
    assert decay.convVar() is m.dt
    conv = decay.convs[0]
    assert conv.basis().GetName() == "exp(-abs(@0)/@1)_dt_tau"
    assert not decay.is_direct_gen_safe("w")
    assert not decay.is_direct_gen_safe("tau")  # the convolutions depend on it too
    conv.changeBasis(None)
    assert (conv.basis(), conv._basis_code) == (None, 0)


def test_a_mixing_decay_answers_roots_questions_about_itself() -> None:
    m = Model()
    coef = m.bmix(m.tm)
    assert coef.is_direct_gen_safe("tagFlav")
    assert coef.is_direct_gen_safe("dt")  # the truth model's time, whatever depends on it
    assert not m.gm1.is_direct_gen_safe("tau")
    assert coef.normalized_name([m.dt]) == "bmix_over_bmix_Int[dt]"
    from xrdroot.roofit.pdfs.anaconv import CoefVar

    assert CoefVar(coef, 1).getVal() == pytest.approx(-0.8, rel=1e-15)
    m.w.setVal(0.5)  # no dilution: the oscillation's coefficient vanishes, and its term is skipped
    m.dt.setVal(0.4)
    assert coef.getVal() == pytest.approx((1 - 0.05) * math.exp(-0.4 / 1.548), rel=1e-15)


def test_generators_draw_only_what_they_are_asked_for() -> None:
    m = Model()
    bmix, cp = (
        m.bmix(m.tm),
        RooBCPEffDecay(
            "be",
            "be",
            m.dt,
            m.tag,
            m.tau,
            m.dm,
            m.w,
            ref(-1.0),
            ref(1.0),
            ref(0.7),
            ref(1.0),
            m.dw,
            m.tm,
        ),
    )
    assert bmix.gen_code(frozenset(["tagFlav"]), True) == (0, frozenset())
    assert cp.gen_code(frozenset(["tagFlav"]), True) == (0, frozenset())
    cp.init_generator(1)
    generator().SetSeed(4357)
    data = cp.generate([m.dt], 50)
    assert data.numEntries() == 50
    generator().SetSeed(4357)
    tags = bmix.generate([m.tag], 40)
    assert set(np.unique(tags.column("tagFlav"))) == {-1.0, 1.0}
    narrow = RooGaussModel("narrow", "narrow", m.dt, ref(9.9), ref(1.0))
    event = narrow.generate_event(1, _Draws(10.5, 9.95), (-10.0, 10.0))
    assert event == {"dt": 9.95}


def test_a_resolution_that_cannot_draw_itself_leaves_the_decay_to_numerical_sampling() -> None:
    from xrdroot.roofit.generation.contexts import NumericContext
    from xrdroot.roofit.generation.convolution import context_for_convolution

    m = Model()
    shifted = RooGaussModel(
        "shifted", "shifted", m.dt, RooFormulaVar("mean", "mean", "0.1*@0", [m.dt]), ref(1.0)
    )
    decay = RooDecay("decay", "decay", m.dt, m.tau, shifted, "SingleSided")
    assert isinstance(context_for_convolution(decay, frozenset(["dt"])), NumericContext)

    class Undrawable(RooDecay):
        def gen_code(self, direct: frozenset[str], static_ok: bool) -> tuple[int, frozenset[str]]:
            return 0, frozenset()

    stubborn = Undrawable("stubborn", "stubborn", m.dt, m.tau, m.gm1, "SingleSided")
    assert isinstance(context_for_convolution(stubborn, frozenset(["dt"])), NumericContext)


def test_a_gaussian_convolution_of_per_event_lifetimes_is_evaluated_event_by_event() -> None:
    m = Model()
    decay = RooDecay("decay", "decay", m.dt, m.tau, m.gm1, "DoubleSided")
    conv = decay.convs[0]
    taus = np.array([1.2, 1.548])
    found = conv.compute({"dt": np.array([0.5, 0.5]), "tau": taus})
    m.dt.setVal(0.5)
    single = []
    for one in taus:
        m.tau.setVal(one)
        single.append(conv.getVal())
    assert list(found) == single


def test_a_projection_can_average_over_part_of_a_dataset_and_slices_need_no_data() -> None:
    from xrdroot.roofit.cmdargs import commands
    from xrdroot.roofit.plot.projections import view

    dt, dterr, _, _, decay, landau = _per_event_errors()
    generator().SetSeed(4357)
    proto = landau.generate([dterr], 50)
    frame = dt.frame()
    data = decay.generate([dt], RooCmdArg("ProtoData", proto))
    data.plotOn(frame)
    seen = view(decay, frame, commands([RooCmdArg("ProjWData", [dterr], proto, True)]))
    assert (seen.averaged, seen.projected, seen.binned) == (["dterr"], [], True)
    assert view(decay, dt.frame(), commands([])).projected == []


def test_an_asymmetry_with_nothing_else_to_project_says_nothing_of_projections(capsys: Any) -> None:
    m = Model()
    bmix = m.bmix(m.tm)
    generator().SetSeed(4357)
    data = bmix.generate([m.dt, m.mix], 100)
    frame = m.dt.frame()
    data.plotOn(frame)
    bmix.plotOn(frame, Asymmetry=m.mix, LineColor=4)
    assert "projects variables" not in capsys.readouterr().out
    assert frame.getObject(1)._core["TAttLine"]["fLineColor"] == 4


def test_a_conditional_product_integrates_factor_by_factor_where_it_can() -> None:
    from xrdroot.roofit.pdfs.histpdf import RooHistPdf
    from xrdroot.roofit.pdfs.prodpdf import RooProdPdf

    dt, dterr, _, _, decay, landau = _per_event_errors()
    generator().SetSeed(4357)
    pdf_err = RooHistPdf("pdfErr", "pdfErr", [dterr], landau.generate([dterr], 500).binnedClone())
    model = RooProdPdf("model", "model", [pdf_err], Conditional=([decay], [dt]))
    nset = frozenset(["dt", "dterr"])
    assert not model.announce_projection(frozenset(["dt"]), nset)
    dterr.setVal(1.0)
    ctx = {"dt": np.array([0.2])}
    assert model.fraction(frozenset(["dt"]), ctx, nset, None) == pytest.approx(
        pdf_err.value(ctx, frozenset(["dterr"])), rel=1e-12
    )
    assert model.normalized_name([dt, dterr]) == "model"
    assert model.analytic_names(nset, None) == nset
    assert float(np.asarray(model.integrate(nset, {}))) == pytest.approx(1.0, rel=1e-6)
