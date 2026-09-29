"""Miroslav Morhac's spectrum processing - ROOT's ``TSpectrum`` family - step for step.

ROOT's ``hist/spectrum`` estimates a background under peaks by clipping
(SNIP), smooths by a Markov chain, sharpens peaks by Gold's and
Richardson-Lucy's deconvolution, finds peaks in what that leaves, fits them,
and transforms spectra into Haar, Walsh, cosine, Fourier and the other
bases. These modules are those algorithms over NumPy arrays of ``float64``,
the C++'s ``Double_t``: each loop that ROOT runs element by element is run
here over the whole array at once, but with every element's arithmetic
done in the order ROOT does it - a sum added term by term from the same
first term, a ``/ 6`` where ROOT divides by six - so that what comes out is
ROOT's to the last bit, quirks and all. Where a loop feeds on its own
results, it is a Python loop over Python floats, which are C doubles.

:mod:`xrdroot.pyroot.spectra` is ROOT's classes over these.
"""
