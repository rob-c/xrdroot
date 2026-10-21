"""Core display forwards through declared providers and has a pure HTML fallback."""

from __future__ import annotations

from xrdroot import display, plugins


def test_plot_forwards_the_object_axes_backend_options_and_style_to_its_provider(monkeypatch):
    seen = []
    made, axes, result = display.Displayed(), object(), object()

    def draw(*args, **kwargs):
        seen.append((args, kwargs))
        return result

    def required(point):
        assert point == "display.plot"
        return draw

    monkeypatch.setattr(display, "require", required)
    assert made.plot(axes, "text", "E1", color="red") is result
    assert seen == [((made, axes, "text", "E1"), {"color": "red"})]


def test_notebook_html_uses_its_provider_and_keeps_the_plain_representation(monkeypatch):
    made = display.Displayed()
    seen = []

    def figure(obj):
        seen.append(obj)
        return "<svg>picture</svg>"

    def found(point):
        assert point == "display.html"
        return figure

    monkeypatch.setattr(display, "lookup", found)
    assert made._repr_mimebundle_() == {"text/plain": repr(made), "text/html": "<svg>picture</svg>"}
    assert seen == [made]


def test_without_an_html_provider_the_notebook_escapes_the_object_representation(monkeypatch):
    class Sample(display.Displayed):
        def __repr__(self):
            return "<object>&"

    monkeypatch.delitem(plugins.PROVIDERS, ("display.html", "default"))
    assert Sample()._repr_html_() == "<pre>&lt;object&gt;&amp;</pre>"
