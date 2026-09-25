"""ROOT's drawing options, colours, markers and styles, read before anything is drawn."""

from __future__ import annotations

import pytest

from xrdroot import Histogram
from xrdroot.plot import colors
from xrdroot.plot.attributes import found, look, restyled, split_style
from xrdroot.plot.model import Look
from xrdroot.plot.options import choose

# -- the option string ----------------------------------------------------------------------------


def test_an_option_is_split_into_the_longest_words_at_each_place():
    assert choose("colz", "two-dimensional histogram").words == {"COL", "Z"}
    assert choose("E1SAME", "histogram").words == {"E1", "SAME"}
    assert choose(" hist , same ", "histogram").words == {"HIST", "SAME"}
    assert choose("AP", "graph").words == {"A", "P"}
    assert choose("LEGO2Z", "two-dimensional histogram").words == {"LEGO", "Z"}


def test_text_keeps_the_angle_roots_numbers_are_written_at():
    chosen = choose("TEXT45", "histogram")
    assert chosen.words == {"TEXT"} and chosen.angle == 45.0
    assert choose("TEXT", "histogram").angle == 0.0


def test_what_is_not_an_option_is_refused_with_what_is_left():
    with pytest.raises(ValueError, match="'QQ', which begins with no drawing option"):
        choose("EQQ", "histogram")


def test_an_option_root_has_but_this_does_not_draw_is_refused_with_why():
    with pytest.raises(ValueError, match="'SCAT' is not drawn here: ROOT itself has retired"):
        choose("SCAT", "two-dimensional histogram")
    with pytest.raises(ValueError, match="pass logy=True"):
        choose("HIST LOGY", "histogram")


def test_an_option_for_another_kind_of_thing_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="'COL' does not draw a histogram"):
        choose("COLZ", "histogram")
    with pytest.raises(ValueError, match="does not draw a graph"):
        choose("E1", "graph")


def test_the_drawing_words_leave_out_where_and_how_much():
    chosen = choose("SAME NORM PLC", "histogram")
    assert chosen.drawing == frozenset()
    assert chosen.without("SAME").words == {"NORM", "PLC"}
    assert chosen.has("NORM", "E") and not chosen.has("E")


# -- colours --------------------------------------------------------------------------------------


def test_a_colour_index_is_roots_colour_to_the_bit():
    assert colors.color(2) == "#ff0000"
    assert colors.color(10) == "#fefefe"  # ROOT's "white" is not quite
    assert colors.color(41) == "#d3ce87"
    assert colors.color(633) == "#cc0000"  # kRed+1
    assert colors.color(5000) == colors.UNKNOWN


def test_a_colour_by_its_ecolor_name_is_offset_as_a_macro_offsets_it():
    assert colors.color("kRed+1") == colors.color(633)
    assert colors.color("kAzure - 3") == colors.color(857)
    assert colors.color("kP10Blue") == colors.color(117)
    assert colors.color("kRed+40") == colors.UNKNOWN


def test_a_colour_that_is_not_roots_is_left_for_the_library():
    assert colors.color("crimson") == "crimson"
    assert colors.color("kNotAColour") == "kNotAColour"
    assert colors.color(None) is None


def test_the_palettes_are_roots_255_colours_by_name_or_number():
    assert len(colors.palette("bird")) == 255
    assert colors.palette(57) == colors.palette("kBird") == colors.PALETTES["bird"]
    assert colors.palette(112) == colors.PALETTES["viridis"]
    assert colors.palette("magma") is None and colors.palette(3) is None


def test_several_things_are_coloured_evenly_across_the_palette():
    bird = colors.PALETTES["bird"]
    assert colors.palette_color(0, 1) == bird[0]
    assert colors.palette_color(2, 3) == bird[-1]
    assert colors.palette_color(1, 3, "nonsense") == bird[127]


# -- the attributes -------------------------------------------------------------------------------


def _styled(**attributes) -> Histogram:
    made = Histogram.new("h", [0, 1, 2], [1, 2])
    for group, values in attributes.items():
        made.members["TH1"][group].update(values)
    return made


def test_the_attributes_are_found_however_far_down_they_are_inherited():
    assert found({"a": {"TAttLine": {"fLineColor": 2}}}, "TAttLine") == {"fLineColor": 2}
    assert found({"a": 1, "b": {"c": 2}}, "TAttLine") is None
    assert found([], "TAttLine") is None


def test_a_histograms_attributes_become_its_look():
    styled = _styled(
        TAttLine={"fLineColor": 4, "fLineWidth": 3, "fLineStyle": 2},
        TAttMarker={"fMarkerStyle": 24, "fMarkerColor": 2, "fMarkerSize": 1.5},
        TAttFill={"fFillColor": 5, "fFillStyle": 1001},
    )
    made = look(styled.members, markers=True)
    assert (made.color, made.width, made.dash) == ("#0000ff", 3.0, "dashed")
    assert (made.marker, made.hollow, made.marker_color, made.marker_size) == (
        "circle", True, "#ff0000", 1.5,
    )  # fmt: skip
    assert made.fill == "#ffff00" and look(styled.members).marker is None


@pytest.mark.parametrize(
    ("style", "fill", "alpha", "hatch"),
    [(0, None, 1.0, None), (3004, "#ff0000", 1.0, "/"), (3999, "#ff0000", 1.0, "/"),
     (4050, "#ff0000", 0.5, None), (1001, "#ff0000", 1.0, None)],
)  # fmt: skip
def test_a_fill_style_is_hollow_hatched_translucent_or_solid(style, fill, alpha, hatch):
    made = look(_styled(TAttFill={"fFillColor": 2, "fFillStyle": style}).members)
    assert (made.fill, made.alpha, made.hatch) == (fill, alpha, hatch)


def test_fill_colour_zero_is_hollow_and_an_unknown_marker_a_circle():
    made = look(_styled(TAttMarker={"fMarkerStyle": 99}).members, markers=True)
    assert made.fill is None and made.marker == "circle"


def test_the_callers_keywords_win_over_every_attribute():
    base = Look(fill="#00ff00")
    made = restyled(base, {"color": 2, "linewidth": 2, "linestyle": 3, "marker": 20, "ms": 2})
    assert (made.color, made.marker_color, made.width, made.dash) == (
        "#ff0000", "#ff0000", 2.0, "dotted",
    )  # fmt: skip
    assert (made.marker, made.hollow, made.marker_size) == ("circle", False, 2.0)
    assert restyled(base, {"color": "red", "markercolor": "blue"}).marker_color == "blue"
    assert restyled(base, {"marker": "s", "ls": "--"})[6:8] == ("square", False)
    assert restyled(base, {"ls": "--"}).dash == "dashed"


def test_fill_true_fills_with_the_fill_or_the_line_and_false_not_at_all():
    assert restyled(Look(fill="#00ff00"), {"fill": True}).fill == "#00ff00"
    assert restyled(Look(color="#123456"), {"fill": True}).fill == "#123456"
    assert restyled(Look(fill="#00ff00"), {"fill": False}).fill is None
    assert restyled(Look(), {"fill": "kRed"}).fill == "#ff0000"


def test_a_marker_or_a_dash_that_is_not_drawn_is_refused_with_the_ones_that_are():
    with pytest.raises(ValueError, match=r"marker='hexagon'.*circle"):
        restyled(Look(), {"marker": "hexagon"})
    with pytest.raises(ValueError, match=r"linestyle='wavy'.*dashdot"):
        restyled(Look(), {"linestyle": "wavy"})


def test_the_keywords_are_sorted_into_the_look_the_frame_and_the_librarys():
    marks, frame, native = split_style({"color": 1, "logy": True, "zorder": 3})
    assert (marks, frame, native) == ({"color": 1}, {"logy": True}, {"zorder": 3})
