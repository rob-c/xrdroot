"""Interactive namespace construction belongs to the application, with a session adapter."""

import xrdroot
from xrdroot import apinamespace, plugins, session


def test_session_names_are_supplied_by_the_declared_application_adapter(monkeypatch):
    expected = {"value": object()}
    points = []

    def resolve(point):
        points.append(point)
        return lambda: expected

    monkeypatch.setattr(plugins, "require", resolve)
    assert session.preloaded() is expected
    assert points == ["session.preloaded"]


def test_application_names_follow_live_exports_and_retain_module_identity(monkeypatch):
    expected = object()
    monkeypatch.setattr(xrdroot, "Histogram", expected)
    found = apinamespace.preloaded()
    assert found["Histogram"] is expected
    assert found["xrdroot"] is xrdroot
    assert set(found) == set(xrdroot.__all__) | {"xrdroot", "np"}
    assert found["np"].__name__ == "numpy"
