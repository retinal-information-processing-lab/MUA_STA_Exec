"""Progress reporting, and stopping a run through it."""
import time

import pytest

from mua_sta.progress import Cancelled, Reporter, format_duration


def collect():
    seen = []
    return seen, Reporter(lambda text, fraction: seen.append((text, fraction)))


def test_duration_reads_as_a_clock():
    assert format_duration(0) == "00:00"
    assert format_duration(7.4) == "00:07"
    assert format_duration(95) == "01:35"
    assert format_duration(3725) == "1:02:05"
    assert format_duration(-3) == "00:00"


def test_it_reports_a_percentage_and_a_time_left():
    seen, report = collect()
    report(0, 100, "reading")
    time.sleep(0.2)
    report(50, 100, "reading")
    text, fraction = seen[-1]
    assert fraction == pytest.approx(0.5)
    assert "reading" in text and "50%" in text and "<" in text


def test_it_does_not_redraw_faster_than_the_eye(monkeypatch):
    seen, report = collect()
    for done in range(0, 50):
        report(done, 100, "reading")
    assert len(seen) < 10              # the first, then throttled
    report(100, 100, "reading")
    assert seen[-1][1] == 1.0          # but finishing is always shown


def test_a_new_task_restarts_the_clock_and_is_shown_at_once():
    seen, report = collect()
    report(0, 10, "first")
    report(1, 10, "second")
    assert [text.split(" ")[0] for text, _ in seen] == ["first", "second"]


def test_a_task_that_cannot_measure_itself_still_shows_its_time():
    seen, report = collect()
    report.note("detecting")
    text, fraction = seen[-1]
    assert fraction == -1.0
    assert text.startswith("detecting") and "%" not in text


def test_stopping_raises_at_the_next_report():
    seen, report = collect()
    report(1, 10, "work")
    assert not report.cancelled
    report.cancel()
    assert report.cancelled
    with pytest.raises(Cancelled):
        report(2, 10, "work")


def test_a_step_unwinds_when_it_is_stopped():
    """Stopping rides on the progress calls, so a step ends through its own finally blocks."""
    seen, report = collect()
    closed = []

    def step(progress):
        try:
            for done in range(100):
                progress(done, 100, "work")
                time.sleep(0.02)
        finally:
            closed.append(True)

    report.cancel()
    with pytest.raises(Cancelled):
        step(report)
    assert closed == [True]
