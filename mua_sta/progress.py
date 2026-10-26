"""How far along the thing that is running is, and when it will be done.

The steps report as tqdm does — how many units of the current task are finished out of how
many — and :class:`Reporter` turns that into a percentage, the time spent and the time left,
measured from the rate so far. Only the running task is described: a weighted guess across a
whole run would have to invent the relative cost of reading a file, detecting spikes and
rebuilding a stimulus, and would be wrong on the first unusual recording.

Steps call ``progress(done, total, label)``. ``total`` of 0 means the task cannot say how long
it is, and only the elapsed time is shown.
"""
from __future__ import annotations

import time
from typing import Callable

#: Do not redraw more often than this, in seconds.
INTERVAL = 0.1


class Cancelled(Exception):
    """Raised inside a step when the run has been stopped.

    Stopping rides on the progress calls a step already makes: the next one raises, the step
    unwinds through its own ``finally`` blocks, and nothing has to poll a flag.
    """


def format_duration(seconds: float) -> str:
    """``mm:ss``, or ``h:mm:ss`` once it runs past an hour."""
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class Reporter:
    """Turns ``(done, total, label)`` from a step into a line of text and a fraction.

    ``emit(text, fraction)`` is called with a fraction in [0, 1], or -1 when the task cannot
    say how far along it is.
    """

    def __init__(self, emit: Callable[[str, float], None]):
        self._emit = emit
        self._label: str | None = None
        self._started = time.monotonic()
        self._last = 0.0
        self._cancelled = False

    def cancel(self):
        """Ask the running step to stop at its next progress call."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def __call__(self, done: int, total: int, label: str):
        if self._cancelled:
            raise Cancelled(label)
        now = time.monotonic()
        if label != self._label:            # a new task: restart the clock, always show it
            self._label, self._started, self._last = label, now, 0.0
        finished = total and done >= total
        if not finished and now - self._last < INTERVAL:
            return
        self._last = now
        elapsed = now - self._started
        if not total:
            self._emit(f"{label}  {format_duration(elapsed)}", -1.0)
            return
        fraction = min(1.0, max(0.0, done / total))
        remaining = elapsed * (1 - fraction) / fraction if fraction > 0 else 0.0
        self._emit(f"{label}  {fraction * 100:3.0f}%  {format_duration(elapsed)}"
                   f"<{format_duration(remaining)}", fraction)

    def note(self, label: str):
        """A task that cannot measure itself; its elapsed time still runs."""
        self(0, 0, label)
