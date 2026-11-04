import sys
from pathlib import Path

if sys.stdout is None or sys.stderr is None:
    # pythonw, which MUA_STA.bat starts, has no console: every message would be lost, and the
    # first progress bar spikeinterface draws (measuring the noise) raises on the missing
    # stream, so no run gets past it. Send both where MUA_STA.sh sends them — before anything
    # else is imported, so an import error is in the log too.
    log_dir = Path.home() / ".mua_sta"
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(log_dir / "mua_sta.log", "w", encoding="utf-8", buffering=1)

from .gui import run

sys.exit(run())
