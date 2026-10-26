"""Make the package importable from a checkout, and keep Qt off a real display."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Qt must know it is headless before the first QApplication is made.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
