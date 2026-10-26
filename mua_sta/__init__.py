"""MUA_STA: multi-unit spike-triggered averages from a MEA recording, while the retina lives.

Pick a recording and the stimulus that was shown — a classic checkerboard or shifting white
noise — and get a 16x16 grid of multi-unit receptive fields. The slow steps (reading the
trigger channel, detecting spikes, rebuilding the stimulus) are cached on what they depend on,
so a mistyped parameter costs only the work it actually affects.
"""

__version__ = "2.0.0"
