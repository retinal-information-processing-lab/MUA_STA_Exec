"""Trigger detection: where an onset is, and whether the triggers are regular."""
import numpy as np

from mua_sta import triggers as tg
from mua_sta.rigs import ADC_ZERO


def pulse_train(n_pulses, period, rise=4, amplitude=19000, fs=20000.0):
    """A TTL train on a raw ADC channel, with a rising edge a few samples long."""
    levels = np.full(n_pulses * period + period, ADC_ZERO, dtype=np.uint16)
    onsets = []
    for i in range(n_pulses):
        start = i * period + 10
        onsets.append(start)
        levels[start:start + rise] = ADC_ZERO + (
            amplitude * np.arange(1, rise + 1) // rise).astype(np.uint16)
        levels[start + rise:start + period // 2] = ADC_ZERO + amplitude
    return levels, np.array(onsets), fs


def test_onset_is_the_foot_of_the_rising_edge():
    """Detection walks back from the crossing, so the onset does not move with the threshold.

    That is what lets two rigs with different thresholds give comparable frame times. The
    onset is the last sample still at baseline, one before the edge starts to rise — the same
    convention as the standard pipeline, so onsets from both agree.
    """
    levels, starts, _ = pulse_train(5, period=200, rise=6)
    for threshold in (5000, 10000, 18000):
        found = tg.detect_onsets(levels, threshold)
        np.testing.assert_array_equal(found, starts - 1)


def test_no_onsets_below_the_pulse():
    levels, _, _ = pulse_train(5, period=200)
    assert tg.detect_onsets(levels, 25000).size == 0


def test_a_pulse_at_sample_zero_does_not_wrap():
    """Walking back must stop at the start of the recording, not index from the end."""
    levels = np.full(50, ADC_ZERO, dtype=np.uint16)
    levels[0:20] = ADC_ZERO + 19000
    assert tg.detect_onsets(levels, 5000).size == 0      # already high: no rising edge


def test_sanity_check_reports_regular_triggers():
    levels, expected, fs = pulse_train(40, period=200)
    result = tg.sanity_check(tg.detect_onsets(levels, 10000), fs)
    assert result.count == expected.size
    assert result.modal_interval == 200
    assert result.error_onsets.size == 0
    assert result.frequency == fs / 200
    assert "regular" in result.describe()


def test_sanity_check_flags_a_missing_trigger():
    """A dropped trigger shifts every later frame, which quietly ruins the STA."""
    levels, _, fs = pulse_train(40, period=200)
    levels[20 * 200:21 * 200] = ADC_ZERO                 # drop pulse 20
    result = tg.sanity_check(tg.detect_onsets(levels, 10000), fs)
    assert result.error_onsets.size == 1
    assert "irregular" in result.describe()


def test_sequence_count():
    levels, _, fs = pulse_train(40, period=200)
    result = tg.sanity_check(tg.detect_onsets(levels, 10000), fs)
    assert result.n_sequences(frames_per_sequence=12) == 3


def test_envelope_keeps_narrow_pulses():
    """Plain decimation would hide most triggers; the plot needs a min/max envelope."""
    levels, _, fs = pulse_train(200, period=200, rise=2)
    t, low, high = tg.envelope(levels, fs, n_points=50)
    assert t.size == low.size == high.size == 50
    assert high.max() > 18000 and (high > 18000).sum() > 40
    assert low.min() == 0


def test_envelope_of_a_short_trace_is_the_trace():
    levels = np.full(10, ADC_ZERO, dtype=np.uint16)
    t, low, high = tg.envelope(levels, 20000.0, n_points=50)
    assert t.size == 10
    np.testing.assert_array_equal(low, high)


def test_threshold_in_microvolts():
    assert tg.threshold_to_uv(17714) == 17714 * tg.UV_PER_LEVEL
