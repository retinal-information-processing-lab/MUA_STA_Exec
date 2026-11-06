"""Prove this package cannot modify a ``.mcd`` recording.

Ported from ``test_safety.py`` in ``/mnt/Extra/mcd_format_test/`` so that it runs in ``pytest``
here, over this package's own source rather than upstream's file list.

The reason it exists: MC_Rack writes ``.mcd`` files read/write and leaves them that way, so
every recording on the rig is writable and file permissions guarantee nothing. A ``.mcd`` is
the rawest copy of an irreplaceable recording, so the guarantee has to be a property of the
code — ``mcd.py`` opens recordings ``O_RDONLY`` and refuses any write whose target is one — and
these tests check that the property still holds after a change.
"""
import ast
import os
import tempfile
from pathlib import Path

import pytest

from mua_sta import mcd

PACKAGE = Path(__file__).resolve().parent.parent / "mua_sta"

#: Expressions that denote a source recording. A write-mode open on any of these is the exact
#: mistake these tests exist to catch.
SOURCE_EXPRS = ("self.path", "self._fh", "mcd_path", "m.path", "info.path", "recording_path")

WRITE_FLAGS = ("O_WRONLY", "O_RDWR", "O_CREAT", "O_APPEND", "O_TRUNC")


def _mode_of(call, source):
    """The mode string of an ``open()`` / ``os.open()`` call, or None if it is neither."""
    if isinstance(call.func, ast.Name) and call.func.id == "open":
        if len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
            return str(call.args[1].value)
        for keyword in call.keywords:
            if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                return str(keyword.value.value)
        return "r"                                   # open() defaults to reading
    if (isinstance(call.func, ast.Attribute) and call.func.attr == "open"
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "os"):
        flags = ast.get_source_segment(source, call.args[1]) if len(call.args) > 1 else ""
        return "WRITE" if any(f in (flags or "") for f in WRITE_FLAGS) else "r"
    return None


def _opens():
    """Every ``open``/``os.open`` in the package, as ``(where, target, mode, is_write)``."""
    found = []
    for path in sorted(PACKAGE.glob("*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            mode = _mode_of(node, source)
            if mode is None:
                continue
            target = ast.get_source_segment(source, node.args[0]) if node.args else "?"
            is_write = mode == "WRITE" or any(c in mode for c in "wax+")
            found.append((f"{path.name}:{node.lineno}", target or "?", mode, is_write))
    return found


def test_the_audit_has_something_to_look_at():
    """A glob that matched nothing would make the two tests below pass for the wrong reason."""
    assert (PACKAGE / "mcd.py").is_file()
    assert any(target == "out_path" for _, target, _, is_write in _opens() if is_write)


def test_no_write_mode_open_on_a_source_path():
    offenders = [f"{where} -> open({target}, {mode!r})"
                 for where, target, mode, is_write in _opens()
                 if is_write and any(e in target for e in SOURCE_EXPRS)]
    assert not offenders, offenders


def test_source_paths_are_opened_only_through_open_readonly():
    """Even a read-mode ``open()`` on a recording is refused, so the ``O_RDONLY`` guarantee
    lives in one function instead of being repeated and eventually forgotten."""
    offenders = [f"{where} -> open({target}, {mode!r})"
                 for where, target, mode, is_write in _opens()
                 if not is_write and any(e in target for e in SOURCE_EXPRS)]
    assert not offenders, offenders


def test_the_reader_opens_a_recording_read_only(tmp_path):
    import fcntl
    import mcdbuilder
    path = mcdbuilder.write_mcd(tmp_path / "rec.mcd", mcdbuilder.ramp_frames(10, 4),
                                block_samples=5)
    with mcd.McdFile(path) as m:
        fd = m._fh.fileno()
        assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        with pytest.raises(OSError):
            os.write(fd, b"X")


def test_a_recording_is_detected_by_content(tmp_path):
    disguised = tmp_path / "recording.bak"
    disguised.write_bytes(b"MCSSTRM " + b"\xff" * 8 + b"\0" * 64)
    decoy = tmp_path / "notarecording.bak"
    decoy.write_bytes(b"NOTMCS__" + b"\0" * 64)
    assert mcd.is_mcs_recording(disguised)
    assert not mcd.is_mcs_recording(decoy)
    assert not mcd.is_mcs_recording(tmp_path / "does-not-exist")


def _refuses(target, source=None):
    try:
        mcd._refuse_if_source(target, source)
        return False
    except ValueError:
        return True


def test_the_write_guard_refuses_every_way_of_naming_a_recording(tmp_path):
    recording = tmp_path / "recording.bak"          # a recording not named .mcd
    recording.write_bytes(b"MCSSTRM " + b"\xff" * 8 + b"\0" * 64)
    decoy = tmp_path / "notarecording.bak"
    decoy.write_bytes(b"NOTMCS__" + b"\0" * 64)
    link, hard = tmp_path / "link.raw", tmp_path / "hard.raw"
    os.symlink(recording, link)
    os.link(recording, hard)

    assert _refuses(tmp_path / "recording.mcd")                 # by name, even if absent
    assert _refuses(recording)                                  # by content
    assert _refuses(recording, recording)                       # the source itself
    assert _refuses(link, recording)                            # a symlink to it
    assert _refuses(hard, recording)                            # a hard link to it
    assert not _refuses(tmp_path / "something.raw")             # an ordinary output
    assert not _refuses(decoy)                                  # odd name, not a recording


def test_a_full_read_leaves_the_recording_untouched(tmp_path):
    """Every path that touches the file, then the file compared byte for byte."""
    import hashlib
    import mcdbuilder
    path = mcdbuilder.write_mcd(tmp_path / "rec.mcd", mcdbuilder.ramp_frames(60, 8),
                                block_samples=20)
    before = (path.stat().st_size, path.stat().st_mtime_ns,
              hashlib.sha256(path.read_bytes()).hexdigest())

    from mua_sta import recording as rec
    info = rec.open_info(path, n_channels=8)
    with mcd.McdFile(path) as m:
        m.read(0, 10)
        list(m.blocks())
        m.channel_levels(0)
        m.describe()
        with pytest.raises(ValueError):
            m.to_raw(path)                           # refuses to target the source
    rec.read_channel(info, 3)
    rec.as_spikeinterface(info).get_traces(start_frame=0, end_frame=10)

    after = (path.stat().st_size, path.stat().st_mtime_ns,
             hashlib.sha256(path.read_bytes()).hexdigest())
    assert before == after
