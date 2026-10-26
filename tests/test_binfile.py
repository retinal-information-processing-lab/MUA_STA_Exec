"""The bin reader, and that a frame survives the trip through a rig's optics unchanged."""
import numpy as np
import pytest

from mua_sta import rigs
from mua_sta.binfile import BinFile, read_header, stored_shape, write_bin


def f_pattern(rows: int, cols: int) -> np.ndarray:
    """An F: no symmetry at all, so it tells apart every rotation and every mirror.

    A checkerboard or a blob looks the same under most of them; an F does not, which is why
    the lab's orientation test pattern is one.
    """
    frame = np.zeros((rows, cols), dtype=float)
    frame[1:rows - 1, 1:2] = 1.0                    # upright stroke, on the left
    frame[1:2, 1:cols - 1] = 1.0                    # top stroke, full width
    frame[rows // 2:rows // 2 + 1, 1:cols - 2] = 1.0    # middle stroke, one shorter
    return frame


@pytest.mark.parametrize("mea", sorted(rigs.RIGS))
@pytest.mark.parametrize("shape", [(12, 12), (8, 14)])
def test_round_trip(tmp_path, mea, shape):
    """Writing then reading gives back the same frame, square or not."""
    frames = np.stack([f_pattern(*shape), np.zeros(shape)])
    path = write_bin(tmp_path / "f.bin", frames, mea)
    with BinFile(path, mea) as stimulus:
        assert len(stimulus) == 2
        assert stimulus.shape == shape
        np.testing.assert_allclose(stimulus.frame(0), frames[0], atol=1 / 255)
        np.testing.assert_allclose(stimulus.frame(1), frames[1], atol=1 / 255)


@pytest.mark.parametrize("mea", sorted(rigs.RIGS))
def test_stored_shape_follows_the_rig(tmp_path, mea):
    """A rig that transposes stores a non-square frame the other way round.

    Reshaping with a fixed axis order — what both earlier readers did — is right for one rig
    and silently wrong for the other as soon as the frame is not square.
    """
    rows, cols = 8, 14
    path = write_bin(tmp_path / "f.bin", f_pattern(rows, cols)[None], mea)
    header = read_header(path)
    assert (header["ysize"], header["xsize"]) == (rows, cols)
    expected = (cols, rows) if rigs.rig(mea).optical_transform == "rot90_flipud" else (rows, cols)
    assert stored_shape(cols, rows, rigs.rig(mea).optical_transform) == expected


def test_mea3_inverts_polarity(tmp_path):
    """MEA3 displays inverted, so a white frame is stored black. The reader undoes it."""
    white = np.ones((1, 6, 6))
    path = write_bin(tmp_path / "w.bin", white, mea=3)
    with BinFile(path, 3) as stimulus:
        assert stimulus.stored(0).max() == 0            # stored inverted
        np.testing.assert_allclose(stimulus.frame(0), 1.0, atol=1 / 255)


def test_frame_count_comes_from_the_file_not_the_header(tmp_path):
    """The header's frame count is 2 bytes and wraps; the file size does not."""
    path = write_bin(tmp_path / "many.bin", np.zeros((3, 4, 4)), mea=2)
    raw = bytearray(path.read_bytes())
    raw[4:6] = (999).to_bytes(2, "little")              # lie about the count
    path.write_bytes(raw)
    with pytest.warns(UserWarning, match="wraps past 65535"):
        with BinFile(path, 2) as stimulus:
            assert len(stimulus) == 3


def test_rejects_a_frame_too_big_for_the_rig(tmp_path):
    path = write_bin(tmp_path / "big.bin", np.zeros((1, 1200, 2000)), mea=3)
    with pytest.raises(ValueError, match="too big"):
        BinFile(path, 3)
