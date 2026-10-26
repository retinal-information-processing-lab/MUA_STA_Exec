"""Rebuilding both stimuli: the vec, the analysis grid, and the lattice detector."""
import numpy as np
import pytest

from mua_sta import stimulus as st
from mua_sta.binfile import write_bin
from mua_sta.rigs import apply_optical_transform, rig


def write_vec(path, frame_indices):
    """A .vec as the stimulus software writes one: a header row, then one row per frame."""
    rows = np.zeros((len(frame_indices) + 1, 5))
    rows[0, 1] = len(frame_indices) + 1
    rows[1:, 1] = frame_indices
    np.savetxt(path, rows, fmt="%d")
    return path


def rescale(frame, new_size):
    """Bilinear resize, so check edges come out soft as in a library rendered at 1.4x."""
    old = np.arange(frame.shape[0])
    new = np.linspace(0, frame.shape[0] - 1, new_size)
    rows = np.stack([np.interp(new, old, row) for row in frame])
    return np.stack([np.interp(new, old, col) for col in rows.T], axis=1)


def write_swn_bin(path, mea, n_checks=6, shifts=4, pixels_per_check=12, n_frames=24, seed=0,
                  render_scale=1.0):
    """A miniature shifting-white-noise library: a coarse grid offset by whole shift steps.

    ``render_scale`` above 1 renders it larger with interpolation, the way the real MEA3
    library is drawn at 672 px for a 480 px design: checks end up a fractional number of
    pixels wide and their edges are soft.
    """
    rng = np.random.default_rng(seed)
    size = n_checks * pixels_per_check
    step = pixels_per_check // shifts
    out_size = int(round(size * render_scale))
    frames = np.empty((n_frames, out_size, out_size))
    for i in range(n_frames):
        coarse = rng.random((n_checks + 2, n_checks + 2)) > 0.5
        fine = np.repeat(np.repeat(coarse, pixels_per_check, 0), pixels_per_check, 1).astype(float)
        oy, ox = rng.integers(0, shifts, 2) * step
        frame = fine[oy:oy + size, ox:ox + size]
        frames[i] = frame if render_scale == 1.0 else rescale(frame, out_size)
    write_bin(path, frames, mea)
    return path


def novel_then_repeated(n_novel, rng=None):
    """A playlist shaped like a real one: the novel frames shuffled, then a repeated block."""
    rng = rng or np.random.default_rng(0)
    novel = rng.permutation(n_novel).tolist()
    return novel + [n_novel + i % 2 for i in range(n_novel)]


def test_read_vec(tmp_path):
    indices = [7, 3, 9, 0]
    header, frames = st.read_vec(write_vec(tmp_path / "s.vec", indices))
    assert header[1] == len(indices) + 1
    np.testing.assert_array_equal(frames, indices)


def test_read_vec_warns_when_the_header_disagrees(tmp_path):
    path = write_vec(tmp_path / "s.vec", [1, 2, 3])
    rows = np.loadtxt(path)
    rows[0, 1] = 99
    np.savetxt(path, rows, fmt="%d")
    with pytest.warns(UserWarning, match="announces 99 rows"):
        st.read_vec(path)


def test_novel_frame_count_is_half_the_playlist():
    """A sequence is a novel half then a repeated half, so half the library is novel."""
    assert st.novel_frame_count(np.arange(54000)) == 27000


def test_cell_centres_handle_a_fractional_step():
    """672 px over 80 cells is 8.4 px each: a constant stride would drift by three pixels."""
    centres = st.cell_centres(672, 80)
    assert centres.size == 80
    assert centres[0] == 4 and centres[-1] == 668
    np.testing.assert_allclose(np.diff(centres).mean(), 8.4, atol=0.01)
    assert centres.max() < 672 and (np.diff(centres) > 0).all()


@pytest.mark.parametrize("mea", [2, 3])
def test_detect_swn_lattice(tmp_path, mea):
    path = write_swn_bin(tmp_path / "swn.bin", mea, n_checks=6, shifts=4)
    checks, shifts, note = st.detect_swn_lattice(path, mea)
    assert (checks, shifts) == (6, 4)
    assert "24x24 grid" in note


def test_swn_build_is_binary_on_the_right_lattice(tmp_path):
    """A cell centre sits inside one check, so every sample is black or white."""
    bin_path = write_swn_bin(tmp_path / "swn.bin", 3, n_checks=6, shifts=4, n_frames=24)
    vec_path = write_vec(tmp_path / "swn.vec", novel_then_repeated(8))
    spec = st.SwnSpec(bin_path, vec_path, 6, 4, 3)
    frames = spec.build(8)
    assert frames.shape == (8, 24, 24)
    assert set(np.unique(frames)) <= {0, 255}


def test_swn_build_follows_the_vec_order(tmp_path):
    """Novel frames are shuffled in the .vec, so position i is not frame i of the library."""
    bin_path = write_swn_bin(tmp_path / "swn.bin", 3, n_frames=24)
    order = [3, 0, 2, 1]                        # the novel half of an 8-row playlist, shuffled
    vec_path = write_vec(tmp_path / "swn.vec", order + [4, 5, 4, 5])
    spec = st.SwnSpec(bin_path, vec_path, 6, 4, 3)
    built = spec.build(len(order))
    from mua_sta.binfile import BinFile
    with BinFile(bin_path, 3) as library:
        for position, index in enumerate(order):
            expected = library.frame(index)
            centres = st.cell_centres(expected.shape[0], 24)
            np.testing.assert_allclose(
                built[position] / 255.0, expected[np.ix_(centres, centres)], atol=1 / 255)


def test_swn_build_warns_on_a_lattice_that_does_not_fit(tmp_path):
    bin_path = write_swn_bin(tmp_path / "swn.bin", 3, n_checks=6, shifts=4, n_frames=24,
                             render_scale=1.4)
    vec_path = write_vec(tmp_path / "swn.vec", novel_then_repeated(8))
    spec = st.SwnSpec(bin_path, vec_path, 6, 7, 3)        # 42 cells over a 4-shift grid
    with pytest.warns(UserWarning, match="neither black nor white"):
        spec.build(8)


def test_swn_build_refuses_more_frames_than_the_vec_plays(tmp_path):
    bin_path = write_swn_bin(tmp_path / "swn.bin", 3, n_frames=24)
    vec_path = write_vec(tmp_path / "swn.vec", novel_then_repeated(4))
    spec = st.SwnSpec(bin_path, vec_path, 6, 4, 3)
    with pytest.raises(ValueError, match="novel frames"):
        spec.build(10)


def test_checkerboard_bit_order(tmp_path):
    """Bits are consumed from bit 0, least significant first, row-major inside a frame."""
    source = tmp_path / "src"
    source.write_bytes(bytes([0b00001101, 0]))        # frame 0: 1,0,1,1 -> [[1,0],[1,1]]
    spec = st.CheckerboardSpec(source, 2, 2, mea=2)
    frames = spec.build(2)
    expected = apply_optical_transform(
        np.array([[1, 0], [1, 1]], dtype=np.uint8) * 255, rig(2).optical_transform)
    np.testing.assert_array_equal(frames[0], expected)
    np.testing.assert_array_equal(frames[1], np.zeros((2, 2), dtype=np.uint8))


def test_checkerboard_inverts_polarity_on_mea3(tmp_path):
    """MEA3 displays inverted, so a set bit was shown black. The previous version of this
    tool left that out, which flipped the sign of every MEA3 checkerboard STA."""
    source = tmp_path / "src"
    source.write_bytes(bytes([0b11111111]))
    assert st.CheckerboardSpec(source, 2, 2, mea=2).build(1).max() == 255
    assert st.CheckerboardSpec(source, 2, 2, mea=3).build(1).max() == 0


def test_checkerboard_refuses_a_source_that_is_too_short(tmp_path):
    source = tmp_path / "src"
    source.write_bytes(bytes([0xFF]))
    with pytest.raises(ValueError, match="holds .* bits"):
        st.CheckerboardSpec(source, 8, 8, mea=2).build(10)


def test_checkerboard_cell_size_comes_from_the_check_width(tmp_path):
    """A 42-check board of 16 px checks covers 672 px, not the whole 760 px DMD, so the check
    width has to be given rather than derived from the DMD size."""
    source = tmp_path / "src"
    source.write_bytes(bytes(1024))
    assert st.CheckerboardSpec(source, 42, 42, 3, 16).cell_size_um == pytest.approx(40.0)
    assert st.CheckerboardSpec(source, 42, 42, 2, 16).cell_size_um == pytest.approx(56.0)


def test_check_width_does_not_invalidate_a_rebuilt_checkerboard(tmp_path):
    """It only scales the reported µm; the frames are identical."""
    source = tmp_path / "src"
    source.write_bytes(bytes(1024))
    a = st.CheckerboardSpec(source, 8, 8, 3, 16).cache_deps(4)
    b = st.CheckerboardSpec(source, 8, 8, 3, 32).cache_deps(4)
    assert a == b


def test_detect_swn_lattice_on_a_library_rendered_at_a_fractional_scale(tmp_path):
    """The real MEA3 library: 10 checks of 67.2 px, 8 shifts of 8.4 px, soft edges."""
    path = write_swn_bin(tmp_path / "swn.bin", 3, n_checks=6, shifts=4, pixels_per_check=12,
                         n_frames=24, render_scale=1.4)
    assert st.detect_swn_lattice(path, 3)[:2] == (6, 4)


def test_swn_stays_binary_at_a_fractional_render_scale(tmp_path):
    """Sampling at cell centres avoids the interpolated pixels at the check edges."""
    bin_path = write_swn_bin(tmp_path / "swn.bin", 3, n_checks=6, shifts=4, n_frames=24,
                             render_scale=1.4)
    vec_path = write_vec(tmp_path / "swn.vec", novel_then_repeated(8))
    frames = st.SwnSpec(bin_path, vec_path, 6, 4, 3).build(8)
    grey = np.count_nonzero((frames > 8) & (frames < 247)) / frames.size
    assert grey < st.NON_BINARY_WARN
