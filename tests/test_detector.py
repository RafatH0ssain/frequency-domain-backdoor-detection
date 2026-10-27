"""Smoke tests for the detector metrics.

These need torch, scipy, scikit-learn, and matplotlib. Without them the module
skips instead of failing, so a plain `pytest` still succeeds on a machine that
only wants to read the code.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from detector import apply_blur, evaluate, to_gray, variance_of_laplacian


def sharp_image():
    """A synthetic 3x64x64 image with a hard checkerboard, so it has real edges."""
    torch = pytest.importorskip("torch")
    tile = torch.zeros(64, 64)
    tile[::2, ::2] = 1.0
    tile[1::2, 1::2] = 1.0
    return tile.unsqueeze(0).repeat(3, 1, 1)


def test_blur_lowers_variance_of_laplacian():
    img = sharp_image()
    blurred = apply_blur(img, 2.5)
    assert variance_of_laplacian(blurred) < variance_of_laplacian(img)


def test_to_gray_shape_and_range():
    gray = to_gray(sharp_image())
    assert gray.shape == (64, 64)
    assert 0.0 <= float(gray.min()) and float(gray.max()) <= 1.0


def test_evaluate_returns_auc_in_unit_interval():
    clean = [10.0, 11.0, 9.5, 10.5, 12.0]
    blur = [2.0, 1.5, 2.5, 3.0, 1.0]
    result = evaluate(clean, blur, flag_below=True)
    assert 0.0 <= result["auc"] <= 1.0
    assert result["auc"] > 0.5, "a clean separation should beat chance"


def test_identical_distributions_score_near_chance():
    same = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = evaluate(same, same, flag_below=True)
    assert abs(result["auc"] - 0.5) < 0.01
