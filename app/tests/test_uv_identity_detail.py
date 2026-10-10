import cv2
import numpy as np
import pytest

from roop.uv_identity_detail import _compact_dark_marks, _coverage_gate
from roop import uv_identity_detail


def test_skin_blob_is_retained_but_shadow_edge_and_stroke_are_rejected():
    image = np.full((96, 96), 180, np.uint8)
    cv2.circle(image, (24, 24), 3, 105, -1)  # freckle or small mole
    cv2.line(image, (50, 12), (50, 78), 85, 2)  # hair or wrinkle
    image[:, 80:] = 90  # broad illumination/shadow edge
    skin = np.ones(image.shape, np.uint8)

    marks = _compact_dark_marks(image.astype(np.float32), skin)

    assert marks[24, 24] > 5
    assert marks[45, 50] < 1
    assert marks[45, 80] < 1


def test_coverage_admits_conflicting_single_reference_marks_gradually():
    support = np.asarray([.05, .15, .2, .8], np.float32)
    conservative = _coverage_gate(support, 0)
    broad = _coverage_gate(support, 1)

    assert conservative[1] == 0
    assert conservative[-1] == 1
    assert broad[1] > conservative[1]
    assert broad[2] > conservative[2]
    assert np.all(broad >= conservative)


def test_offline_missing_uv_model_fails_without_download(tmp_path, monkeypatch):
    monkeypatch.setattr(uv_identity_detail, 'models_directory', lambda: tmp_path)
    monkeypatch.setattr(uv_identity_detail, 'runtime_config', lambda: {'offline': True})
    monkeypatch.setattr(uv_identity_detail.urllib.request, 'urlopen',
                        lambda *args, **kwargs: pytest.fail('network was used'))

    with pytest.raises(FileNotFoundError, match='offline mode prevents download'):
        uv_identity_detail._assets()
