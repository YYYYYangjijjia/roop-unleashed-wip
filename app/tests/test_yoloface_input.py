"""Exercise the real detector preprocessing/decoding without loading a model."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop import yoloface


class FakeSession:
    def __init__(self):
        self.blobs = []

    def get_inputs(self):
        return [SimpleNamespace(name='input', shape=[1, 3, 640, 640])]

    def run(self, outputs, inputs):
        blob = inputs['input']
        self.blobs.append(blob)
        # These are model-canvas coordinates, not original-frame coordinates.
        result = np.zeros((1, 20, 2), dtype=np.float32)
        result[0, :5, 0] = [100, 80, 60, 40, 0.9]
        result[0, 5:, 0] = np.array([
            [80, 70, 1], [120, 70, 1], [100, 80, 1],
            [85, 90, 1], [115, 90, 1],
        ]).ravel()
        return [result]


class YoloFaceInputTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        with patch.object(yoloface, 'conditional_download'), patch.object(
                yoloface.onnxruntime, 'InferenceSession', return_value=self.session):
            self.detector = yoloface.YoloFaceDetector(['CPUExecutionProvider'])
        self.frame = np.empty((480, 640, 3), dtype=np.uint8)
        self.frame[:] = [0, 127, 255]

    def test_native_size_preserves_model_shape_and_original_normalization(self):
        boxes, keypoints = self.detector.detect(self.frame, det_size=640)
        blob = self.session.blobs[-1]
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertEqual(blob.dtype, np.float32)
        np.testing.assert_allclose(blob[0, :, 0, 0],
                                   (np.array([0, 127, 255]) - 127.5) / 128.0)
        self.assertTrue(np.all(blob[0, :, 480:, :] == -127.5 / 128.0))
        np.testing.assert_allclose(boxes[0, :4], [70, 60, 130, 100])
        np.testing.assert_allclose(keypoints[0, 0], [80, 70])

    def test_close_up_rescue_keeps_fixed_tensor_and_maps_coordinates(self):
        boxes, keypoints = self.detector.detect(self.frame, det_size=320)
        blob = self.session.blobs[-1]
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        np.testing.assert_allclose(blob[0, :, 239, 319],
                                   (np.array([0, 127, 255]) - 127.5) / 128.0)
        self.assertTrue(np.all(blob[0, :, 240:, :] == -127.5 / 128.0))
        self.assertTrue(np.all(blob[0, :, :, 320:] == -127.5 / 128.0))
        np.testing.assert_allclose(boxes[0, :4], [140, 120, 260, 200])
        np.testing.assert_allclose(keypoints[0], np.array([
            [160, 140], [240, 140], [200, 160], [170, 180], [230, 180],
        ]))

    def test_larger_request_is_bounded_by_model_canvas(self):
        boxes, _ = self.detector.detect(self.frame, det_size=1280)
        self.assertEqual(self.session.blobs[-1].shape, (1, 3, 640, 640))
        np.testing.assert_allclose(boxes[0, :4], [70, 60, 130, 100])

    def test_nonpositive_size_never_reaches_native_session(self):
        for size in (0, -320):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, 'positive'):
                self.detector.detect(self.frame, det_size=size)
        self.assertEqual(self.session.blobs, [])


if __name__ == '__main__':
    unittest.main()
