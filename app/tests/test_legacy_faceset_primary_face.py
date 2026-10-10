"""A legacy source photo must contribute one identity even with bystanders."""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from insightface.app.common import Face

import source_gallery


class LegacyFacesetPrimaryFaceTests(unittest.TestCase):
    def test_background_detection_does_not_enter_source_identity(self):
        frame = np.zeros((512, 512, 3), np.uint8)
        background = Face(bbox=np.array([350, 140, 370, 165]),
                          kps=np.zeros((5, 2), np.float32))
        main = Face(bbox=np.array([120, 100, 390, 440]),
                    kps=np.zeros((5, 2), np.float32))
        captured = []
        with tempfile.TemporaryDirectory() as folder:
            open(os.path.join(folder, '0.png'), 'wb').close()
            with mock.patch.object(source_gallery.util, 'unzip'), \
                 mock.patch.object(source_gallery, 'get_image_frame', return_value=frame), \
                 mock.patch.object(source_gallery, 'extract_face_images',
                                   return_value=[[background, frame], [main, frame]]), \
                 mock.patch.object(source_gallery, '_mask_offsets_from_cfg', return_value=[]), \
                 mock.patch.object(source_gallery, '_sources_append', side_effect=lambda fs, thumb: captured.append(fs)), \
                 mock.patch.object(source_gallery.util, 'convert_to_gradio', side_effect=lambda image: image):
                source_gallery._ingest_faceset_in_dir('sample.fsz', folder)
        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0].faces), 1)
        self.assertIs(captured[0].faces[0], main)


if __name__ == '__main__':
    unittest.main()
