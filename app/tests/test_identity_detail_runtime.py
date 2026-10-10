"""Source-detail extraction and visible restoration without a model download."""

import unittest

import cv2
import numpy as np

from roop.FaceSet import FaceSet
from roop.face_util import swap_template_points
from roop.identity_detail import (_template_warp, build_detail_representation,
                                  restore_identity_detail)
from roop.mean_identity_preview import render_mean_identity


class SourceDetailRuntimeTests(unittest.TestCase):
    @staticmethod
    def source_crop():
        source = np.full((128, 128, 3), 170, np.uint8)
        for centre in ((42, 64), (50, 72), (84, 68)):
            cv2.circle(source, centre, 3, (55, 55, 55), -1)
        return source

    def test_legacy_faceset_derives_and_caches_detail_from_reference_image(self):
        faceset = FaceSet()
        faceset.faces = [{'kps': swap_template_points(128, 'arcface')}]
        faceset.ref_images = [self.source_crop()]
        detail = faceset.identity_detail_for()
        self.assertEqual(detail['schema'], 'roop.identity_detail.v2')
        self.assertIs(faceset.identity_detail_for(), detail)

    def test_strength_restores_source_marks_and_zero_is_unchanged(self):
        detail = build_detail_representation(self.source_crop(), 1.0)
        target = np.full((128, 128, 3), 160, np.uint8)
        self.assertTrue(np.array_equal(restore_identity_detail(target, detail, strength=0), target))
        half = restore_identity_detail(target, detail, strength=.5)
        full = restore_identity_detail(target, detail, strength=1)
        self.assertLess(int(half[64, 42, 0]), 160)
        self.assertLess(int(full[64, 42, 0]), int(half[64, 42, 0]))
        self.assertTrue(np.array_equal(full[4, 4], target[4, 4]))

    def test_visibility_mask_protects_original_pixels(self):
        detail = build_detail_representation(self.source_crop(), 1.0)
        target = np.full((128, 128, 3), 160, np.uint8)
        visible = np.ones((128, 128), np.float32)
        visible[:, :64] = 0
        out = restore_identity_detail(target, detail, strength=1,
                                      visibility_mask=visible)
        self.assertTrue(np.array_equal(out[:, :64], target[:, :64]))
        self.assertLess(int(out[68, 84, 0]), 160)

    def test_high_resolution_map_keeps_small_source_marks(self):
        source = np.full((256, 256, 3), 170, np.uint8)
        for centre in ((92, 127), (105, 145), (166, 132)):
            cv2.circle(source, centre, 2, (40, 40, 40), -1)
        detail = build_detail_representation(source, 1.0, detail_size=256)
        self.assertEqual(detail['shape'], [256, 256])
        target = np.full_like(source, 160)
        restored = restore_identity_detail(target, detail, strength=1)
        self.assertLess(int(restored[127, 92, 0]), 155)

    def test_fixed_template_uses_forward_alignment(self):
        src = swap_template_points(128, 'arcface').astype(np.float32)
        dst = swap_template_points(256, 'mtcnn_512').astype(np.float32)
        transform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        channel = np.zeros((128, 128), np.float32)
        cv2.circle(channel, tuple(np.rint(src[2]).astype(int)), 5, 1.0, -1)
        expected = cv2.warpAffine(channel, transform, (256, 256))
        np.testing.assert_allclose(_template_warp(channel, (256, 256), 'mtcnn_512'),
                                   expected)

    def test_five_point_landmarks_keep_detail_off_eyes(self):
        from insightface.app.common import Face
        points = swap_template_points(256, 'arcface')
        source = np.full((256, 256, 3), 170, np.uint8)
        eye = tuple(np.rint(points[0]).astype(int))
        cheek = (73, 170)
        for spot in (eye, cheek):
            cv2.circle(source, spot, 3, (35, 35, 35), -1)
        detail = build_detail_representation(source, 1.0, detail_size=256)
        target = np.full_like(source, 160)
        restored = restore_identity_detail(
            target, detail, target_face=Face(kps=points),
            matrix=np.array([[1., 0., 0.], [0., 1., 0.]], np.float32),
            strength=1)
        self.assertTrue(np.array_equal(restored[eye[1], eye[0]], target[eye[1], eye[0]]))
        self.assertLess(int(restored[cheek[1], cheek[0], 0]), 155)

    def test_mean_preview_uses_average_embedding_and_smooth_face_guide(self):
        from insightface.app.common import Face
        faceset = FaceSet()
        kps = swap_template_points(128, 'arcface')
        faceset.faces = [Face(embedding=np.array([4., 0., 0.], np.float32), kps=kps),
                         Face(embedding=np.array([0., 2., 0.], np.float32), kps=kps)]
        faceset.ref_images = [self.source_crop()]
        seen = {}

        class FakeSwapper:
            def get(self, guide, guide_face, mean_face, paste_back=False):
                seen['guide'] = guide.copy()
                seen['embedding'] = mean_face.embedding.copy()
                seen['paste_back'] = paste_back
                return cv2.resize(guide, (128, 128)), None

        reference, render, index = render_mean_identity(
            faceset, 'unused.onnx', swapper_factory=lambda _: FakeSwapper())
        np.testing.assert_allclose(seen['embedding'], [2., 1., 0.])
        self.assertFalse(seen['paste_back'])
        self.assertEqual(index, 0)
        self.assertEqual(reference.shape, (256, 256, 3))
        self.assertEqual(render.shape, (128, 128, 3))
        self.assertGreater(float(np.mean(np.abs(
            reference.astype(np.float32) - seen['guide'].astype(np.float32)))), 0.1)


if __name__ == '__main__':
    unittest.main()
