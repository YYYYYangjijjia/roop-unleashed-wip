"""CPU/model-free contracts for AlphaFace identity, geometry and execution APIs."""
import os
import ast
import copy
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from skimage import transform as trans

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from insightface.app.common import Face
from roop.FaceSet import FaceSet
from roop.face_util import (_ALPHAFACE_YAW_TEMPLATES, _alphaface_alignment,
                            align_crop, estimate_norm, swap_template_points)
from roop.processors.FaceSwapInsightFace import (FaceSwapInsightFace,
    SWAP_MODELS, prepare_swap_session, swap_batch_capacity)
from roop.procmgr_tiling import PixelBoostMixin
from roop.procmgr_masking import MaskingMixin
from roop.identity_detail import _template_warp, CANONICAL_SIZE


class AlphaFaceIdentityTests(unittest.TestCase):
    def test_all_65_raw_references_projected_once_not_hyperswap_unit_mean(self):
        rng = np.random.default_rng(31)
        vectors = rng.normal(size=(65, 512)).astype(np.float32)
        vectors *= np.linspace(1, 4, 65, dtype=np.float32)[:, None]
        fs = FaceSet()
        fs.faces = [Face(embedding=v.copy()) for v in vectors]
        original = vectors.copy()
        processor = FaceSwapInsightFace()
        processor.embedding_mode = 'raw_emap'
        processor.emap = np.eye(512, dtype=np.float32)
        mean = fs.average_identity_face()
        latent = processor._compute_latent(mean)
        expected = vectors.mean(axis=0)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(latent[0], expected, atol=1e-7)
        unit_mean = mean.mean_normed_embedding
        unit_mean /= np.linalg.norm(unit_mean)
        self.assertGreater(np.max(np.abs(latent[0] - unit_mean)), .01)
        np.testing.assert_array_equal(np.stack([f.embedding for f in fs.faces]), original)

    def test_projection_orientation_and_invalid_identity(self):
        processor = FaceSwapInsightFace()
        processor.embedding_mode = 'raw_emap'
        processor.emap = np.roll(np.eye(512, dtype=np.float32), 1, axis=1)
        source = np.zeros(512, np.float32)
        source[0] = 3
        result = processor._compute_latent(Face(embedding=source))
        self.assertEqual(np.argmax(result), 1)
        for value in [np.zeros(512), np.full(512, np.nan), np.ones(511)]:
            with self.assertRaises(ValueError):
                processor._compute_latent(Face(embedding=value))


class AlphaFaceExecutionTests(unittest.TestCase):
    def test_fixed_batch_no_trt_no_graph_rewrite(self):
        spec = SWAP_MODELS['alphaface']
        self.assertEqual(swap_batch_capacity(spec, 1024), 1)
        with patch('roop.processors.FaceSwapInsightFace._BATCH_SWAP', True), \
             patch('roop.processors.FaceSwapInsightFace.onnx.load') as load:
            model, providers = prepare_swap_session(spec, 'unchanged.onnx',
                [('TensorrtExecutionProvider', {}), ('CUDAExecutionProvider', {'device_id': 2}), 'CPUExecutionProvider'])
        self.assertEqual(model, 'unchanged.onnx')
        self.assertEqual(providers[0][0], 'CUDAExecutionProvider')
        self.assertEqual(providers[0][1]['device_id'], 2)
        load.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'CUDA or CPU'):
            prepare_swap_session(spec, 'unused', ['TensorrtExecutionProvider'])

    def make_processor(self):
        processor = FaceSwapInsightFace()
        processor.loaded_model_key = 'alphaface'
        processor.embedding_mode = 'raw_emap'
        processor.emap = np.eye(512, dtype=np.float32)
        processor.embed_input_name = 'source_embedding'
        processor._batch_unsupported = True
        self.feeds = []
        def infer(feed):
            self.feeds.append(feed)
            self.assertEqual(set(feed), {'target', 'source_embedding'})
            self.assertEqual(feed['source_embedding'].shape, (1, 512))
            self.assertEqual(feed['target'].shape, (1, 3, 256, 256))
            return [feed['target'].copy()]
        processor._infer = infer
        return processor

    def test_run_and_batch_apis_keep_identity_order_and_never_batch_graph(self):
        processor = self.make_processor()
        a = Face(embedding=np.eye(512, dtype=np.float32)[0])
        b = Face(embedding=np.eye(512, dtype=np.float32)[1])
        blob1 = np.full((1, 3, 256, 256), .2, dtype=np.float32)
        blob2 = np.full_like(blob1, .8)
        self.assertEqual(processor.Run(a, None, blob1).shape, (3, 256, 256))
        result = processor.RunBatch(a, None, [blob1, blob2])
        np.testing.assert_array_equal(result[0], blob1[0])
        result = processor.RunBatchMulti([(b, None, blob2), (a, None, blob1)])
        np.testing.assert_array_equal(result[0], blob2[0])
        self.assertEqual([int(np.argmax(f['source_embedding'])) for f in self.feeds], [0, 0, 0, 1, 0])
        self.assertIsNone(processor.take_masks())

    def test_invalid_model_output_raises_instead_of_returning_original(self):
        processor = self.make_processor()
        source = Face(embedding=np.ones(512, np.float32))
        blob = np.ones((1, 3, 256, 256), np.float32)
        for out in [np.zeros_like(blob), np.full_like(blob, np.nan), np.ones((1, 3, 128, 128))]:
            processor._infer = lambda _, value=out: [value]
            with self.assertRaisesRegex(RuntimeError, 'invalid face crop'):
                processor.Run(source, None, blob)


class AlphaFaceGeometryTests(unittest.TestCase):
    def test_actual_processmgr_alignment_block_publishes_one_shared_matrix(self):
        tree = ast.parse((APP / 'roop/ProcessMgr.py').read_text(encoding='utf-8-sig'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ProcessMgr')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'process_face')
        start = next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id == 'swap_p' for t in n.targets))
        end = next(i for i,n in enumerate(method.body[start:], start) if isinstance(n,ast.Assign)
                   and any(ast.unparse(t) == 'self._tls.cur_M' for t in n.targets))
        fn = ast.parse('def align(self, plate, target_face):\n    pass').body[0]
        fn.body = copy.deepcopy(method.body[start:end+1])
        fn.body += ast.parse('return aligned_img, M').body
        ns = {'align_crop': align_crop}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'ProcessMgr-alignment', 'exec'), ns)
        plate = np.full((280,280,3), 50, np.uint8)
        kps = (_ALPHAFACE_YAW_TEMPLATES[4]*2 + [24,18]).astype(np.float32)
        target = Face(kps=kps)
        owner = SimpleNamespace(processors=[SimpleNamespace(type='swap', model_output_size=256, model_template='alphaface')],
            options=SimpleNamespace(subsample_size=128), _tls=SimpleNamespace(frame_idx=None))
        aligned, matrix = ns['align'](owner, plate, target)
        expected, expected_matrix = align_crop(plate, kps, 256, 'alphaface')
        np.testing.assert_array_equal(aligned, expected)
        np.testing.assert_array_equal(matrix, expected_matrix)
        self.assertIs(target.matrix, matrix)
        self.assertIs(owner._tls.cur_M, matrix)

    def test_processmgr_canonical_manual_mask_moves_to_actual_yaw_crop(self):
        tree = ast.parse((APP / 'roop/ProcessMgr.py').read_text(encoding='utf-8-sig'))
        fn = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name == '_resize_to_ss')
        kps = (_ALPHAFACE_YAW_TEMPLATES[0]*2 + [24,18]).astype(np.float32)
        matrix = estimate_norm(kps, 256, 'alphaface')
        canonical = estimate_norm(kps,256)
        def compose(a,b):
            return (np.vstack([a,[0,0,1]]) @ np.vstack([b,[0,0,1]]))[:2]
        ns = {'cv2':cv2,'np':np,'subsample_size':256,'swap_template':'alphaface',
              'estimate_norm':estimate_norm,'M':matrix,'target_face':Face(kps=kps),
              '_compose_affine':compose,'_invert_affine':cv2.invertAffineTransform}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(fn)],type_ignores=[])), 'ProcessMgr-mask','exec'),ns)
        mask = np.zeros((256,256),np.float32)
        cv2.circle(mask,(140,110),12,1.,-1)
        expected = cv2.warpAffine((mask*255).astype(np.uint8),compose(matrix,cv2.invertAffineTransform(canonical)),
                                 (256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)/255.
        np.testing.assert_allclose(ns['_resize_to_ss'](mask),expected,atol=1e-7)

    def test_five_target_yaw_templates_fit_and_scale_consistently(self):
        for index, template in enumerate(_ALPHAFACE_YAW_TEMPLATES):
            expected = template * 2
            expected[:, 0] += 16
            origin = trans.SimilarityTransform(scale=1.3, rotation=.21, translation=(41, 23))
            landmarks = origin.inverse(expected).astype(np.float32)
            matrix = estimate_norm(landmarks, 256, 'alphaface')
            got = cv2.transform(landmarks[None], matrix)[0]
            np.testing.assert_allclose(got, expected, atol=2e-4)
            np.testing.assert_allclose(estimate_norm(landmarks, 512, 'alphaface'), matrix * 2, atol=1e-6)
            np.testing.assert_allclose(swap_template_points(256, 'alphaface', landmarks), expected, atol=1e-6)
        with self.assertRaises(ValueError):
            swap_template_points(256, 'alphaface')

    def test_alpha_border_and_crop_match_verified_trial_contract(self):
        sys.path.insert(0, str(APP / 'tools'))
        from trial_alphaface import align_face
        crop = np.full((180, 180, 3), 140, np.uint8)
        landmarks = _ALPHAFACE_YAW_TEMPLATES[2] * 1.5
        trial_crop, trial_matrix, _, _ = align_face(crop, landmarks)
        actual, matrix = align_crop(crop, landmarks, 256, 'alphaface')
        np.testing.assert_allclose(matrix, trial_matrix, atol=1e-12)
        np.testing.assert_array_equal(actual, trial_crop)

    def test_rgb_and_pixelboost_roundtrip_retains_spatial_correspondence(self):
        mixin = PixelBoostMixin()
        p = SimpleNamespace(model_mean=[0, 0, 0], model_standard_deviation=[1, 1, 1], model_denormalize=False)
        crop = np.random.default_rng(3).integers(0, 256, (512, 512, 3), dtype=np.uint8)
        tiles = mixin.implode_pixel_boost(crop, 256, 2)
        outputs = []
        for tile in tiles:
            blob = mixin.prepare_crop_frame(tile, p)
            np.testing.assert_allclose(blob[0, 0], tile[:, :, 2] / 255, atol=1e-7)
            outputs.append(mixin.normalize_swap_frame(blob[0], p))
        np.testing.assert_array_equal(mixin.explode_pixel_boost(outputs, 256, 2, 512), crop)

    def test_actual_paste_back_recovers_target_landmark_location(self):
        landmarks = (_ALPHAFACE_YAW_TEMPLATES[1] * 2 + [35, 20]).astype(np.float32)
        plate = np.zeros((300, 300, 3), np.uint8)
        centre = tuple(np.rint(landmarks[2]).astype(int))
        cv2.circle(plate, centre, 4, (0, 0, 250), -1)
        crop, matrix = align_crop(plate, landmarks, 256, 'alphaface')
        owner = SimpleNamespace(options=SimpleNamespace(show_face_area_overlay=False),
            _scale_paste=lambda inverse, shape, lm: (inverse, lm),
            _model_mask_matte=lambda *args: None, blur_area=lambda matte, blend: matte)
        result = MaskingMixin.paste_upscale(owner, crop, crop, matrix, np.zeros_like(plate), 1., [0,0,0,0,0])
        ys, xs = np.where(result[:, :, 2] > 150)
        self.assertGreater(len(xs), 0)
        self.assertLess(abs(xs.mean() - centre[0]), 1)
        self.assertLess(abs(ys.mean() - centre[1]), 1)

    def test_identity_detail_uses_selected_target_template_direction(self):
        landmarks = _ALPHAFACE_YAW_TEMPLATES[0] * 2
        src = swap_template_points(CANONICAL_SIZE, 'arcface').astype(np.float32)
        dst = swap_template_points(256, 'alphaface', landmarks).astype(np.float32)
        forward, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        channel = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), np.float32)
        cv2.circle(channel, tuple(np.rint(src[2]).astype(int)), 5, 1., -1)
        expected = cv2.warpAffine(channel, forward, (256,256), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        np.testing.assert_allclose(_template_warp(channel, (256,256), 'alphaface', landmarks), expected)


if __name__ == '__main__':
    unittest.main()
