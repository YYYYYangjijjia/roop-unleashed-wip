"""Model-free tests of the real average helper and ProcessMgr identity branch."""
import ast
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop.FaceSet import FaceSet


def _selection_branch():
    """Execute the actual selection block without importing GPU/video modules."""
    source = (APP / 'roop' / 'ProcessMgr.py').read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'ProcessMgr')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'process_face')
    blocks = [node for node in method.body if isinstance(node, ast.If)
              and '_average_source_faces' in ast.unparse(node)]
    if len(blocks) != 1:
        raise AssertionError('Expected one actual ProcessMgr identity-selection block')
    function = ast.parse(
        'def select(self, face_index, target_face, bank_yaw_deg, bank_pitch_deg):\n'
        '    inputface = None\n'
        '    selected_src_idx = 0\n'
    ).body[0]
    function.body.append(copy.deepcopy(blocks[0]))
    function.body.append(ast.Return(value=ast.Tuple(
        elts=[ast.Name(id='inputface', ctx=ast.Load()),
              ast.Name(id='selected_src_idx', ctx=ast.Load())], ctx=ast.Load())))
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    calls = []

    def bank_selector(poses, yaw, pitch, count):
        calls.append((yaw, pitch, count))
        return 1

    namespace = {'_select_source_bank_index': bank_selector}
    exec(compile(module, str(APP / 'roop' / 'ProcessMgr.py'), 'exec'), namespace)
    return namespace['select'], calls


def _faceset():
    fs = FaceSet()
    fs.faces = [SimpleNamespace(embedding=np.array(v, dtype=np.float32),
                               mask_offsets=(0, 0, 0, 0, 20, 10))
                for v in ([8, 1, 2], [1, 6, 3], [2, 1, 9])]
    fs.face_poses = [(0, 0), (45, 0), (-45, 0)]
    return fs


def _real_faceset():
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    from insightface.app.common import Face
    fs = _faceset()
    fs.faces = [Face(embedding=face.embedding.copy(),
                     mask_offsets=face.mask_offsets,
                     kps=np.ones((5, 2), dtype=np.float32)) for face in fs.faces]
    return fs


class AverageIdentityTests(unittest.TestCase):
    def test_hyperswap_mean_is_mean_of_unit_references_not_unit_raw_mean(self):
        fs = _real_faceset()
        raw = np.stack([f.embedding.copy() for f in fs.faces])
        expected = np.mean(raw / np.linalg.norm(raw, axis=1, keepdims=True), axis=0)
        fs.AverageEmbeddings()
        result = fs.average_identity_face()
        np.testing.assert_allclose(result.mean_normed_embedding, expected)
        self.assertLess(np.linalg.norm(result.mean_normed_embedding), 1.0)
        np.testing.assert_array_equal(result.embedding, np.mean(raw, axis=0))
        np.testing.assert_allclose(result.normed_embedding,
                                  raw.mean(axis=0) / np.linalg.norm(raw.mean(axis=0)))
        self.assertIsNone(fs.original_first_face().mean_normed_embedding)

    def test_actual_swapper_latent_keeps_native_mean_magnitude_and_legacy_switch(self):
        # Exercise the real inference adapter without loading any GPU/model.
        path = APP / 'roop/processors/FaceSwapInsightFace.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'FaceSwapInsightFace')
        method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                                    and n.name == '_compute_latent'))
        namespace = {'np': np, 'Face': object}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     str(path), 'exec'), namespace)
        compute = namespace['_compute_latent']
        fs = _real_faceset()
        averaged = fs.average_identity_face()
        processor = SimpleNamespace(embedding_mode='normed', plugin_options={})
        np.testing.assert_array_equal(compute(processor, averaged),
                                      averaged.mean_normed_embedding.reshape(1, -1))
        processor.plugin_options = {'hyperswap_native_average': False}
        np.testing.assert_array_equal(compute(processor, averaged),
                                      averaged.normed_embedding.reshape(1, -1))
        processor.plugin_options = {'hyperswap_native_average': True}
        np.testing.assert_array_equal(compute(processor, fs.faces[0]),
                                      fs.faces[0].normed_embedding.reshape(1, -1))
        processor.embedding_mode = 'normed_emap'
        processor.emap = None
        np.testing.assert_array_equal(compute(processor, averaged),
                                      averaged.normed_embedding.reshape(1, -1))

    def test_real_insightface_original_first_face_restores_backup_only_on_copy(self):
        fs = _real_faceset()
        original_embedding = fs.faces[0].embedding.copy()
        fs.AverageEmbeddings()
        stored_mean = fs.faces[0].embedding.copy()
        restored = fs.original_first_face()
        np.testing.assert_array_equal(restored.embedding, original_embedding)
        np.testing.assert_array_equal(restored.normed_embedding,
                                      original_embedding / np.linalg.norm(original_embedding))
        np.testing.assert_array_equal(fs.faces[0].embedding, stored_mean)
        restored.embedding[:] = -100
        restored.kps[:] = -200
        np.testing.assert_array_equal(fs.embeddings_backup, original_embedding)
        np.testing.assert_array_equal(fs.faces[0].embedding, stored_mean)
        np.testing.assert_array_equal(fs.faces[0].kps, np.ones((5, 2), dtype=np.float32))
        v2 = _real_faceset()
        v2.format_version = 2
        np.testing.assert_array_equal(v2.original_first_face().embedding, v2.faces[0].embedding)
        v2.faces = []
        self.assertIsNone(v2.original_first_face())

    def test_actual_pose_branch_uses_original_first_identity_after_legacy_average(self):
        select, _ = _selection_branch()
        fs = _real_faceset()
        original = fs.faces[0].embedding.copy()
        fs.AverageEmbeddings()
        stored_mean = fs.faces[0].embedding.copy()
        restored = fs.original_first_face()
        averaged = fs.average_identity_face()
        manager = SimpleNamespace(input_face_datas=[fs],
            options=SimpleNamespace(source_identity_mode='pose', use_source_bank=False),
            _average_source_faces=[averaged], _pose_first_faces=[restored],
            _tls=SimpleNamespace())
        target = SimpleNamespace(bbox=np.array([1, 2, 3, 4]))
        chosen, index = select(manager, 0, target, 0, 0)
        self.assertEqual(index, 0)
        self.assertIs(chosen, restored)
        np.testing.assert_array_equal(chosen.embedding, original)
        manager.options.source_identity_mode = 'average'
        chosen, _ = select(manager, 0, target, 0, 0)
        self.assertIs(chosen, averaged)
        np.testing.assert_array_equal(chosen.embedding, stored_mean)
        np.testing.assert_array_equal(fs.faces[0].embedding, stored_mean)
        np.testing.assert_array_equal(fs.embeddings_backup, original)

    def test_real_insightface_face_mean_and_copy_ownership(self):
        for variant in ('legacy', 'v2', 'single'):
            with self.subTest(variant=variant):
                fs = _real_faceset()
                if variant == 'single':
                    fs.faces = fs.faces[:1]
                expected = np.mean([f.embedding for f in fs.faces], axis=0)
                if variant == 'legacy':
                    fs.AverageEmbeddings()
                else:
                    fs.format_version = 2
                stored = [f.embedding.copy() for f in fs.faces]
                result = fs.average_identity_face()
                self.assertIs(type(result), type(fs.faces[0]))
                np.testing.assert_array_equal(result.embedding, expected)
                np.testing.assert_allclose(result.normed_embedding, expected / np.linalg.norm(expected))
                result.embedding[:] = -100
                result.kps[:] = -200
                for face, before in zip(fs.faces, stored):
                    np.testing.assert_array_equal(face.embedding, before)
                    np.testing.assert_array_equal(face.kps, np.ones((5, 2), dtype=np.float32))
                np.testing.assert_array_equal(fs.average_identity_face().embedding, expected)

    def test_real_insightface_faces_runtime_copy_sources_and_targets(self):
        from roop.compat_a.runtime import ACompatRenderer
        fs = _real_faceset()
        fs.a_compat_ready = True
        fs.ref_images = [np.zeros((8, 8, 3), dtype=np.uint8)]
        cfg = SimpleNamespace(a_compatibility_mode=True, source_identity_mode='average',
            provider='cpu', selected_enhancer='None', enhancer_type='none',
            mask_engine='DFL XSeg', mask_engine_2='None',
            a_mask_erosion=1, a_mask_blur=15)
        options = SimpleNamespace(processors={'faceswap': {}, 'mask_xseg': {}},
                                  selected_index=0, source_identity_mode='average')
        source_embedding = fs.faces[0].embedding.copy()
        target = fs.faces[1]
        with patch('roop.compat_a.detector.AFaceDetector'), \
                patch('roop.compat_a.processors.AInSwapper'), \
                patch('roop.compat_a.processors.AXSeg'):
            renderer = ACompatRenderer([fs], [target], options, cfg)
            cloned = renderer.engine.input_face_datas[0]
            self.assertIsNot(cloned, fs)
            self.assertIs(cloned.ref_images, fs.ref_images)
            self.assertIs(type(cloned.faces[0]), type(fs.faces[0]))
            self.assertEqual(tuple(cloned.faces[0].mask_offsets), (0, 0, 0, 0, 1, 15))
            self.assertEqual(tuple(fs.faces[0].mask_offsets), (0, 0, 0, 0, 20, 10))
            cloned.faces[0].embedding[:] = -100
            cloned.faces[0].kps[:] = -200
            renderer.engine.target_face_datas[0].kps[:] = -300
            np.testing.assert_array_equal(fs.faces[0].embedding, source_embedding)
            np.testing.assert_array_equal(fs.faces[0].kps, np.ones((5, 2), dtype=np.float32))
            np.testing.assert_array_equal(target.kps, np.ones((5, 2), dtype=np.float32))
            renderer.release()

    def test_legacy_backup_is_not_averaged_twice(self):
        fs = _faceset()
        expected = np.mean([f.embedding for f in fs.faces], axis=0)
        fs.AverageEmbeddings()
        stored = [f.embedding.copy() for f in fs.faces]
        backup = fs.embeddings_backup.copy()
        for _ in range(3):
            result = fs.average_identity_face()
            np.testing.assert_array_equal(result.embedding, expected)
            result.embedding[:] = -100
        for face, before in zip(fs.faces, stored):
            np.testing.assert_array_equal(face.embedding, before)
        np.testing.assert_array_equal(fs.embeddings_backup, backup)

    def test_v2_uses_raw_mean_and_preserves_individual_vectors(self):
        fs = _faceset()
        fs.format_version = 2
        fs.identity_embedding = np.array([100, 0, 0], dtype=np.float32)
        fs.reference_weights = [1, 0, 0]
        before = [f.embedding.copy() for f in fs.faces]
        mean = fs.average_identity_face()
        np.testing.assert_array_equal(mean.embedding, np.mean(before, axis=0))
        self.assertIsNone(fs.embeddings_backup)
        for face, old in zip(fs.faces, before):
            np.testing.assert_array_equal(face.embedding, old)

    def test_single_and_empty_facesets(self):
        fs = _faceset()
        fs.faces = fs.faces[:1]
        result = fs.average_identity_face()
        self.assertIsNot(result, fs.faces[0])
        np.testing.assert_array_equal(result.embedding, fs.faces[0].embedding)
        result.mask_offsets = (1, 1, 1, 1, 1, 1)
        self.assertNotEqual(result.mask_offsets, fs.faces[0].mask_offsets)
        fs.faces = []
        self.assertIsNone(fs.average_identity_face())

    def test_actual_branch_average_overrides_bank_and_pose_and_can_switch(self):
        select, bank_calls = _selection_branch()
        fs = _faceset()
        originals = [f.embedding.copy() for f in fs.faces]
        averaged = fs.average_identity_face()
        manager = SimpleNamespace(input_face_datas=[fs],
            options=SimpleNamespace(source_identity_mode='average', use_source_bank=True),
            _average_source_faces=[averaged], _pose_first_faces=[fs.original_first_face()],
            _tls=SimpleNamespace())
        target = SimpleNamespace(bbox=np.array([1, 2, 3, 4]))
        chosen, index = select(manager, 0, target, 45, 0)
        self.assertIs(chosen, averaged)
        self.assertEqual(index, 0)
        self.assertEqual(bank_calls, [])

        manager.options.source_identity_mode = 'pose'
        manager.options.use_source_bank = False
        chosen, index = select(manager, 0, target, 45, 0)
        self.assertIs(chosen, fs.faces[1])
        self.assertEqual(index, 1)
        manager.options.use_source_bank = True
        chosen, index = select(manager, 0, target, -45, 0)
        self.assertIs(chosen, fs.faces[1])
        self.assertEqual(len(bank_calls), 1)
        manager.options.source_identity_mode = 'average'
        chosen, index = select(manager, 0, target, -45, 0)
        self.assertIs(chosen, averaged)
        self.assertEqual(len(bank_calls), 1)
        for face, before in zip(fs.faces, originals):
            np.testing.assert_array_equal(face.embedding, before)

    def test_compat_initialize_sets_batch_dispatch_contract(self):
        tree = ast.parse((APP / 'roop' / 'ProcessMgr.py').read_text(encoding='utf-8-sig'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'ProcessMgr')
        initialize = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                          and node.name == 'initialize')
        namespace = {'roop': SimpleNamespace(globals=SimpleNamespace(CFG=SimpleNamespace()))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[initialize], type_ignores=[])),
                     'ProcessMgr.initialize', 'exec'), namespace)
        manager = SimpleNamespace(processors=[object()], total_swaps=7)

        def release_resources():
            manager.processors.clear()

        manager.release_resources = release_resources
        options = SimpleNamespace(a_compatibility_mode=True)
        with patch('roop.compat_a.runtime.ACompatRenderer') as renderer:
            namespace['initialize'](manager, [_faceset()], [], options)
            renderer.assert_called_once()
            self.assertIs(manager._a_compat_renderer, renderer.return_value)
        self.assertFalse(options.frame_processing)
        self.assertEqual(manager.processors, [])
        self.assertIsNone(manager.kps_stabilizer)
        self.assertIsNone(manager.enh_stabilizer)
        self.assertFalse(manager._stab_active)

    def test_compat_preview_reuses_sessions_across_frame_requests(self):
        from roop.compat_a.config import request_config
        tree = ast.parse((APP / 'roop' / 'ProcessMgr.py').read_text(encoding='utf-8-sig'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'ProcessMgr')
        initialize = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                          and node.name == 'initialize')
        cfg = SimpleNamespace(a_compatibility_mode=True, a_mask_blur=15,
                              source_identity_mode='average')
        namespace = {'roop': SimpleNamespace(globals=SimpleNamespace(CFG=cfg))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[initialize], type_ignores=[])),
                     'ProcessMgr.initialize', 'exec'), namespace)
        releases = []
        manager = SimpleNamespace(processors=[], total_swaps=7)
        manager.release_resources = lambda: releases.append(True)
        fs = _faceset()

        def options_for(frame, blur=15):
            return SimpleNamespace(a_compatibility_mode=True, a_mask_blur=blur,
                a_compat_cfg=request_config(cfg, {'frame': frame, 'fake_preview': True,
                                                   'a_mask_blur': blur}))

        with patch('roop.compat_a.runtime.ACompatRenderer') as renderer:
            namespace['initialize'](manager, [fs], [], options_for(1))
            namespace['initialize'](manager, [fs], [], options_for(2))
            self.assertEqual(renderer.call_count, 1)
            self.assertEqual(len(releases), 1)
            self.assertEqual(manager._a_compat_swap_base, 7)
            namespace['initialize'](manager, [fs], [], options_for(3, blur=20))
            self.assertEqual(renderer.call_count, 2)
            self.assertEqual(len(releases), 2)

    def test_compat_refuses_pose_identity_selection(self):
        from roop.compat_a.runtime import validate_a_configuration
        options = SimpleNamespace(a_compatibility_mode=True, source_identity_mode='pose',
                                  processors={'faceswap': {}, 'mask_xseg': {}})
        with self.assertRaisesRegex(ValueError, 'source_identity_mode must be average'):
            validate_a_configuration(options, SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
