"""Model-free checks of the real frame gate and identity-matching code."""
import ast
from contextlib import nullcontext
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from threading import RLock
from typing import Callable, List
import unittest
from unittest.mock import Mock, patch

import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from roop.frame_rules import (normalize_frame_rules, frame_rule_policy,
                             frame_rule_signature, frame_rules_digest)


def _rules(mode='skip', ids=None, start=10, end=20):
    return normalize_frame_rules([{'start': start, 'end': end, 'mode': mode,
                                   'person_ids': ids or []}])


def _production_manager():
    """Load actual methods without starting/importing CUDA model infrastructure."""
    tree = ast.parse((APP / 'roop' / 'ProcessMgr.py').read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ProcessMgr')
    names = {'_frame_policy', '_frame_rule_key', 'process_frame', 'swap_faces', 'process_frames'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=[ast.ClassDef(name='Manager', bases=[], keywords=[],
                                        body=methods, decorator_list=[])], type_ignores=[])
    enum = SimpleNamespace(SKIP_FRAME_IF_DISSIMILAR=1, USE_LAST_SWAPPED=2,
                           USE_ORIGINAL_FRAME=3, SKIP_FRAME=4)
    detector = Mock(return_value=[])
    env = dict(Frame=np.ndarray, List=List, Callable=Callable, np=np, os=os,
               cv2=SimpleNamespace(IMREAD_COLOR=1, imdecode=lambda data, _: data),
               wait_while_paused=lambda: None,
               frame_rule_policy=frame_rule_policy, frame_rule_signature=frame_rule_signature,
               roop=SimpleNamespace(globals=SimpleNamespace(pause=False, processing=True,
                                                            no_face_action=2, vr_mode=False)),
               eNoFaceAction=enum, get_all_faces=detector,
               _prof=lambda *a, **kw: nullcontext(), _gpu_guard=lambda **kw: nullcontext(),
               analysis_pooled=lambda: False, solve_pose_5pt=lambda _: None,
               _GEOMETRY_FILTER=None, _PARTIAL_MISS_RESCUE=False,
               _DEBUG_MATCH=False, _SWAP_LOG=None,
               _audit_hit=lambda *a: None, _audit_swapped_gapfill=lambda *a: None,
               FAIL_SCRFD='no-face', face_contact=SimpleNamespace(unreliable=lambda _: False),
               _ada=SimpleNamespace(active_threshold=lambda x: x,
                   identity_distance=lambda a, b, _: 1.0 - float(np.dot(a.embedding, b.embedding))))
    exec(compile(ast.fix_missing_locations(module), str(APP / 'roop' / 'ProcessMgr.py'), 'exec'), env)
    return env['Manager'], detector


class FrameRuleTests(unittest.TestCase):
    def test_inclusive_and_priority(self):
        rules = _rules('only', [8, 20], 1, 20) + _rules('only', [20], 10, 30)
        self.assertEqual(frame_rule_policy(rules, 1), {8, 20})
        self.assertEqual(frame_rule_policy(rules, 10), {20})
        self.assertEqual(frame_rule_policy(rules, 30), {20})
        self.assertIsNone(frame_rule_policy(rules, 31))
        self.assertEqual(frame_rule_policy(rules + _rules(start=15, end=15), 15), set())
        self.assertEqual(frame_rule_policy(_rules('only', []), 10), set())
        self.assertEqual(frame_rule_policy(_rules('only', [8]) + _rules('only', [20]), 10), set())

    def test_bad_rules_rejected(self):
        for value in ({}, [{'mode': 'skip', 'start': True, 'end': 4}],
                      [{'mode': 'skip', 'start': 4, 'end': 1}],
                      [{'mode': 'only', 'start': 1, 'end': 4, 'person_ids': ['1']}],
                      [{'mode': 'unknown', 'start': 1, 'end': 4}]):
            with self.assertRaises(ValueError):
                normalize_frame_rules(value)

    def test_region_and_resume_fingerprint(self):
        rules = _rules()
        self.assertNotEqual(frame_rule_signature(rules, 9), frame_rule_signature(rules, 21))
        self.assertNotEqual(frame_rules_digest(rules), frame_rules_digest(_rules(end=21)))

    def test_skip_bypasses_models_compat_and_fallback_and_preserves_pixels(self):
        Manager, detector = _production_manager()
        mgr = Manager()
        mgr.options = SimpleNamespace(frame_rules=_rules(), frame_rule_frame_offset=9)
        mgr._a_compat_renderer = Mock()
        mgr._publish_live = Mock()
        mgr.swap_faces = Mock(side_effect=AssertionError('must not swap'))
        frame = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        self.assertIs(mgr.process_frame(frame, frame_idx=0), frame)
        self.assertIs(mgr.process_frame(frame, frame_idx=10), frame)
        detector.assert_not_called()
        mgr._a_compat_renderer.process_frame.assert_not_called()
        mgr.swap_faces.assert_not_called()
        # Offset models trim/resume: local 11 means source frame 21, now normal.
        self.assertIsNone(mgr._frame_policy(11))

    def test_only_keeps_original_person_to_source_rank_even_with_track_mode(self):
        Manager, detector = _production_manager()
        mgr = Manager()
        mgr.options = SimpleNamespace(frame_rules=_rules('only', [20], 1, 3),
            frame_rule_frame_offset=0, swap_mode='all', selected_index=0,
            face_distance_threshold=.3)
        mgr._tls = SimpleNamespace()
        mgr._stab_active = mgr._precomputed_mode = mgr._temporal_mode = False
        mgr._track_mode = True  # Must not trust a globally bound forbidden track.
        mgr.last_found_bboxes = None
        mgr.target_face_groups = [8, 8, 20]
        a = SimpleNamespace(embedding=np.array([1., 0.]), bbox=np.array([1., 1., 4., 4.]))
        b = SimpleNamespace(embedding=np.array([0., 1.]), bbox=np.array([6., 1., 9., 4.]))
        mgr.target_face_datas = [a, a, b]
        mgr.input_face_datas = [object(), object()]
        mgr.face_masks = {}
        detector.side_effect = lambda _: [a, b]
        selected = []
        def composite(pending, plate, temp, others):
            selected.extend(pending)
            return temp
        mgr._composite_faces = composite
        frame = np.zeros((12, 12, 3), np.uint8)
        count, _ = mgr.swap_faces(frame, frame.copy(), frame_idx=0)
        self.assertEqual(count, 1)
        self.assertEqual(selected[0][0], 1)
        self.assertIs(selected[0][1], b)
        self.assertEqual(mgr.options.swap_mode, 'all')  # No cross-thread mutation.
        selected.clear()
        count, _ = mgr.swap_faces(frame, frame.copy(), frame_idx=4)
        self.assertEqual(count, 2)  # Original mode outside the rule is untouched.

    def test_extract_workers_use_absolute_list_index_not_thread_chunk_index(self):
        Manager, _ = _production_manager()
        mgr = Manager()
        mgr.options = SimpleNamespace(frame_rules=_rules(start=10, end=11),
                                      frame_rule_frame_offset=9, frame_processing=False)
        called = []
        mgr.process_frame = lambda frame, frame_idx: called.append(frame_idx) or frame
        mgr._frame_ok = Mock()
        mgr._write_image = Mock()
        files = ['f0', 'f1', 'f2', 'f3']
        with patch.object(np, 'fromfile', side_effect=lambda f, **kw: np.full((2, 2, 3), int(f[1]), np.uint8)):
            mgr.process_frames(files, files, ['f3', 'f0', 'f2', 'f1'], None)
        self.assertEqual(called, [3, 2])  # Source frames 13 and 12; 10/11 bypass.
        self.assertEqual(mgr._write_image.call_count, 4)
        mgr.options.frame_rules = []
        called.clear()
        with patch.object(np, 'fromfile', return_value=np.zeros((2, 2, 3), np.uint8)):
            mgr.process_frames(files, files, ['f3', 'f0', 'f2', 'f1'], None)
        self.assertEqual(called, [3, 0, 2, 1])

    def test_resume_updates_rule_offset_at_same_point_as_reader_start(self):
        tree = ast.parse((APP / 'roop' / 'ProcessMgr.py').read_text(encoding='utf-8'))
        run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'run_batch_inmem')
        resume = next(n for n in ast.walk(run) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == 'skip > 0')
        function = ast.parse('def resume(self, frame_start, frame_count, skip):\n    pass').body[0]
        function.body = [n for n in resume.body if isinstance(n, (ast.Assign, ast.AugAssign))]
        function.body += ast.parse('return frame_start, frame_count').body
        env = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<actual resume>', 'exec'), env)
        mgr = SimpleNamespace(options=SimpleNamespace(frame_rule_frame_offset=25))
        self.assertEqual(env['resume'](mgr, 25, 40, 7), (32, 33))
        self.assertEqual(mgr.options.frame_rule_frame_offset, 32)

    def test_compatibility_hold_is_cleared_on_either_side_of_preserve_interval(self):
        Manager, _ = _production_manager()
        mgr = Manager()
        mgr.options = SimpleNamespace(frame_rules=_rules(start=3, end=4), frame_rule_frame_offset=0)
        mgr._tls = SimpleNamespace()
        mgr._cur_kps_stab = mgr._cur_enh_stab = lambda: None
        mgr._publish_live = Mock()
        mgr._a_compat_swap_base = 0
        old = np.ones((2, 2, 3), np.uint8)
        frame = np.zeros((2, 2, 3), np.uint8)
        engine = SimpleNamespace(last_swapped_frame=old, num_frames_no_face=2)
        compat = SimpleNamespace(_lock=RLock(), engine=engine, total_swaps=0)
        compat.process_frame = Mock(side_effect=lambda f: engine.last_swapped_frame if engine.last_swapped_frame is not None else f)
        mgr._a_compat_renderer = compat
        mgr.process_frame(frame, frame_idx=0)
        engine.last_swapped_frame = old
        self.assertIs(mgr.process_frame(frame, frame_idx=2), frame)
        self.assertIs(mgr.process_frame(frame, frame_idx=4), frame)
        self.assertIsNone(engine.last_swapped_frame)
        self.assertEqual(engine.num_frames_no_face, 0)
        self.assertEqual(compat.process_frame.call_count, 2)


if __name__ == '__main__':
    unittest.main()
