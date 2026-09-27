"""Exercise pre-pass detection calls on synthetic arrays; no model loads."""
from contextlib import nullcontext, redirect_stdout
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roop.procmgr_tracking import TrackingMixin
from roop.frame_rules import normalize_frame_rules


class FrameRuleTrackingTests(unittest.TestCase):
    def manager(self):
        mgr = TrackingMixin()
        mgr.options = SimpleNamespace(frame_rules=normalize_frame_rules([
            {'start': 6, 'end': 7, 'mode': 'skip'}]),
            frame_rule_frame_offset=4, stabilize_face=False)
        mgr._publish_live = Mock()
        mgr.progress_gradio = None
        mgr.target_face_groups = []
        mgr.target_face_datas = []
        mgr._assign_track_sources = Mock(return_value=({}, 0., 0, {}))
        mgr.kps_stabilizer = Mock()
        return mgr

    def test_temporal_scan_never_detects_preserve_frames_and_no_rules_keeps_old_path(self):
        frames = [np.full((8, 8, 3), i, np.uint8) for i in range(10)]
        mgr = self.manager()
        seen = []
        with patch('roop.globals.processing', True), \
             patch('roop.face_util.get_all_faces', side_effect=lambda f: seen.append(int(f[0, 0, 0])) or []), \
             patch('roop.procmgr_tracking._gpu_guard', side_effect=lambda **kw: nullcontext()), \
             patch('roop.procmgr_tracking.analysis_pooled', return_value=False), \
             patch('roop.session_pool.detmask_pooling_enabled', return_value=False), \
             redirect_stdout(io.StringIO()):
            mgr._precompute_tracks('', 4, 10, 6, awebp_frames=frames, step=1, collect_obs=True)
            self.assertEqual(seen, [4, 7, 8, 9])
            self.assertEqual(mgr._track_scanned, 6)
            mgr.options.frame_rules = []
            seen.clear()
            mgr._precompute_tracks('', 4, 10, 6, awebp_frames=frames, step=1, collect_obs=True)
            self.assertEqual(seen, [4, 5, 6, 7, 8, 9])

    def test_stabilization_prepass_also_skips_detector(self):
        mgr = self.manager()
        frames = [np.full((8, 8, 3), i, np.uint8) for i in range(10)]
        seen = []
        with patch('roop.globals.processing', True), \
             patch('roop.procmgr_tracking.get_all_faces', side_effect=lambda f: seen.append(int(f[0, 0, 0])) or []), \
             patch('roop.procmgr_tracking._gpu_guard', side_effect=lambda **kw: nullcontext()), \
             patch('roop.procmgr_tracking.analysis_pooled', return_value=False):
            mgr._precompute_stabilized_kps('', frames, 4, 10, 6)
        self.assertEqual(seen, [4, 7, 8, 9])

    def test_temporal_tail_does_not_reappear_after_preserve_interval(self):
        mgr = self.manager()
        mgr.options.frame_rule_frame_offset = 0
        mgr.options.frame_rules = normalize_frame_rules([{'start': 3, 'end': 4, 'mode': 'skip'}])
        mgr._track_scanned = 9
        face = SimpleNamespace(bbox=np.array([1., 1., 5., 5.]))
        track = {'obs': {0: face, 1: face}, 'emb_mean': np.array([1., 0.])}
        with patch.object(mgr, '_coast_face', return_value=face), \
             patch('roop.procmgr_tracking._TEMPORAL_HOLD_FRAMES', 8), \
             patch('roop.orientation.resolve_track_rolls', return_value=0):
            cached = mgr._build_temporal_faces([track], 3)
        self.assertEqual(sorted(cached), [0, 1])


if __name__ == '__main__':
    unittest.main()
