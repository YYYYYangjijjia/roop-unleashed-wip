"""Contact previews borrow video identity tracks without reusing stale people."""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import core  # noqa: E402


class PreviewContactTracks(unittest.TestCase):
    def setUp(self):
        core._preview_track_cache.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.video = os.path.join(self.temp.name, 'target.mp4')
        with open(self.video, 'wb') as out:
            out.write(b'video')
        self.calls = []
        self.mgr = SimpleNamespace(
            target_face_datas=[object(), object()],
            target_face_groups=[0, 1],
            options=SimpleNamespace(selected_index=0,
                                    face_distance_threshold=0.75,
                                    frame_rules=[]),
            _track_assignments={}, _track_mode=False,
        )

        def scan(path, start, end, count, **kwargs):
            self.calls.append((path, start, end, count, kwargs))
            # Frame zero is ambiguous; a later clean observation established
            # this track's source. The pre-pass propagates that source to zero.
            self.mgr._track_assignments = {0: [((20.0, 20.0), 0, None)]}

        self.mgr._precompute_tracks = scan
        self.total = mock.patch.object(core, 'get_video_frame_total', return_value=200)
        self.total.start()
        self.addCleanup(self.total.stop)

    def test_first_frame_uses_later_clean_identity_and_caches_it(self):
        core._preview_contact_tracks(
            self.mgr, [{'_emb_contam': 0.6}], 0, self.video)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1:4], (0, 120, 120))
        self.assertTrue(self.calls[0][4]['preview'])
        self.assertTrue(self.mgr._track_mode)
        self.assertIn(0, self.mgr._track_assignments)

        self.mgr._track_assignments = {}
        core._preview_contact_tracks(self.mgr, [], 0, self.video)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.mgr._track_mode)

    def test_clear_face_keeps_the_fast_single_frame_path(self):
        core._preview_contact_tracks(
            self.mgr, [{'_emb_contam': 0.0}], 0, self.video)
        self.assertFalse(self.calls)
        self.assertFalse(self.mgr._track_mode)

    def test_new_target_person_cannot_inherit_an_old_track(self):
        core._preview_contact_tracks(
            self.mgr, [{'_emb_contam': 0.6}], 0, self.video)
        self.mgr.target_face_datas = [object(), object()]
        core._preview_contact_tracks(
            self.mgr, [{'_emb_contam': 0.6}], 0, self.video)
        self.assertEqual(len(self.calls), 2)

    def test_unassigned_track_does_not_override_safe_skip(self):
        def unassigned(*args, **kwargs):
            self.mgr._track_assignments = {0: [((20.0, 20.0), None, None)]}

        self.mgr._precompute_tracks = unassigned
        core._preview_contact_tracks(
            self.mgr, [{'_emb_contam': 0.6}], 0, self.video)
        self.assertFalse(self.mgr._track_mode)

    def test_preview_scan_runs_while_video_render_is_idle(self):
        manager = core.ProcessMgr(None)
        manager.options = self.mgr.options
        manager.target_face_datas = []
        manager.target_face_groups = []
        manager.input_face_datas = []
        reads = []

        def read(_path, frame_number):
            reads.append(frame_number)
            return np.zeros((32, 32, 3), dtype=np.uint8)

        with (mock.patch.object(core.roop.globals, 'processing', False),
              mock.patch('roop.capturer.get_video_frame', side_effect=read),
              mock.patch('roop.face_util.get_all_faces', return_value=[])):
            manager._precompute_tracks(self.video, 0, 2, 2, preview=True)

        self.assertEqual(reads, [1, 2])
        self.assertEqual(manager._track_scanned, 2)
        self.assertFalse(any(manager._track_assignments.values()))

    def test_later_clean_faces_bind_both_initial_contact_tracks(self):
        class Face(dict):
            def __getattr__(self, key):
                return self[key]

        manager = core.ProcessMgr(None)
        manager.options = self.mgr.options
        manager.target_face_datas = [
            SimpleNamespace(embedding=np.array([1.0, 0.0], np.float32)),
            SimpleNamespace(embedding=np.array([0.0, 1.0], np.float32)),
        ]
        manager.target_face_groups = [0, 1]
        manager.input_face_datas = [object(), object()]

        def read(_path, frame_number):
            return np.full((300, 400, 3), frame_number, np.uint8)

        def detect(frame):
            frame_index = int(frame[0, 0, 0]) - 1
            touching = frame_index < 4
            right_x = 175 if touching else 205
            return [
                Face(bbox=np.array([100, 80, 180, 180], np.float32),
                     embedding=np.array([0.0, 1.0] if touching else [1.0, 0.0], np.float32),
                     _emb_contam=0.6 if touching else 0.0),
                Face(bbox=np.array([right_x, 80, right_x + 80, 180], np.float32),
                     embedding=np.array([0.0, 1.0], np.float32),
                     _emb_contam=0.6 if touching else 0.0),
            ]

        with (mock.patch.object(core.roop.globals, 'processing', False),
              mock.patch('roop.capturer.get_video_frame', side_effect=read),
              mock.patch('roop.face_util.get_all_faces', side_effect=detect)):
            manager._precompute_tracks(self.video, 0, 12, 12,
                                       step=2, preview=True)

        first = sorted(manager._track_assignments[0], key=lambda entry: entry[0][0])
        self.assertEqual([entry[1] for entry in first], [0, 1])

    def test_dirty_preview_uses_tracks_instead_of_the_neighbours_embedding(self):
        class Face(dict):
            def __getattr__(self, key):
                try:
                    return self[key]
                except KeyError as exc:
                    raise AttributeError(key) from exc

        left = Face(bbox=np.array([100, 80, 180, 180], np.float32),
                    embedding=np.array([0.0, 1.0], np.float32),
                    _emb_contam=0.6, det_score=0.99)
        right = Face(bbox=np.array([205, 80, 285, 180], np.float32),
                     embedding=np.array([0.0, 1.0], np.float32),
                     _emb_contam=0.6, det_score=0.99)
        manager = core.ProcessMgr(None)
        manager.options = SimpleNamespace(
            frame_rules=[], swap_mode='selected', selected_index=0,
            face_distance_threshold=0.75)
        manager._track_mode = True
        manager._track_assignments = {0: [
            ((140.0, 130.0), 0, np.array([1.0, 0.0], np.float32)),
            ((245.0, 130.0), 1, np.array([0.0, 1.0], np.float32)),
        ]}
        manager._stab_active = manager._precomputed_mode = manager._temporal_mode = False
        manager.last_found_bboxes = None
        manager.target_face_groups = [0, 1]
        manager.target_face_datas = [
            SimpleNamespace(embedding=np.array([1.0, 0.0], np.float32)),
            SimpleNamespace(embedding=np.array([0.0, 1.0], np.float32)),
        ]
        manager.input_face_datas = [object(), object()]
        manager.face_masks = {}
        selected = []

        def composite(pending, plate, temp, others):
            selected.extend(pending)
            return temp

        manager._composite_faces = composite
        frame = np.zeros((300, 400, 3), np.uint8)
        with (mock.patch('roop.ProcessMgr.get_all_faces', return_value=[left, right]),
              mock.patch('roop.ProcessMgr._GEOMETRY_FILTER', None)):
            count, _ = manager.swap_faces(frame, frame.copy(), frame_idx=0)

        self.assertEqual(count, 2)
        self.assertEqual([(source, face is left) for source, face in selected],
                         [(0, True), (1, False)])


if __name__ == '__main__':
    unittest.main()
