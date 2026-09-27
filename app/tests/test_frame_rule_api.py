import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_rule_api import PersonTokens, clip_identity, resolve_rules, validate_rule_options


def load_function(filename, name, namespace):
    source = Path(__file__).resolve().parents[1] / filename
    node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


class FrameRuleApiTests(unittest.TestCase):
    def test_person_token_survives_added_angle_and_rank_shift(self):
        registry = PersonTokens()
        a, b, c = object(), object(), object()
        first, _ = registry.sync([0, 9], [a, b])
        now, mapping = registry.sync([9, 9], [b, c])
        self.assertEqual(now, [first[1], first[1]])
        self.assertEqual(mapping[first[1]], 9)

    def test_removed_person_and_regroup_never_reuse_token(self):
        registry = PersonTokens()
        a, b = object(), object()
        old, _ = registry.sync([0], [a])
        new, mapping = registry.sync([0], [b])
        self.assertNotEqual(old, new)
        with self.assertRaises(ValueError):
            resolve_rules([dict(start=1, end=5, mode='only', person_ids=old)], mapping, 10)
        registry.clear()
        latest, _ = registry.sync([0], [b])
        self.assertNotEqual(new, latest)

    def test_rules_translate_token_to_raw_group_not_rank(self):
        rules = resolve_rules([dict(start=2, end=5, mode='only', person_ids=['p2'])], {'p2': 9}, 10)
        self.assertEqual(rules[0]['person_ids'], [9])
        with self.assertRaises(ValueError):
            resolve_rules([dict(start=2, end=11, mode='skip')], {}, 10)

    def test_clip_identity_separates_same_filename_and_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            a = Path(folder) / 'a' / 'video.mp4'
            b = Path(folder) / 'b' / 'video.mp4'
            a.parent.mkdir(); b.parent.mkdir()
            a.write_bytes(b'one'); b.write_bytes(b'one')
            old = clip_identity(a)
            self.assertNotEqual(old, clip_identity(b))
            a.write_bytes(b'longer')
            self.assertNotEqual(old, clip_identity(a))

    def test_incompatible_whole_video_effects_rejected(self):
        rules = [dict(start=1, end=2, mode='skip')]
        for config in ({'upscale_after_swap': True}, {'interp_after_swap': 'rife_2x'},
                       {'mask_engine': 'Segment Anything 2 (tracked)'}):
            with self.assertRaises(ValueError):
                validate_rule_options(rules, config)
        validate_rule_options(rules, {'a_compatibility_mode': True})
        with self.assertRaises(ValueError):
            validate_rule_options([dict(mode='only')], {'a_compatibility_mode': True})

    def test_preview_skip_returns_before_any_detection_or_model_settings(self):
        # Execute the real endpoint function in a small harness, not an imitation
        # of its skip logic. Any access beyond the early return fails this test.
        source = Path(__file__).resolve().parents[1] / 'api.py'
        node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_preview_locked')
        frame = object()
        resolver = Mock(return_value=[dict(start=3, end=5, mode='skip', person_ids=[])])
        detector = Mock(side_effect=AssertionError('detector must not run'))
        namespace = {'_progress': {'processing': False}, '_update_mask_offsets_from_payload': Mock(),
                     '_validate_optional_swap_model': Mock(),
                     'state': SimpleNamespace(selected_target_index=0),
                     'list_files_process': [SimpleNamespace(filename='video.mp4')],
                     'util': SimpleNamespace(is_video=lambda _: True),
                     'get_video_frame': Mock(return_value=frame),
                     '_resolve_frame_rules': resolver, 'get_all_faces': detector,
                     '_bgr_to_preview_dataurl': lambda f: 'unchanged' if f is frame else 'wrong'}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        for fake in (False, True):
            result = namespace['_preview_locked']({'frame': 4, 'fake_preview': fake})
            self.assertEqual(result['image'], 'unchanged')
            self.assertEqual(result['faces'], [])
            self.assertEqual(result['frame_rule'], 'skip')
        detector.assert_not_called()

    def test_job_preparation_keeps_rules_on_selected_video_only(self):
        files = [SimpleNamespace(filename='first'), SimpleNamespace(filename='second')]
        rule = dict(start=3, end=8, mode='skip')
        resolver = Mock(side_effect=lambda raw, entry, payload: raw)
        prepare = load_function('api.py', '_prepare_job_frame_rules', {
            '_target_person_ids': Mock(), 'clip_identity': lambda path: path,
            '_resolve_frame_rules': resolver})
        prepare({'target_index': 1, 'frame_rules': [rule],
                 'frame_rules_by_target': {'second': [rule]}}, {'files': files})
        self.assertEqual(files[0].frame_rules, [])
        self.assertEqual(files[1].frame_rules, [rule])
        resolver.assert_called_once()
        with self.assertRaises(ValueError):
            prepare({'frame_rules_by_target': {'removed': [rule]}}, {'files': files})

    def test_queue_same_name_requires_identity_and_keeps_payload(self):
        files = [SimpleNamespace(filename='a/video.mp4'), SimpleNamespace(filename='b/video.mp4')]
        state = SimpleNamespace()
        progress = {'error': ''}
        run = Mock(side_effect=lambda payload: progress.update(error='sentinel'))
        dispatch = load_function('routes_queue.py', '_run_one', {
            'os': os, 'clip_identity': lambda path: path, 'list_files_process': files,
            'state': state, '_apply_segment': Mock(), '_progress': progress,
            '_snapshot_outputs': None, '_run_swap': run})
        self.assertEqual(dispatch({'target_name': 'video.mp4'})[0], 'failed')
        run.assert_not_called()
        rules = {'b/video.mp4': [dict(start=3, end=8, mode='skip')]}
        result = dispatch({'target_name': 'video.mp4', 'target_clip_id': 'b/video.mp4',
                           'payload': {'frame_rules_by_target': rules}})
        self.assertEqual(result, ('failed', 'sentinel'))
        self.assertEqual(run.call_args.args[0]['target_index'], 1)
        self.assertEqual(run.call_args.args[0]['frame_rules_by_target'], rules)
        self.assertEqual(dispatch({'target_name': 'video.mp4', 'target_clip_id': 'gone'})[0], 'failed')


if __name__ == '__main__':
    unittest.main()
