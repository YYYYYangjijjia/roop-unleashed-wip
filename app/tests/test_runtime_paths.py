"""Configured model roots must reach both simple and model-specific loaders."""

import os
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop import runtime_paths
from roop.model_loader import get_models_directory
from roop.model_registry import ModelSpec, ensure_model_downloaded
from roop.utilities import resolve_relative_path


class RuntimePathsTest(unittest.TestCase):
    def test_default_and_external_root(self):
        with tempfile.TemporaryDirectory() as workspace:
            with patch.object(runtime_paths, 'LOCAL_CONFIG', Path(workspace) / 'absent.json'):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop('ROOP_MODELS_DIR', None)
                    self.assertEqual(runtime_paths.models_directory(), runtime_paths.DEFAULT_MODELS_DIR)
                external = Path(workspace) / 'shared models'
                external.mkdir()
                with patch.dict(os.environ, {'ROOP_MODELS_DIR': str(external)}):
                    self.assertEqual(Path(resolve_relative_path('../models/buffalo_l/det_10g.onnx')),
                                     external / 'buffalo_l' / 'det_10g.onnx')
                    self.assertEqual(Path(get_models_directory()), external)
                    self.assertEqual(Path(resolve_relative_path('..')),
                                     runtime_paths.PROJECT_ROOT / 'app')

    def test_missing_explicit_root_fails_instead_of_creating_library(self):
        with tempfile.TemporaryDirectory() as workspace:
            missing = Path(workspace) / 'typo'
            with patch.dict(os.environ, {'ROOP_MODELS_DIR': str(missing)}):
                with self.assertRaisesRegex(FileNotFoundError, 'Configured model directory'):
                    resolve_relative_path('../models/inswapper_128.onnx')
            self.assertFalse(missing.exists())

    def test_offline_shared_model_with_wrong_hash_is_preserved(self):
        with tempfile.TemporaryDirectory() as workspace:
            model = Path(workspace) / 'shared.onnx'
            model.write_bytes(b'user-owned model')
            spec = ModelSpec('shared', 'shared.onnx', 'https://example.invalid/shared.onnx',
                             '0' * 64, len(b'user-owned model'), 'arcface', 'test')
            with patch('roop.utilities.network_downloads_allowed', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'failed integrity check'):
                    ensure_model_downloaded(spec, workspace)
            self.assertEqual(model.read_bytes(), b'user-owned model')

    def test_failed_online_replacement_preserves_shared_model(self):
        with tempfile.TemporaryDirectory() as workspace:
            model = Path(workspace) / 'shared.onnx'
            model.write_bytes(b'user-owned model')
            expected = hashlib.sha256(b'correct download').hexdigest()
            spec = ModelSpec('shared', 'shared.onnx', 'https://example.invalid/shared.onnx',
                             expected, len(b'correct download'), 'arcface', 'test')
            with patch('roop.utilities.network_downloads_allowed', return_value=True), \
                 patch('roop.model_registry.urllib.request.urlopen', return_value=io.BytesIO(b'bad download')):
                with self.assertRaisesRegex(ValueError, 'failed integrity check'):
                    ensure_model_downloaded(spec, workspace)
            self.assertEqual(model.read_bytes(), b'user-owned model')


if __name__ == '__main__':
    unittest.main()
