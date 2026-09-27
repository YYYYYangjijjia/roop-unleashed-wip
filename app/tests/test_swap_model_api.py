from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swap_model_api import alphaface_status, validate_alphaface_request


class AlphaFaceApiTests(unittest.TestCase):
    def test_local_install_requires_model_and_projection_and_never_downloads(self):
        spec = {'alphaface': {'file': 'alpha.onnx', 'projection_file': 'emp.npy', 'output_size': 256}}
        cfg = SimpleNamespace(swap_model='inswapper', a_compatibility_mode=False)
        with tempfile.TemporaryDirectory() as path:
            self.assertFalse(alphaface_status(path, spec)['installed'])
            (Path(path) / 'alpha.onnx').write_bytes(b'model')
            with self.assertRaisesRegex(ValueError, 'emp.npy'):
                validate_alphaface_request({'swap_model': 'alphaface'}, cfg, path, spec)
            (Path(path) / 'emp.npy').write_bytes(b'projection')
            self.assertTrue(alphaface_status(path, spec)['installed'])
            validate_alphaface_request({'swap_model': 'alphaface'}, cfg, path, spec)
            with self.assertRaisesRegex(ValueError, 'A 兼容'):
                validate_alphaface_request({'swap_model': 'alphaface', 'a_compatibility_mode': True}, cfg, path, spec)
            self.assertFalse(cfg.a_compatibility_mode)

    def test_unregistered_alphaface_rejected_other_swappers_unchanged(self):
        cfg = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, '重启'):
            validate_alphaface_request({'swap_model': 'alphaface'}, cfg, '.', {})
        validate_alphaface_request({'swap_model': 'inswapper'}, cfg, '.', {})


if __name__ == '__main__':
    unittest.main()
