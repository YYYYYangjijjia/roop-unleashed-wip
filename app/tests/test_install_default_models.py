"""Portable defaults must be installable without touching existing model files."""

import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from tools import install_default_models as models  # noqa: E402


class TestDefaultModelInstaller(unittest.TestCase):
    def test_profile_covers_active_default_settings(self):
        import yaml

        config = yaml.safe_load((APP / "config.default.yaml").read_text(encoding="utf-8"))
        self.assertEqual((config["swap_model"], config["detector_engine"], config["mask_engine"]),
                         ("inswapper", "retinaface_r50", "DFL XSeg"))
        self.assertGreater(config["expression_restore_strength"], 0)
        self.assertGreater(config["identity_detail_strength"], 0)
        self.assertTrue(config["use_3d_recon"])
        filenames = {row[0] for row in models.ASSETS}
        self.assertTrue({"inswapper_128.onnx", "retinaface_r50.onnx", "xseg.onnx",
                         "liveportrait/warping_spade.onnx", "face_landmarker.task",
                         "canonical_face_model.obj", "resnet18.onnx"} <= filenames)

    def test_download_checks_hash_and_never_replaces_a_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            source.write_bytes(b"verified model")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            target = root / "models" / "test.onnx"
            models.fetch(source.as_uri(), target, source.stat().st_size, expected)
            self.assertTrue(models.check_existing(target, expected, source.stat().st_size))
            target.write_bytes(b"different model")
            with self.assertRaisesRegex(RuntimeError, "preserved"):
                models.check_existing(target, expected, source.stat().st_size)
            self.assertEqual(target.read_bytes(), b"different model")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                models.fetch(source.as_uri(), root / "wrong.onnx", source.stat().st_size, "0" * 64)
            self.assertFalse((root / "wrong.onnx").exists())
            self.assertFalse((root / "wrong.onnx.part").exists())

    def test_buffalo_archive_extracts_verified_members(self):
        for prefix in ("", "buffalo_l/"):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload = b"buffalo model"
                expected = hashlib.sha256(payload).hexdigest()
                archive = root / "source.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.writestr(prefix + "det_10g.onnx", payload)
                with patch.object(models, "BUFFALO_FILES", {"det_10g.onnx": expected}), \
                     patch.object(models, "BUFFALO_URL", archive.as_uri()), \
                     patch.object(models, "BUFFALO_SIZE", archive.stat().st_size):
                    with self.assertRaisesRegex(RuntimeError, "Missing buffalo_l"):
                        models.ensure_buffalo(root / "empty", offline=True, verify_only=False)
                    models.ensure_buffalo(root / "target", offline=False, verify_only=False)
                    self.assertEqual((root / "target" / "buffalo_l" / "det_10g.onnx").read_bytes(), payload)
                    models.ensure_buffalo(root / "target", offline=True, verify_only=True)


if __name__ == "__main__":
    unittest.main()
