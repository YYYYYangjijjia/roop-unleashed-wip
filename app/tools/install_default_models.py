"""Install the weights used by config.default.yaml, reusing verified local files.

The manifest pins the same files present in the author's working profile.
Other UI models remain optional and are fetched only when selected.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from roop.runtime_paths import DEFAULT_MODELS_DIR, models_directory, runtime_config


BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
BUFFALO_SIZE = 288621354
BUFFALO_FILES = {
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "2d106det.onnx": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
    "1k3d68.onnx": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
    "genderage.onnx": "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}

DEEPFAKE = "https://huggingface.co/countfloyd/deepfake/resolve/ee7d0fe0c284700192ff11e14ecdab5b40185204/"
RETINAFACE = "https://huggingface.co/nakamura196/retinaface-r50-onnx/resolve/b4181a5aca26d7ccf0e236c7c6d9acf5d9c78f2b/"
LIVEPORTRAIT = "https://huggingface.co/warmshao/FasterLivePortrait/resolve/eb937f4bec7186598df2d1f68e1ddbb488ae1de5/liveportrait_onnx/"

# Relative filename, immutable or versioned source, SHA-256 of the local
# working model, and expected byte count. The installer never replaces a
# mismatching existing file in a shared model library.
ASSETS = (
    ("retinaface_r50.onnx", RETINAFACE + "retinaface_r50.onnx", "ea3f7d12894980d52d3fd7f3f5cab463d48e9cfbcc06f40616bdeeab3e3f31d1", 109110500),
    ("inswapper_128.onnx", DEEPFAKE + "inswapper_128.onnx", "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af", 554253681),
    ("xseg.onnx", DEEPFAKE + "xseg.onnx", "0b57328efcb839d85973164b617ceee9dfe6cfcb2c82e8a033bba9f4f09b27e5", 70327737),
    ("liveportrait/appearance_feature_extractor.onnx", LIVEPORTRAIT + "appearance_feature_extractor.onnx", "d070afccca7f528ffb0ef5052b21588b42225a996661833f7bdede562d1ab921", 3355896),
    ("liveportrait/motion_extractor.onnx", LIVEPORTRAIT + "motion_extractor.onnx", "219a46174297b2b411bb3c5dce48d3a8c8e07a9d82120a0da21f49f58b67fca6", 112648514),
    ("liveportrait/warping_spade.onnx", LIVEPORTRAIT + "warping_spade.onnx", "b0e7a566db8fba690c23523bcd2faa4f0d13f05418db84a779397a851062ad69", 421233096),
    ("liveportrait/stitching.onnx", LIVEPORTRAIT + "stitching.onnx", "8e33658425e3f1014dc35b28f60ca7459189e8f78f20540262a2fe5ac1dc6ec7", 182363),
    ("face_landmarker.task", "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff", 3758596),
    ("canonical_face_model.obj", "https://raw.githubusercontent.com/google-ai-edge/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model.obj", "8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618", 45999),
    ("resnet18.onnx", "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx", "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f", 53205364),
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def check_existing(path: Path, expected: str, size: int | None = None) -> bool:
    if not path.exists():
        return False
    if not path.is_file() or (size is not None and path.stat().st_size != size) or digest(path) != expected:
        raise RuntimeError(f"Existing model does not match the default profile; preserved: {path}")
    print(f"REUSE {path}", flush=True)
    return True


def fetch(url: str, path: Path, size: int | None = None, expected: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if size is None:
        amount = "unknown size"
    elif size >= 1048576:
        amount = f"{size / 1048576:.0f} MiB"
    else:
        amount = f"{size / 1024:.0f} KiB"
    print(f"DOWNLOAD {path.name} ({amount})", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "roop-unleashed-default-models/1.0"})
    total = 0
    sha = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=45) as source, part.open("wb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
                sha.update(block)
                total += len(block)
                if total // (32 * 1024 * 1024) > (total - len(block)) // (32 * 1024 * 1024):
                    print(f"  {total / 1048576:.0f} MiB received", flush=True)
        if size is not None and total != size:
            raise RuntimeError(f"Wrong size for {path.name}: {total} != {size}")
        if expected is not None and sha.hexdigest() != expected:
            raise RuntimeError(f"SHA-256 mismatch for {path.name}: {sha.hexdigest()}")
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def ensure_buffalo(root: Path, offline: bool, verify_only: bool) -> None:
    missing = []
    for name, expected in BUFFALO_FILES.items():
        path = root / "buffalo_l" / name
        if not check_existing(path, expected):
            missing.append(name)
    if not missing:
        return
    if offline or verify_only:
        raise RuntimeError(f"Missing buffalo_l models: {', '.join(missing)}")
    archive = root / "buffalo_l.zip"
    if not archive.exists():
        fetch(BUFFALO_URL, archive, BUFFALO_SIZE)
    elif not archive.is_file() or archive.stat().st_size != BUFFALO_SIZE:
        raise RuntimeError(f"Existing buffalo_l archive is invalid; preserved: {archive}")
    with zipfile.ZipFile(archive) as bundle:
        archive_names = set(bundle.namelist())
        for name in missing:
            # The official v0.7 archive stores ONNX files at its root. Also
            # accept a nested copy so an already-downloaded archive can be used.
            member = next((candidate for candidate in (name, "buffalo_l/" + name)
                           if candidate in archive_names), None)
            if member is None:
                raise RuntimeError(f"buffalo_l.zip is missing {name}; preserved: {archive}")
            dest = root / "buffalo_l" / name
            part = dest.with_name(dest.name + ".part")
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                with bundle.open(member) as source, part.open("wb") as output:
                    sha = hashlib.sha256()
                    while block := source.read(1024 * 1024):
                        output.write(block)
                        sha.update(block)
                if sha.hexdigest() != BUFFALO_FILES[name]:
                    raise RuntimeError(f"SHA-256 mismatch inside buffalo_l.zip: {member}")
                os.replace(part, dest)
                print(f"VERIFIED {dest}", flush=True)
            finally:
                part.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="check local files without network access")
    args = parser.parse_args()
    config = runtime_config()
    if not os.environ.get("ROOP_MODELS_DIR") and config.get("models_dir", "app/models") == "app/models":
        DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    root = models_directory()
    offline = bool(config.get("offline")) or os.environ.get("ROOP_OFFLINE", "").lower() in ("1", "true", "yes", "on")
    ensure_buffalo(root, offline, args.verify_only)
    for filename, url, expected, size in ASSETS:
        path = root / filename
        if check_existing(path, expected, size):
            continue
        if offline or args.verify_only:
            raise RuntimeError(f"Missing default-profile model: {path}")
        fetch(url, path, size, expected)
        print(f"VERIFIED {path}", flush=True)
    print("Default model profile is complete.", flush=True)


if __name__ == "__main__":
    main()
