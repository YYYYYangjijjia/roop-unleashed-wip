"""Smoke-test the dependencies and models used by the portable default profile.

Run after install_default_models.py, with the same Python as the backend. This
uses the real application imports and UV Tasks constructor, without a video.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def main() -> None:
    for module in (
        "torch", "torchvision", "onnxruntime", "cv2", "av", "insightface",
        "gradio", "fastapi", "mediapipe", "roop.core", "api",
        "roop.processors.FaceSwapInsightFace",
        "roop.processors.Mask_XSeg",
        "roop.processors.Expression_LivePortrait",
    ):
        importlib.import_module(module)
        print(f"READY {module}", flush=True)

    from roop.uv_identity_detail import UVIdentityDetail

    engine = UVIdentityDetail()
    engine.detector.close()
    print("READY UV identity detail (FaceLandmarker and skin parser)", flush=True)
    print("Default preview/render runtime is ready.", flush=True)


if __name__ == "__main__":
    main()
