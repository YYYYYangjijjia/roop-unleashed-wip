"""Install the two optional AlphaFace assets with pinned hashes.

The application keeps AlphaFace local-only during preview/render. Run this
installer explicitly or add it to runtime.local.json optional_models.
"""
import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from roop.model_registry import ModelSpec, ensure_model_downloaded
from roop.runtime_paths import models_directory, runtime_config


ASSETS = (
    ModelSpec(
        key='alphaface_swapper',
        filename='alphaface_trial/alphaface_swapper_fused_norm.onnx',
        url='https://github.com/kodek4/VisoMaster-Fusion/releases/download/alphaface-model-v1/alphaface_swapper_fused_norm.onnx',
        sha256='5514d967ab6cc27e1b0edc092e05ee97d235adccb4da68574a9b1a1e221a4c6a',
        size_bytes=554528732,
        template='alphaface',
        description='Pinned AlphaFace ONNX export',
    ),
    ModelSpec(
        key='alphaface_projection',
        filename='alphaface_trial/emp.npy',
        url='https://raw.githubusercontent.com/VisoMasterFusion/VisoMaster-Fusion/d86cc97f499c39d275b7ef5a0d0ba03ea1b4b2a3/model_assets/alphaface/emp.npy',
        sha256='cee626bc81721d71c5d6cb1f76f830b9ae46f595514b0884dd8ae34785576764',
        size_bytes=1048704,
        template='alphaface',
        description='Pinned AlphaFace W600K identity projection',
    ),
)


def main():
    if runtime_config().get('offline'):
        os.environ['ROOP_OFFLINE'] = '1'
    root = models_directory()
    for spec in ASSETS:
        path = ensure_model_downloaded(spec, str(root))
        print(f'Verified {path}')


if __name__ == '__main__':
    main()
