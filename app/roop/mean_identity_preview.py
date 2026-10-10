"""A face-only diagnostic render of a FaceSet's averaged identity vector.

An ArcFace embedding has no unique inverse image.  InSwapper is used as a
conditional renderer on a fixed, smoothed source-face crop; the result shows
what that swap model can synthesize from the same mean vector used at runtime.
The original aligned crop is returned beside it for texture comparison.
"""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


def render_mean_identity(faceset, model_path, swapper_factory=None):
    """Return (source reference, mean-vector model render, source index).

    The guide has its fine skin texture removed before inference so freckles in
    the original image cannot trivially copy into the diagnostic render.
    Inference uses CPU to avoid allocating another CUDA context while a user is
    inspecting a faceset. It runs only when explicitly requested from the UI.
    """
    if faceset is None or not faceset.faces:
        raise ValueError("Load a source face or faceset first.")
    reference, index = faceset.identity_detail_reference()
    if reference is None:
        raise ValueError("The selected faceset has no usable reference image and landmarks.")
    mean_face = faceset.average_identity_face()
    if mean_face is None:
        raise ValueError("The selected faceset has no identity embedding.")

    model_path = Path(model_path)
    if swapper_factory is None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Local InSwapper model is missing: {model_path}")
        import onnxruntime
        from insightface.model_zoo.inswapper import INSwapper

        session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        swapper = INSwapper(model_file=str(model_path), session=session)
    else:
        swapper = swapper_factory(model_path)

    # Reference is already an ArcFace 256 crop. A median filter removes small
    # source marks from the *guide* without blurring the eyes, nose, and mouth
    # that the swap model needs for a usable conditional render.
    # The comparison reference remains untouched. Output stays native 128px.
    guide = cv2.medianBlur(reference, 9)
    from roop.face_util import swap_template_points
    guide_face = SimpleNamespace(kps=swap_template_points(256, "arcface"))
    rendered, _ = swapper.get(guide, guide_face, mean_face, paste_back=False)
    rendered = np.asarray(rendered)
    if (rendered.shape != (128, 128, 3) or
            not np.isfinite(rendered).all()):
        raise RuntimeError("InSwapper did not produce a valid 128px face crop.")
    return reference, rendered.astype(np.uint8), index
