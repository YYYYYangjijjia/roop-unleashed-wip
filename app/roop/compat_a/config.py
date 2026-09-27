"""Build an isolated settings view for one preview or queued job."""
from types import SimpleNamespace


def request_config(cfg, payload):
    values = dict(vars(cfg))
    # Frame/index/fake_preview are request coordinates, not model settings.
    # Excluding them also lets consecutive previews reuse the same sessions.
    values.update({key: value for key, value in payload.items() if key in values})
    aliases = {'autorotate': 'autorotate_faces', 'enhancer': 'selected_enhancer',
               'upscale': 'subsample_upscale', 'detection': 'face_detection_mode',
               'clip_text': 'mask_clip_text'}
    for source, target in aliases.items():
        if source in payload:
            values[target] = payload[source]
    return SimpleNamespace(**values)
