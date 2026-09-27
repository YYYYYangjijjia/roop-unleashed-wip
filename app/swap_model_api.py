"""Lightweight API contract for locally installed optional swap models."""
from pathlib import Path


def alphaface_status(models_dir, registry):
    spec = registry.get('alphaface')
    if not spec:
        return {'supported': False, 'installed': False, 'missing': [], 'local_only': True}
    files = [spec.get('file'), spec.get('projection_file')]
    missing = [name or 'model specification' for name in files
               if not name or not (Path(models_dir) / name).is_file()
               or (Path(models_dir) / name).stat().st_size == 0]
    return {'supported': True, 'installed': not missing, 'missing': missing,
            'local_only': True, 'resolution': spec.get('output_size', 256)}


def validate_alphaface_request(payload, cfg, models_dir, registry):
    model = payload.get('swap_model', getattr(cfg, 'swap_model', 'inswapper'))
    if model != 'alphaface':
        return
    status = alphaface_status(models_dir, registry)
    if not status['supported']:
        raise ValueError('当前后端尚未支持 AlphaFace，请重启 C 后再选择该模型。')
    if payload.get('a_compatibility_mode', getattr(cfg, 'a_compatibility_mode', False)):
        raise ValueError('AlphaFace 不支持 A 兼容处理。请在主页手动关闭 A 兼容处理；平均身份向量可以保留。')
    if not status['installed']:
        raise ValueError('AlphaFace 本地文件缺失：' + ', '.join(status['missing']) + '。不会自动下载或回退其他模型。')
