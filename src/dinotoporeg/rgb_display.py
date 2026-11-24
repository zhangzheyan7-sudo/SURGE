"""Verified RGB substitution for visualization only. Never call during inference."""
import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image


def registry_path():
    configured=os.environ.get('REGISTRATION_RGB_DISPLAY_REGISTRY')
    if configured:return Path(configured)
    for parent in Path(__file__).resolve().parents:
        if (parent/'_unified').is_dir():
            return parent/'_unified/display_rgb/registry.json'
    return Path(__file__).resolve().parent/'display_rgb/registry.json'


@lru_cache(maxsize=16)
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve_optical(optical, dataset=None, sample_id=None, output_path=None):
    """Return display pixels plus provenance; reject mismatched gray crops or shapes."""
    image=np.asarray(optical)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('RGB display resolver expects HxWx3 uint8 pixels')
    meta={'optical_display_status':'original_input','rgb_display_only':'true'}
    if os.environ.get('REGISTRATION_RGB_DISPLAY','1')=='0':
        meta['optical_display_reason']='disabled';return image,meta
    if output_path is not None and (dataset is None or sample_id is None):
        match=re.match(r'(.+)_sample_(.+)_matching_lines$',Path(output_path).stem)
        if match:dataset,sample_id=match.groups()
    if dataset is None or sample_id is None:
        meta['optical_display_reason']='missing_sample_key';return image,meta
    registry=registry_path()
    if not registry.is_file():
        meta['optical_display_reason']='no_registry';return image,meta
    datasets=read_json(str(registry)).get('datasets',{})
    if dataset not in datasets:
        meta['optical_display_reason']='no_verified_RGB_for_dataset';return image,meta
    location=Path(datasets[dataset])
    if not location.is_absolute():location=registry.parent/location
    entry=read_json(str(location)).get('samples',{}).get(str(sample_id),{})
    if entry.get('status')!='verified':
        meta['optical_display_reason']='unresolved_RGB_correspondence';return image,meta
    if list(image.shape[:2])!=entry['crop_size']:
        meta['optical_display_reason']='crop_size_mismatch';return image,meta
    digest=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    if digest==entry['display_rgb_sha256']:
        meta['optical_display_status']='verified_RGB_already_supplied';return image,meta
    if digest!=entry['inference_crop_sha256']:
        meta['optical_display_reason']='input_crop_checksum_mismatch';return image,meta
    path=Path(entry['rgb_path'])
    if not path.is_absolute():path=location.parent/path
    display=np.array(Image.open(path).convert('RGB'))
    if display.shape!=image.shape or hashlib.sha256(display.tobytes()).hexdigest()!=entry['display_rgb_sha256']:
        raise ValueError(f'RGB display asset changed: {path}')
    meta.update(optical_display_status='verified_RGB',optical_display_rgb_path=str(path),
                inference_crop_sha256=digest,optical_display_reason='display_only_after_inference')
    return display,meta
