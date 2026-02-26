__version__ = '0.1.0'
"""
vsl/ - VSL Recognition Package
================================
Import nhanh:
    from vsl import cfg, DualTransformer, load_model
    from vsl import RealtimeExtractor, VideoExtractor
    from vsl import InferenceEngine
    from vsl import UIRenderer
    from vsl import get_display_name, is_idle_label, resample_sequence
"""

from vsl.config           import cfg, FACE_KEY_INDICES, KEY_BLENDSHAPES, MODEL_URLS
from vsl.utils            import (download_model, load_display_names,
                                   get_display_name, save_display_name,
                                   is_idle_label, resample_sequence)
from vsl.model            import DualTransformer, load_model
from vsl.extractor        import RealtimeExtractor, VideoExtractor
from vsl.inference_engine import InferenceEngine
from vsl.ui               import UIRenderer

__all__ = [
    'cfg', 'FACE_KEY_INDICES', 'KEY_BLENDSHAPES', 'MODEL_URLS',
    'download_model', 'load_display_names', 'get_display_name',
    'save_display_name', 'is_idle_label', 'resample_sequence',
    'DualTransformer', 'load_model',
    'RealtimeExtractor', 'VideoExtractor',
    'InferenceEngine',
    'UIRenderer',
]