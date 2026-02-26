"""
collector/ - Webcam data collection package
=============================================
    from collector import init_hf, upload_to_hf
    from collector import FullBodyDrawer, draw_text_bg, lm_to_px
    from collector import FramingChecker
    from collector import FacialExpressionAnalyzer
    from collector import InteractionVisualizer
"""

from collector.hf_upload   import init_hf, upload_to_hf, HF_REPO_ID
from collector.drawing      import FullBodyDrawer, draw_text_bg, lm_to_px, lm_dist
from collector.framing      import FramingChecker
from collector.expression   import FacialExpressionAnalyzer
from collector.interaction  import InteractionVisualizer

__all__ = [
    'init_hf', 'upload_to_hf', 'HF_REPO_ID',
    'FullBodyDrawer', 'draw_text_bg', 'lm_to_px', 'lm_dist',
    'FramingChecker',
    'FacialExpressionAnalyzer',
    'InteractionVisualizer',
]