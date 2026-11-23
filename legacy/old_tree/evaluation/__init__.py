"""
Evaluation package initialization.
"""

from evaluation.distance_engine import DistanceEngine
from evaluation.correlation import EvaluationEngine, CorrelationResult
from evaluation.unified_engine import UnifiedEvaluationEngine
from evaluation.gradcam import GradCAM, compute_spatial_attention_map, overlay_heatmap
