"""Probes package initialization."""

from visionrl.probes.adaptation import (
    AdaptedPolicy,
    arm_config_table,
    clean_feature_stats,
    compare_arms,
    run_tta,
)
from visionrl.probes.feature_bank import (
    FeatureBank,
    RefBank,
    collect_clean,
    score_episode,
)
from visionrl.probes.metrics import (
    centroid_distance,
    compute_pac_drift,
    gcm_airm_distance,
    grassmann_distance,
    knn_distance,
    semantic_pool,
)

__all__ = [
    "centroid_distance",
    "compute_pac_drift",
    "gcm_airm_distance",
    "semantic_pool",
    "knn_distance",
    "grassmann_distance",
    "FeatureBank",
    "RefBank",
    "collect_clean",
    "score_episode",
    "clean_feature_stats",
    "AdaptedPolicy",
    "run_tta",
    "compare_arms",
    "arm_config_table",
]
