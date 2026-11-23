"""
Evaluation and correlation engine.

Computes Spearman rank correlation (rho) and Pearson (r) between representation distance metrics
and policy performance drops across environments and holdout axes.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


@dataclass
class CorrelationResult:
    """Statistical correlation results summary."""

    metric_name: str
    spearman_rho: float
    spearman_pvalue: float
    pearson_r: float
    pearson_pvalue: float
    num_samples: int
    axis_breakdown: Optional[Dict[str, Dict[str, float]]] = None


class EvaluationEngine:
    """Evaluation Engine for visual RL generalization bounds analysis."""

    @staticmethod
    def correlate(
        distance_metric: Union[np.ndarray, List[float]],
        performance_drop: Union[np.ndarray, List[float]],
        metric_name: str = "d_rep",
        axes: Optional[List[str]] = None,
    ) -> CorrelationResult:
        """Computes Spearman rho and Pearson r correlations between representation distance and performance drop.

        Negative correlation means: higher representation distance predicts larger policy performance drop (lower performance).
        """
        dist_arr = np.array(distance_metric, dtype=np.float64).flatten()
        drop_arr = np.array(performance_drop, dtype=np.float64).flatten()

        assert len(dist_arr) == len(
            drop_arr
        ), f"Length mismatch: {len(dist_arr)} vs {len(drop_arr)}"

        if len(dist_arr) < 2:
            return CorrelationResult(
                metric_name=metric_name,
                spearman_rho=0.0,
                spearman_pvalue=1.0,
                pearson_r=0.0,
                pearson_pvalue=1.0,
                num_samples=len(dist_arr),
            )

        # Handle constant input edge case
        if np.std(dist_arr) < 1e-12 or np.std(drop_arr) < 1e-12:
            rho, rho_p = 0.0, 1.0
            r, r_p = 0.0, 1.0
        else:
            rho, rho_p = stats.spearmanr(dist_arr, drop_arr)
            r, r_p = stats.pearsonr(dist_arr, drop_arr)

        axis_breakdown = None
        if axes is not None and len(axes) == len(dist_arr):
            axis_breakdown = {}
            unique_axes = sorted(list(set(axes)))
            for ax in unique_axes:
                idx = np.where(np.array(axes) == ax)[0]
                if len(idx) >= 2 and np.std(dist_arr[idx]) > 1e-12 and np.std(drop_arr[idx]) > 1e-12:
                    sub_rho, sub_rho_p = stats.spearmanr(dist_arr[idx], drop_arr[idx])
                    sub_r, sub_r_p = stats.pearsonr(dist_arr[idx], drop_arr[idx])
                else:
                    sub_rho, sub_rho_p, sub_r, sub_r_p = 0.0, 1.0, 0.0, 1.0

                axis_breakdown[ax] = {
                    "spearman_rho": float(sub_rho),
                    "spearman_pvalue": float(sub_rho_p),
                    "pearson_r": float(sub_r),
                    "pearson_pvalue": float(sub_r_p),
                    "num_samples": len(idx),
                }

        return CorrelationResult(
            metric_name=metric_name,
            spearman_rho=float(rho),
            spearman_pvalue=float(rho_p),
            pearson_r=float(r),
            pearson_pvalue=float(r_p),
            num_samples=len(dist_arr),
            axis_breakdown=axis_breakdown,
        )

    @staticmethod
    def floor_collapse_rate(
        performance_logs: List[float],
        floor_threshold: float = 15.0,
        mode: str = "upper_bound",
    ) -> float:
        """Computes proportion of evaluation episodes where policy performance drops at/below lower bound (rewards) or at/above upper bound (tracking errors)."""
        if not performance_logs:
            return 0.0
        arr = np.array(performance_logs, dtype=np.float32)
        if mode == "upper_bound":
            collapsed = arr >= floor_threshold
        else:
            collapsed = arr <= floor_threshold
        return float(np.mean(collapsed))

