"""
Semantic weighting package initialization.
"""

from semantic_weighting.base_weighter import BaseSemanticWeighter
from semantic_weighting.segmentation_weighter import SegmentationWeighter
from semantic_weighting.tlda_weighter import TLDAWeighter
from semantic_weighting.saliency_weighter import SaliencyWeighter


def make_semantic_weighter(
    weighter_id: str,
    background_floor: float = 0.1,
    **kwargs,
) -> BaseSemanticWeighter:
    """Factory function to build semantic weighters.

    Supported weighter_ids:
    - 'segmentation', 'gt_mask' -> SegmentationWeighter
    - 'tlda' -> TLDAWeighter
    - 'saliency', 'seqa', 'madi' -> SaliencyWeighter
    """
    if weighter_id in ["segmentation", "gt_mask"]:
        return SegmentationWeighter(background_floor=background_floor)
    elif weighter_id == "tlda":
        return TLDAWeighter(background_floor=background_floor, **kwargs)
    elif weighter_id in ["saliency", "seqa", "madi"]:
        return SaliencyWeighter(background_floor=background_floor, **kwargs)
    else:
        raise ValueError(f"Unknown weighter_id: '{weighter_id}'")
