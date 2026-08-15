"""
Provenance utility for visual RL generalization bounds research codebase.

Ensures every artifact written to cache directories (rollout_cache, embeddings, etc.)
carries the seed, configuration hash, timestamp, and metadata required for 100% reproducibility.
"""

import hashlib
import json
import time
from typing import Any, Dict, Optional


def canonical_json(data: Dict[str, Any]) -> str:
    """Recursively serializes a dictionary into a deterministic canonical JSON string."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Computes a 16-character SHA-256 hash of a configuration dictionary."""
    canonical_str = canonical_json(config)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()[:16]


def stamp_provenance(
    config: Dict[str, Any],
    seed: int,
    env_id: str,
    perturbation_tag: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates a complete provenance dictionary to attach to datasets or cached tensors.

    Args:
        config: Experiment configuration dictionary.
        seed: Random seed used for generation/rollout.
        env_id: Name of environment (e.g. 'coinrun', 'procgen_ood', 'visual_pursuit').
        perturbation_tag: Description of perturbation (e.g. 'clean', 'bg_shift_0.5', 'photometric_0.3').
        extra_metadata: Additional key-value pairs (e.g. git commit, hardware specs).

    Returns:
        Dict containing full provenance provenance record.
    """
    config_hash = compute_config_hash(config)
    provenance_record = {
        "config_hash": config_hash,
        "seed": seed,
        "env_id": env_id,
        "perturbation_tag": perturbation_tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
    }
    if extra_metadata:
        provenance_record["extra_metadata"] = extra_metadata
    return provenance_record


def verify_provenance(
    record: Dict[str, Any], expected_config: Optional[Dict[str, Any]] = None
) -> bool:
    """Validates that a provenance record is self-consistent and matches expected config if provided."""
    if "config" not in record or "config_hash" not in record:
        return False
    computed_hash = compute_config_hash(record["config"])
    if computed_hash != record["config_hash"]:
        return False
    if expected_config is not None:
        if compute_config_hash(expected_config) != computed_hash:
            return False
    return True
