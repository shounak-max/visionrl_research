"""
Consistency Checker for Visual RL Paper Artifacts.

Reads all JSON files in paper_artifacts/ and checks all known invariants.
Flags discrepancies between CONTEXT.md and JSON data sources.

Exit code 0: all checks passed (or only known-flagged issues)
Exit code 1: consistency violations detected
"""

import json
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Ground-truth constants extracted from paper_artifacts/paper_headline_results_combined.json
# (These are the JSON-authoritative numbers; CONTEXT.md deviations are flagged below)
# ---------------------------------------------------------------------------

# Success threshold definition
SUCCESS_THRESHOLD_PX = 5.0        # mean CLE < 5px => tracked successfully
FLOOR_COLLAPSE_THRESHOLD_PX = 15.0  # floor collapse = episode mean > 15px (not 5px)
RANDOM_FLOOR_PX = 22.0            # random policy floor ~22 px

# Phase-0 headline numbers (from paper_artifacts/phase0_sanity_check_results.json)
PHASE0_VISUAL_PURSUIT_PEARSON_R = -0.8853     # from brief / CONTEXT.md (rounded)
PHASE0_N_SAMPLES = 2500

# Table 1 (from paper_headline_results_combined.json)
TABLE1_CLASSICAL_DREP_RHO = 0.0926           # overall_rho for Classical d_rep
TABLE1_SEG_WEIGHTER_RHO   = 0.1075           # overall_rho for SegmentationWeighter (best)
TABLE1_SEG_WEIGHTER_P     = 0.0315           # overall_pvalue

# Table 2 (from paper_headline_results_combined.json — JSON is authoritative)
# CONTEXT.md says End-to-End CNN k=10 geodesic = 2.4133  <<< DISCREPANCY
# JSON says End-to-End CNN k=10 geodesic = 2.2297
TABLE2_E2E_CNN_K10_GEODESIC_JSON    = 2.2297  # JSON value (authoritative)
TABLE2_E2E_CNN_K10_GEODESIC_CONTEXT = 2.4133  # CONTEXT.md value (conflicting)

# Table 3 (from paper_headline_results_combined.json)
# CONTEXT.md says safeguarded floor_collapse_rate = 26.7% (0.267)  <<< DISCREPANCY
# JSON says safeguarded floor_collapse_rate = 0.2 (20.0%)
TABLE3_UNADAPTED_ERROR  = 14.49
TABLE3_UNADAPTED_FLOOR  = 0.4           # 40.0%
TABLE3_NAIVE_ERROR      = 14.49
TABLE3_NAIVE_FLOOR      = 0.4           # 40.0%
TABLE3_SAFE_ERROR       = 13.12
TABLE3_SAFE_FLOOR_JSON  = 0.2           # 20.0% — JSON authoritative
TABLE3_SAFE_FLOOR_CTX   = 0.267         # 26.7% — CONTEXT.md value (conflicting)

# Trust-region rates (suspicious: should not both be 1.0 for safeguarded)
TABLE3_NAIVE_TR_RATE    = 1.0
TABLE3_SAFE_TR_RATE     = 1.0           # THIS IS SUSPICIOUS — flag as known issue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def approx_eq(a: float, b: float, tol: float = 0.001) -> bool:
    return abs(a - b) <= tol


def check_key_path(data: dict, *keys):
    """Navigate nested dict, return value or None if path doesn't exist."""
    cur = data
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


# ---------------------------------------------------------------------------
# Main consistency checks
# ---------------------------------------------------------------------------

def run_checks(artifacts_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """
    Returns (issues, warnings, notes):
      - issues: hard failures (exit 1)
      - warnings: flagged discrepancies (not exit 1 by themselves, but reported)
      - notes: informational
    """
    issues = []
    warnings = []
    notes = []

    # --- Load JSON files ---
    headline_path = artifacts_dir / "paper_headline_results_combined.json"
    phase0_path   = artifacts_dir / "phase0_sanity_check_results.json"
    phase3_path   = artifacts_dir / "phase3_ttss_results.json"
    table3_path   = artifacts_dir / "paper_table3_ttss.json"
    table2_path   = artifacts_dir / "paper_table2_geometry.json"

    for p in [headline_path, phase0_path, phase3_path, table3_path, table2_path]:
        if not p.exists():
            issues.append(f"MISSING REQUIRED FILE: {p}")

    if issues:
        return issues, warnings, notes  # can't proceed

    headline = load_json(headline_path)
    phase0   = load_json(phase0_path)
    phase3   = load_json(phase3_path)
    table3   = load_json(table3_path)
    table2   = load_json(table2_path)

    # ==================================================================
    # CHECK 1: Success threshold definition (5 px)
    # This is a definitional invariant: we just verify JSON files don't
    # contradict the stated floor threshold.
    # ==================================================================
    # The floor_collapse_rate threshold in Table 3 header says >= 15 px.
    # Verify TTSS errors are consistent with floor threshold being 15 px (not 5 px).
    unadapted_error = check_key_path(headline, "table3_ttss", "unadapted_baseline", "mean_tracking_error_px")
    if unadapted_error is not None:
        if unadapted_error > SUCCESS_THRESHOLD_PX:
            notes.append(
                f"CHECK-1 PASS: Unadapted baseline error {unadapted_error:.4f}px > "
                f"success threshold {SUCCESS_THRESHOLD_PX}px — expected (bad tracking)."
            )
        else:
            issues.append(
                f"CHECK-1 FAIL: Unadapted baseline error {unadapted_error:.4f}px is BELOW "
                f"success threshold {SUCCESS_THRESHOLD_PX}px — something is wrong."
            )

        if unadapted_error > FLOOR_COLLAPSE_THRESHOLD_PX:
            notes.append(
                f"CHECK-1b PASS: Unadapted baseline error {unadapted_error:.4f}px > "
                f"floor-collapse threshold {FLOOR_COLLAPSE_THRESHOLD_PX}px — consistent with floor collapse."
            )
        else:
            warnings.append(
                f"CHECK-1b WARN: Unadapted baseline mean error {unadapted_error:.4f}px is BELOW "
                f"floor-collapse threshold {FLOOR_COLLAPSE_THRESHOLD_PX}px. "
                f"This seems inconsistent given 40% floor-collapse rate."
            )

    # ==================================================================
    # CHECK 2: Grassmann k=10, End-to-End CNN geodesic discrepancy
    # CONTEXT.md says 2.4133; JSON says 2.2297
    # ==================================================================
    e2e_k10_json = check_key_path(headline, "table2_geometry", "End-to-End CNN", "k=10", "geodesic_distance")
    if e2e_k10_json is None:
        issues.append("CHECK-2 FAIL: headline JSON missing End-to-End CNN k=10 geodesic_distance.")
    else:
        if approx_eq(e2e_k10_json, TABLE2_E2E_CNN_K10_GEODESIC_JSON):
            notes.append(
                f"CHECK-2 JSON OK: End-to-End CNN k=10 geodesic = {e2e_k10_json:.4f} "
                f"(matches expected JSON value {TABLE2_E2E_CNN_K10_GEODESIC_JSON})."
            )
        else:
            issues.append(
                f"CHECK-2 FAIL: headline JSON End-to-End CNN k=10 geodesic = {e2e_k10_json:.4f}, "
                f"expected {TABLE2_E2E_CNN_K10_GEODESIC_JSON}."
            )

        # Cross-check with CONTEXT.md value
        if not approx_eq(e2e_k10_json, TABLE2_E2E_CNN_K10_GEODESIC_CONTEXT):
            warnings.append(
                f"CHECK-2 DISCREPANCY: CONTEXT.md says End-to-End CNN k=10 geodesic = "
                f"{TABLE2_E2E_CNN_K10_GEODESIC_CONTEXT}, but JSON says {e2e_k10_json:.4f}. "
                f"JSON is authoritative — CONTEXT.md is WRONG here."
            )

    # Also cross-check table2 file
    t2_e2e_k10 = check_key_path(table2, "End-to-End CNN", "k=10", "geodesic_distance")
    if t2_e2e_k10 is not None and e2e_k10_json is not None:
        if approx_eq(t2_e2e_k10, e2e_k10_json):
            notes.append(f"CHECK-2b PASS: paper_table2_geometry.json consistent with headline: {t2_e2e_k10:.4f}.")
        else:
            issues.append(
                f"CHECK-2b FAIL: paper_table2_geometry.json End-to-End CNN k=10 = {t2_e2e_k10:.4f} "
                f"differs from headline {e2e_k10_json:.4f}."
            )

    # ==================================================================
    # CHECK 3: TTSS floor_collapse_rate — JSON vs CONTEXT.md discrepancy
    # JSON says 0.2; CONTEXT.md says 0.267 for safeguarded TTSS
    # ==================================================================
    safe_floor_headline = check_key_path(headline, "table3_ttss", "safeguarded_ttss", "floor_collapse_rate")
    safe_floor_phase3   = check_key_path(phase3, "safeguarded_ttss", "floor_collapse_rate")
    safe_floor_table3   = check_key_path(table3, "safeguarded_ttss", "floor_collapse_rate")

    for name, val in [("headline", safe_floor_headline), ("phase3", safe_floor_phase3), ("table3", safe_floor_table3)]:
        if val is None:
            issues.append(f"CHECK-3 FAIL: {name} JSON missing safeguarded_ttss.floor_collapse_rate.")
        elif approx_eq(val, TABLE3_SAFE_FLOOR_JSON):
            notes.append(f"CHECK-3 PASS: {name} safeguarded floor_collapse_rate = {val} (= 0.2 = 20.0%).")
        elif approx_eq(val, TABLE3_SAFE_FLOOR_CTX):
            warnings.append(
                f"CHECK-3 DISCREPANCY: {name} safeguarded floor_collapse_rate = {val} "
                f"(= {val*100:.1f}% — matches CONTEXT.md's 26.7% claim, NOT JSON authoritative 0.2/20.0%). "
                f"This needs resolution."
            )
        else:
            issues.append(
                f"CHECK-3 FAIL: {name} safeguarded floor_collapse_rate = {val}, "
                f"expected 0.2 (JSON) or 0.267 (CONTEXT.md). Unexpected value."
            )

    warnings.append(
        "CHECK-3 DISCREPANCY (CONTEXT vs JSON): CONTEXT.md states safeguarded TTSS floor_collapse_rate = 26.7% "
        f"(0.267). All JSON files say {TABLE3_SAFE_FLOOR_JSON} (20.0%). "
        "JSON files are authoritative. CONTEXT.md may have been written from an earlier run."
    )

    # ==================================================================
    # CHECK 4: trust_region_acceptance_rate for safeguarded TTSS
    # Both naive and safeguarded show 1.0 — this is SUSPICIOUS
    # ==================================================================
    safe_tr = check_key_path(headline, "table3_ttss", "safeguarded_ttss", "trust_region_acceptance_rate")
    naive_tr = check_key_path(headline, "table3_ttss", "naive_ttt", "trust_region_acceptance_rate")

    if safe_tr is None:
        issues.append("CHECK-4 FAIL: headline JSON missing safeguarded_ttss.trust_region_acceptance_rate.")
    elif approx_eq(safe_tr, 1.0, tol=0.001):
        warnings.append(
            f"CHECK-4 CRITICAL ISSUE: safeguarded_ttss trust_region_acceptance_rate = {safe_tr} "
            f"(= 100%). The trust-region safeguard accepted EVERY candidate update — it effectively "
            f"behaved like naive TTT. This contradicts the paper's claim of 'safeguarded adaptation'. "
            f"This is a KNOWN PROBLEM and must be prominently disclosed in the TTA section."
        )

    if naive_tr is None:
        issues.append("CHECK-4b FAIL: headline JSON missing naive_ttt.trust_region_acceptance_rate.")
    elif approx_eq(naive_tr, 1.0, tol=0.001):
        notes.append(
            f"CHECK-4b NOTE: naive_ttt trust_region_acceptance_rate = {naive_tr} (= 100%). "
            f"Expected — naive TTT uses force_accept=True."
        )

    # Cross-check phase3 file
    p3_safe_tr = check_key_path(phase3, "safeguarded_ttss", "trust_region_acceptance_rate")
    if p3_safe_tr is not None and safe_tr is not None:
        if approx_eq(p3_safe_tr, safe_tr):
            notes.append(f"CHECK-4c PASS: phase3_ttss_results.json safeguarded TR rate = {p3_safe_tr} matches headline.")
        else:
            issues.append(
                f"CHECK-4c FAIL: phase3_ttss_results.json safeguarded TR rate = {p3_safe_tr}, "
                f"headline says {safe_tr}. These should match."
            )

    # ==================================================================
    # CHECK 5: Phase-0 sanity check — visual pursuit Pearson r
    # ==================================================================
    vp_pearson = check_key_path(phase0, "visual_pursuit", "pearson_r")
    if vp_pearson is None:
        issues.append("CHECK-5 FAIL: phase0_sanity_check_results.json missing visual_pursuit.pearson_r.")
    elif approx_eq(vp_pearson, PHASE0_VISUAL_PURSUIT_PEARSON_R, tol=0.01):
        notes.append(
            f"CHECK-5 PASS: Phase-0 visual pursuit Pearson r = {vp_pearson:.4f} ≈ {PHASE0_VISUAL_PURSUIT_PEARSON_R}."
        )
    else:
        warnings.append(
            f"CHECK-5 NOTE: Phase-0 (paper_artifacts) visual pursuit Pearson r = {vp_pearson:.4f}. "
            f"Expected ≈ {PHASE0_VISUAL_PURSUIT_PEARSON_R} (CONTEXT.md / brief). "
            f"The root-level phase0_sanity_check_results.json has a different value (-0.4267) — "
            f"this may be from a different run."
        )

    # Cross-check root-level phase0 vs paper_artifacts phase0
    root_phase0_path = artifacts_dir.parent / "phase0_sanity_check_results.json"
    if root_phase0_path.exists():
        root_phase0 = load_json(root_phase0_path)
        root_vp = check_key_path(root_phase0, "visual_pursuit", "pearson_r")
        pa_vp   = check_key_path(phase0, "visual_pursuit", "pearson_r")
        if root_vp is not None and pa_vp is not None:
            if approx_eq(root_vp, pa_vp, tol=0.001):
                notes.append(f"CHECK-5b PASS: root and paper_artifacts phase0 pearson_r agree: {root_vp:.4f}.")
            else:
                warnings.append(
                    f"CHECK-5b DISCREPANCY: root phase0_sanity_check_results.json pearson_r = {root_vp:.4f}, "
                    f"paper_artifacts version = {pa_vp:.4f}. These are from different runs! "
                    f"paper_artifacts is authoritative."
                )

    # ==================================================================
    # CHECK 6: Table 1 — Classical d_rep overall_rho
    # ==================================================================
    t1_drep = check_key_path(headline, "table1_correlation", "Classical d_rep", "overall_rho")
    if t1_drep is None:
        issues.append("CHECK-6 FAIL: headline missing table1_correlation['Classical d_rep'].overall_rho.")
    elif approx_eq(t1_drep, TABLE1_CLASSICAL_DREP_RHO, tol=0.001):
        notes.append(f"CHECK-6 PASS: Classical d_rep overall_rho = {t1_drep:.4f} ✓")
    else:
        issues.append(
            f"CHECK-6 FAIL: Classical d_rep overall_rho = {t1_drep:.4f}, expected {TABLE1_CLASSICAL_DREP_RHO}."
        )

    # ==================================================================
    # CHECK 7: Table 1 — SegmentationWeighter (best) overall_rho and p-value
    # ==================================================================
    t1_seg = check_key_path(headline, "table1_correlation", "SegmentationWeighter (GT Mask)", "overall_rho")
    t1_seg_p = check_key_path(headline, "table1_correlation", "SegmentationWeighter (GT Mask)", "overall_pvalue")
    if t1_seg is None:
        issues.append("CHECK-7 FAIL: headline missing SegmentationWeighter overall_rho.")
    elif approx_eq(t1_seg, TABLE1_SEG_WEIGHTER_RHO, tol=0.001):
        notes.append(f"CHECK-7 PASS: SegmentationWeighter overall_rho = {t1_seg:.4f} ✓")
    else:
        issues.append(f"CHECK-7 FAIL: SegmentationWeighter overall_rho = {t1_seg:.4f}, expected {TABLE1_SEG_WEIGHTER_RHO}.")

    if t1_seg_p is not None and not approx_eq(t1_seg_p, TABLE1_SEG_WEIGHTER_P, tol=0.001):
        warnings.append(f"CHECK-7b NOTE: SegmentationWeighter p-value = {t1_seg_p:.4f}, expected ≈ {TABLE1_SEG_WEIGHTER_P}.")

    # ==================================================================
    # CHECK 8: Root-level phase3_ttss_results.json vs paper_artifacts version
    # ==================================================================
    root_phase3_path = artifacts_dir.parent / "phase3_ttss_results.json"
    if root_phase3_path.exists():
        root_p3 = load_json(root_phase3_path)
        pa_safe_error = check_key_path(phase3, "safeguarded_ttss", "mean_tracking_error_px")
        root_safe_error = check_key_path(root_p3, "safeguarded_ttss", "mean_tracking_error_px")
        if root_safe_error is not None and pa_safe_error is not None:
            if approx_eq(root_safe_error, pa_safe_error, tol=0.5):
                notes.append(f"CHECK-8 PASS: root and paper_artifacts phase3 safeguarded error agree ({root_safe_error:.2f} vs {pa_safe_error:.2f}).")
            else:
                warnings.append(
                    f"CHECK-8 DISCREPANCY: root phase3 safeguarded error = {root_safe_error:.4f}px, "
                    f"paper_artifacts = {pa_safe_error:.4f}px. These are from DIFFERENT runs. "
                    f"Root file shows ~21.2px; paper_artifacts shows ~13.1px. "
                    f"paper_artifacts is the authoritative published result."
                )

    return issues, warnings, notes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ws_root = Path(__file__).parent.parent  # repo root
    artifacts_dir = ws_root / "paper_artifacts"

    if not artifacts_dir.exists():
        print(f"ERROR: paper_artifacts/ not found at {artifacts_dir}")
        sys.exit(1)

    print("=" * 70)
    print("VISUAL RL PAPER CONSISTENCY CHECKER")
    print(f"Artifacts dir: {artifacts_dir}")
    print("=" * 70)

    issues, warnings, notes = run_checks(artifacts_dir)

    print("\n--- INFORMATIONAL NOTES ---")
    for n in notes:
        print(f"  [NOTE] {n}")

    print("\n--- WARNINGS / DISCREPANCIES ---")
    for w in warnings:
        print(f"  [WARN] {w}")

    print("\n--- HARD FAILURES ---")
    if issues:
        for iss in issues:
            print(f"  [FAIL] {iss}")
    else:
        print("  (none)")

    print()
    if issues:
        print(f"CONSISTENCY CHECK FAILED: {len(issues)} hard issue(s), {len(warnings)} discrepancy/warning(s).")
        print("\nIssues list:")
        for i, iss in enumerate(issues, 1):
            print(f"  {i}. {iss}")
        sys.exit(1)
    else:
        print(
            f"CONSISTENCY CHECK PASSED (with {len(warnings)} known discrepancies flagged "
            f"— see WARNINGS above for paper corrections needed)."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
