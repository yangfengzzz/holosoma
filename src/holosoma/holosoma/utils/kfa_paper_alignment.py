"""Paper-alignment evidence helpers for KungFuAthlete Ground recovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PaperAlignmentEvidenceItem:
    key: str
    status: str
    paper_basis: str
    repo_basis: str
    summary: str


def build_kfa_ground_paper_alignment_evidence() -> list[dict[str, str]]:
    """Return the current paper-alignment evidence table for Ground recovery.

    Status values are intentionally restricted to:
    - ``matched``
    - ``mismatch``
    - ``paper_unspecified``
    """
    items = [
        PaperAlignmentEvidenceItem(
            key="recovery_phase_switching",
            status="matched",
            paper_basis="Paper Section 4.3 describes hybrid motion-tracking and recovery rewards as phase-specific.",
            repo_basis="RewardManager gates `r_mtr` and `r_rc` terms exclusively by `MotionCommand.recovery_active_mask()`.",
            summary="Tracking rewards apply outside recovery, recovery rewards apply only during recovery.",
        ),
        PaperAlignmentEvidenceItem(
            key="recovery_shoulder_gap_threshold",
            status="matched",
            paper_basis="Paper Section 5.1 sets the shoulder-height tracking threshold to 1.0 for recovery rewards.",
            repo_basis="`MotionConfig.recovery_shoulder_height_threshold` and recovery termination config both use `1.0`.",
            summary="Recovery activation and related reward/termination paths share the paper-facing threshold.",
        ),
        PaperAlignmentEvidenceItem(
            key="feet_slip_contact_threshold",
            status="matched",
            paper_basis="Paper Section 5.1 defines foot slip using vertical contact force greater than 8.",
            repo_basis="`feet_slip_penalty` in the paper-facing presets uses `contact_force_threshold=8.0` on the z-force only.",
            summary="Slip gating matches the paper-facing vertical-force threshold.",
        ),
        PaperAlignmentEvidenceItem(
            key="recovery_bad_tracking_hysteresis",
            status="matched",
            paper_basis="Paper Section 4.4 accumulates consecutive bad-tracking steps during recovery before termination.",
            repo_basis="`RecoveryAwareBadTracking` delays termination during recovery using `max_consecutive_bad_tracking_steps=8`.",
            summary="Immediate failure is disabled during recovery and replaced by hysteresis.",
        ),
        PaperAlignmentEvidenceItem(
            key="grsi_rotation_recombination",
            status="matched",
            paper_basis="Paper Section 4.5 constructs gravity-settled recovery states and augments them by rotational recombination.",
            repo_basis="Recovery dataset generation writes raw GRSI states plus processed `rotation_recombination` root states.",
            summary="The training-ready recovery dataset follows the paper-facing GRSI plus recombination flow.",
        ),
        PaperAlignmentEvidenceItem(
            key="low_kinetic_eq17_update_rule",
            status="paper_unspecified",
            paper_basis="The public paper introduces low-kinetic sampling but does not disclose the exact Eq. 17 anchor-weight update rule.",
            repo_basis="`LowKineticAnchorSampler` attributes failures to the nearest preceding anchor as an explicit repo interpretation.",
            summary="Current implementation is documented and test-covered, but not yet exact-paper-verified.",
        ),
        PaperAlignmentEvidenceItem(
            key="domain_randomization_exact_ranges",
            status="paper_unspecified",
            paper_basis="Paper Section 5.1 lists randomized categories but does not publish exact numeric ranges for all factors.",
            repo_basis="The Ground presets define concrete setup/reset randomization ranges in `config_values/wbt/g1/randomization.py`.",
            summary="Randomization coverage matches the paper categories, but exact numeric parity remains underdetermined publicly.",
        ),
    ]
    return [asdict(item) for item in items]
