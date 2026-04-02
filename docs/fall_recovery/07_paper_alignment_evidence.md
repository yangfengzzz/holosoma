# Paper Alignment Evidence

This table records the current evidence standard for the local Ground fall-recovery implementation.

Status values are restricted to:
- `matched`
- `mismatch`
- `paper_unspecified`

`paper_unspecified` means the public paper does not provide enough detail to prove exact reproduction from text alone.

| Area | Status | Paper Basis | Repo Basis | Review Result |
| --- | --- | --- | --- | --- |
| Recovery phase switching | `matched` | Section 4.3 describes a hybrid objective with motion-tracking and recovery reward phases. | `RewardManager` gates `r_mtr` and `r_rc` terms exclusively through `MotionCommand.recovery_active_mask()`. | Phase switching is explicit and non-additive in code. |
| Shoulder-gap recovery threshold | `matched` | Section 5.1 sets the recovery shoulder-height threshold to `1.0`. | `MotionConfig.recovery_shoulder_height_threshold` and the recovery termination preset both use `1.0`. | Recovery activation threshold is aligned across reward and termination paths. |
| Feet slip contact threshold | `matched` | Section 5.1 defines slip when vertical foot contact force exceeds `8`. | `feet_slip_penalty` uses `contact_force_threshold=8.0` on the z-force only in the paper-facing presets. | Slip gating matches the public paper threshold. |
| Recovery bad-tracking hysteresis | `matched` | Section 4.4 accumulates consecutive bad-tracking steps during recovery before termination. | `RecoveryAwareBadTracking` applies hysteresis with `max_consecutive_bad_tracking_steps=8`. | Recovery does not terminate immediately on the first bad-tracking step. |
| GRSI plus rotation recombination | `matched` | Section 4.5 builds gravity-settled recovery states and augments them by rotational recombination. | `generate_recovery_dataset.py` writes raw GRSI data and processed `rotation_recombination` recovery-init data. | Recovery dataset generation matches the public paper workflow. |
| Low-kinetic Eq. 17 update rule | `paper_unspecified` | The public paper introduces low-kinetic sampling but does not disclose the exact Eq. 17 anchor-weight update rule. | `LowKineticAnchorSampler` uses a documented nearest-preceding-anchor failure attribution rule. | Exact reproduction remains blocked on author code or direct clarification. |
| Domain-randomization exact ranges | `paper_unspecified` | Section 5.1 names the randomized categories but does not publish exact numeric ranges for all factors. | Concrete randomization ranges live in `config_values/wbt/g1/randomization.py`. | Coverage matches the paper categories, but exact numeric equivalence is not fully provable from public text. |

## Review Outcome

- The Ground recovery implementation matches the public paper text on the main recovery reward, termination, and dataset-construction paths.
- Exact paper-verification is still blocked by the unresolved low-kinetic Eq. 17 update rule and incompletely specified randomization ranges.
- Training readiness must therefore continue to depend on the Ground parity runtime gate, not on static code review alone.
