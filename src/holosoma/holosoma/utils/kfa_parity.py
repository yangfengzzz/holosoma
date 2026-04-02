"""Helpers for reproducible KungFuAthlete Ground parity runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

KFA_RELEASED_MOTION_ROOT_CANDIDATES = (
    "./KungFuAthleteBot/collection_g129dof/org_smoothed_mj",
    "./datasets/KungFuAthleteBot/org_smoothed_mj",
)
KFA_RELEASED_MOTION_FILE_SUFFIXES = (".npz", "_mj.npz")
KFA_RELEASED_MOTION_LAYOUTS = ("flat", "nested")

KFA_TRAINING_EXAMPLE_CLIP_ID = "1317"
KFA_PRIMARY_REGRESSION_CLIP_ID = "1307"
KFA_OPTIONAL_REGRESSION_CLIP_IDS = ("969", "0203")
KFA_PARITY_SEEDS = (1, 2, 3)

KFA_PARITY_PRESETS = {
    "stand": {
        "experiment": "exp:g1-29dof-wbt-stand-fast-sac",
        "needs_recovery_dataset": False,
    },
    "recovery": {
        "experiment": "exp:g1-29dof-wbt-recovery-fast-sac",
        "needs_recovery_dataset": True,
    },
    "recovery_only": {
        "experiment": "exp:g1-29dof-wbt-recovery-only-fast-sac",
        "needs_recovery_dataset": True,
    },
    "slip3": {
        "experiment": "exp:g1-29dof-wbt-recovery-slip3-fast-sac",
        "needs_recovery_dataset": True,
    },
    "slip5": {
        "experiment": "exp:g1-29dof-wbt-recovery-slip5-fast-sac",
        "needs_recovery_dataset": True,
    },
}


@dataclass(frozen=True)
class KfaParityClipResolution:
    clip_id: str
    path: str
    exists: bool
    dataset_root: str
    layout: str


def _candidate_dataset_roots(dataset_root: str | None = None) -> tuple[str, ...]:
    if dataset_root is not None:
        return (dataset_root,)
    return KFA_RELEASED_MOTION_ROOT_CANDIDATES


def resolve_kfa_dataset_root(dataset_root: str | None = None, require_exists: bool = False) -> str:
    for root in _candidate_dataset_roots(dataset_root):
        if Path(root).exists():
            return str(Path(root).resolve())

    fallback = _candidate_dataset_roots(dataset_root)[0]
    if require_exists:
        raise FileNotFoundError(f"Could not find a KungFuAthlete dataset root from: {_candidate_dataset_roots(dataset_root)}")
    return str(Path(fallback).resolve())


def resolve_kfa_motion_path(clip_id: str, dataset_root: str | None = None, require_exists: bool = False) -> str:
    resolved_root = resolve_kfa_dataset_root(dataset_root=dataset_root, require_exists=require_exists)
    root = Path(resolved_root)

    for layout in KFA_RELEASED_MOTION_LAYOUTS:
        for suffix in KFA_RELEASED_MOTION_FILE_SUFFIXES:
            candidate = root / f"{clip_id}{suffix}" if layout == "flat" else root / clip_id / f"{clip_id}{suffix}"
            if candidate.exists():
                return str(candidate.resolve())

    fallback = root / f"{clip_id}{KFA_RELEASED_MOTION_FILE_SUFFIXES[0]}"
    if require_exists:
        raise FileNotFoundError(f"Could not find clip {clip_id} under {resolved_root}")
    return str(fallback.resolve())


def detect_kfa_motion_layout(motion_path: str, dataset_root: str) -> str:
    try:
        relative = Path(motion_path).resolve().relative_to(Path(dataset_root).resolve())
    except ValueError:
        return "external"
    return "nested" if len(relative.parts) >= 2 else "flat"


def get_kfa_released_motion_path(clip_id: str) -> str:
    return resolve_kfa_motion_path(clip_id, require_exists=False)


def resolve_clip_set(
    train_clip_id: str = KFA_TRAINING_EXAMPLE_CLIP_ID,
    eval_clip_id: str = KFA_PRIMARY_REGRESSION_CLIP_ID,
    optional_eval_clip_ids: tuple[str, ...] = KFA_OPTIONAL_REGRESSION_CLIP_IDS,
    dataset_root: str | None = None,
) -> dict[str, Any]:
    resolved_root = resolve_kfa_dataset_root(dataset_root=dataset_root, require_exists=False)
    optional_clips = []
    for clip_id in optional_eval_clip_ids:
        clip_path = resolve_kfa_motion_path(clip_id, dataset_root=resolved_root, require_exists=False)
        optional_clips.append(
            KfaParityClipResolution(
                clip_id=clip_id,
                path=clip_path,
                exists=Path(clip_path).exists(),
                dataset_root=resolved_root,
                layout=detect_kfa_motion_layout(clip_path, resolved_root),
            )
        )

    train_path = resolve_kfa_motion_path(train_clip_id, dataset_root=resolved_root, require_exists=False)
    eval_path = resolve_kfa_motion_path(eval_clip_id, dataset_root=resolved_root, require_exists=False)

    return {
        "dataset_root": resolved_root,
        "train": KfaParityClipResolution(
            clip_id=train_clip_id,
            path=train_path,
            exists=Path(train_path).exists(),
            dataset_root=resolved_root,
            layout=detect_kfa_motion_layout(train_path, resolved_root),
        ),
        "eval": KfaParityClipResolution(
            clip_id=eval_clip_id,
            path=eval_path,
            exists=Path(eval_path).exists(),
            dataset_root=resolved_root,
            layout=detect_kfa_motion_layout(eval_path, resolved_root),
        ),
        "optional_eval": optional_clips,
    }


def build_parity_artifact_dir(base_dir: str, preset_key: str, seed: int) -> Path:
    return Path(base_dir).resolve() / preset_key / f"seed_{seed}"


def build_training_command(
    preset_key: str,
    motion_file: str,
    seed: int,
    artifact_dir: Path,
    recovery_dataset_path: str,
    logger_preset: str,
    training_num_envs: int | None = None,
    num_learning_iterations: int | None = None,
) -> list[str]:
    preset_cfg = KFA_PARITY_PRESETS[preset_key]
    command = [
        "python",
        "src/holosoma/holosoma/train_agent.py",
        preset_cfg["experiment"],
        f"logger:{logger_preset}",
        f"--training.seed={seed}",
        f"--logger.base-dir={artifact_dir / 'logs'}",
        "--logger.video.enabled=False",
        f"--command.setup_terms.motion_command.params.motion_config.motion_file={motion_file}",
    ]
    if training_num_envs is not None:
        command.append(f"--training.num-envs={training_num_envs}")
    if num_learning_iterations is not None:
        command.append(f"--algo.config.num-learning-iterations={num_learning_iterations}")
    if preset_cfg["needs_recovery_dataset"]:
        command.extend(
            [
                "--command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.enabled=True",
                f"--command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.dataset_path={recovery_dataset_path}",
            ]
        )
    return command


def build_eval_command(
    checkpoint_path: str,
    motion_file: str,
    artifact_dir: Path,
    max_eval_steps: int,
) -> list[str]:
    return [
        "python",
        "src/holosoma/holosoma/eval_agent.py",
        f"--checkpoint={checkpoint_path}",
        f"--logger.base-dir={artifact_dir / 'logs'}",
        "--training.headless=True",
        f"--training.max-eval-steps={max_eval_steps}",
        f"--command.setup_terms.motion_command.params.motion_config.motion_file={motion_file}",
    ]


def build_mujoco_commands(onnx_path: str) -> dict[str, list[str]]:
    return {
        "simulator": [
            "python",
            "src/holosoma/holosoma/run_sim.py",
            "robot:g1-29dof-kfa",
        ],
        "policy": [
            "python",
            "src/holosoma_inference/holosoma_inference/run_policy.py",
            "inference:g1-29dof-wbt",
            f"--task.model-path={onnx_path}",
            "--task.no-use-joystick",
            "--task.use-sim-time",
            "--task.rl-rate=50",
            "--task.interface=lo",
        ],
    }
