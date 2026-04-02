from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro
import yaml

from holosoma.utils.kfa_parity import (
    KFA_OPTIONAL_REGRESSION_CLIP_IDS,
    KFA_PARITY_PRESETS,
    KFA_PARITY_SEEDS,
    KFA_PRIMARY_REGRESSION_CLIP_ID,
    KFA_TRAINING_EXAMPLE_CLIP_ID,
    build_eval_command,
    build_recovery_dataset_generation_command,
    build_recovery_dataset_validation_command,
    build_mujoco_commands,
    build_parity_artifact_dir,
    build_training_command,
    resolve_clip_set,
)


def extract_checkpoint_path(command_output: str) -> str | None:
    patterns = [
        r"Saving checkpoint to\s+(.+\.pt)",
        r"Saved model at:\s*(.+\.pt)",
        r"Checkpoint saved:\s*(.+\.pt)",
        r"Model saved to:\s*(.+\.pt)",
        r"checkpoint saved to\s*(.+\.pt)",
        r"(logs/[^/\s]+/[^/\s]+/model_\d+\.pt)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, command_output, re.IGNORECASE)
        if matches:
            return matches[-1]
    return None


@dataclass(frozen=True)
class KfaGroundParityConfig:
    output_dir: str = "artifacts/parity_ground"
    dataset_root: str | None = None
    recovery_dataset_path: str = "./artifacts/recovery_init/g1_ground_v1.npz"
    train_clip: str = KFA_TRAINING_EXAMPLE_CLIP_ID
    eval_clip: str = KFA_PRIMARY_REGRESSION_CLIP_ID
    optional_eval_clips: tuple[str, ...] = KFA_OPTIONAL_REGRESSION_CLIP_IDS
    preset_keys: tuple[str, ...] = tuple(KFA_PARITY_PRESETS.keys())
    seeds: tuple[int, ...] = KFA_PARITY_SEEDS
    logger_preset: str = "disabled"
    training_num_envs: int | None = None
    num_learning_iterations: int | None = None
    max_eval_steps: int = 300
    execute: bool = False


def _run_command(command: list[str], workdir: Path, log_path: Path) -> str:
    result = subprocess.run(command, cwd=workdir, capture_output=True, text=True, check=False)
    log_path.write_text(result.stdout + "\n\n[stderr]\n" + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    return result.stdout + result.stderr


def main() -> None:
    config = tyro.cli(KfaGroundParityConfig)
    repo_root = Path.cwd()
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_set = resolve_clip_set(
        train_clip_id=config.train_clip,
        eval_clip_id=config.eval_clip,
        optional_eval_clip_ids=config.optional_eval_clips,
        dataset_root=config.dataset_root,
    )

    suite_manifest = {
        "suite_name": "kfa_ground_parity",
        "paper_alignment_status": "paper_approximate",
        "paper_alignment_reason": "Low-kinetic Eq. 17 update rule remains unresolved from public paper text.",
        "readiness_gate": {
            "scope": "ground",
            "required_train_clip": config.train_clip,
            "required_eval_clip": config.eval_clip,
            "required_presets": ["stand", "recovery"],
            "requires_recovery_dataset_for_recovery": True,
            "requires_recovery_dataset_validation": True,
            "requires_same_sim_eval": True,
            "requires_onnx_export": True,
            "requires_mujoco_sim_to_sim_launch": True,
            "stage_order": [
                "stand_train",
                "stand_eval_onnx",
                "stand_mujoco_launch",
                "recovery_dataset_generation",
                "recovery_dataset_validation",
                "recovery_train",
                "recovery_eval_onnx",
                "recovery_mujoco_launch",
            ],
            "implementation_complete_when": "one_seed_end_to_end_passes",
        },
        "dataset_root": clip_set["dataset_root"],
        "train_clip": asdict(clip_set["train"]),
        "eval_clip": asdict(clip_set["eval"]),
        "optional_eval_clips": [asdict(item) for item in clip_set["optional_eval"]],
        "preset_keys": list(config.preset_keys),
        "seeds": list(config.seeds),
        "recovery_dataset_path": config.recovery_dataset_path,
        "logger_preset": config.logger_preset,
        "execute": config.execute,
        "runs": [],
    }

    for preset_key in config.preset_keys:
        for seed in config.seeds:
            artifact_dir = build_parity_artifact_dir(config.output_dir, preset_key, seed)
            artifact_dir.mkdir(parents=True, exist_ok=True)

            training_command = build_training_command(
                preset_key=preset_key,
                motion_file=clip_set["train"].path,
                seed=seed,
                artifact_dir=artifact_dir,
                recovery_dataset_path=config.recovery_dataset_path,
                logger_preset=config.logger_preset,
                training_num_envs=config.training_num_envs,
                num_learning_iterations=config.num_learning_iterations,
            )

            run_manifest = {
                "preset_key": preset_key,
                "experiment": KFA_PARITY_PRESETS[preset_key]["experiment"],
                "paper_alignment_status": "paper_approximate",
                "seed": seed,
                "artifact_dir": str(artifact_dir),
                "train_clip": asdict(clip_set["train"]),
                "eval_clip": asdict(clip_set["eval"]),
                "optional_eval_clips": [asdict(item) for item in clip_set["optional_eval"]],
                "recovery_dataset_path": config.recovery_dataset_path,
                "recovery_dataset_generation_command": build_recovery_dataset_generation_command(
                    recovery_dataset_path=config.recovery_dataset_path,
                    seed=seed,
                ),
                "recovery_dataset_validation_command": build_recovery_dataset_validation_command(
                    recovery_dataset_path=config.recovery_dataset_path,
                ),
                "training_command": training_command,
                "checkpoint_path": None,
                "eval_command": None,
                "onnx_path": None,
                "mujoco_commands": None,
                "status": "planned",
            }

            if config.execute:
                if preset_key == "recovery":
                    recovery_generation_command = run_manifest["recovery_dataset_generation_command"]
                    recovery_validation_command = run_manifest["recovery_dataset_validation_command"]
                    _run_command(
                        recovery_generation_command,
                        repo_root,
                        artifact_dir / "recovery_dataset_generation.log",
                    )
                    _run_command(
                        recovery_validation_command,
                        repo_root,
                        artifact_dir / "recovery_dataset_validation.log",
                    )

                training_output = _run_command(training_command, repo_root, artifact_dir / "train.log")
                checkpoint_path = extract_checkpoint_path(training_output)
                if checkpoint_path is None:
                    raise FileNotFoundError(f"Could not extract checkpoint path for preset {preset_key} seed {seed}")
                checkpoint_path = str(Path(checkpoint_path).resolve())
                eval_command = build_eval_command(
                    checkpoint_path=checkpoint_path,
                    motion_file=clip_set["eval"].path,
                    artifact_dir=artifact_dir,
                    max_eval_steps=config.max_eval_steps,
                )
                _run_command(eval_command, repo_root, artifact_dir / "eval.log")
                onnx_path = str(
                    (Path(checkpoint_path).parent / "exported" / Path(checkpoint_path).name.replace(".pt", ".onnx")).resolve()
                )
                run_manifest["checkpoint_path"] = checkpoint_path
                run_manifest["eval_command"] = eval_command
                run_manifest["onnx_path"] = onnx_path
                run_manifest["mujoco_commands"] = build_mujoco_commands(onnx_path)
                run_manifest["status"] = "completed"

            run_manifest_path = artifact_dir / "run_manifest.yaml"
            run_manifest_path.write_text(yaml.safe_dump(run_manifest, sort_keys=False))
            suite_manifest["runs"].append(run_manifest)

    (output_dir / "suite_manifest.yaml").write_text(yaml.safe_dump(suite_manifest, sort_keys=False))


if __name__ == "__main__":
    main()
