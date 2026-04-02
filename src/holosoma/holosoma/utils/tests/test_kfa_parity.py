from __future__ import annotations

from pathlib import Path

import yaml

from holosoma.utils.kfa_parity import (
    KFA_OPTIONAL_REGRESSION_CLIP_IDS,
    build_eval_command,
    build_mujoco_commands,
    build_parity_artifact_dir,
    build_training_command,
    resolve_clip_set,
    resolve_kfa_dataset_root,
    resolve_kfa_motion_path,
)


def test_resolve_kfa_motion_path_prefers_existing_npz(tmp_path):
    dataset_root = tmp_path / "org_smoothed_mj"
    dataset_root.mkdir()
    (dataset_root / "1317.npz").write_bytes(b"")

    resolved = resolve_kfa_motion_path("1317", dataset_root=str(dataset_root), require_exists=True)

    assert resolved.endswith("/1317.npz")


def test_resolve_clip_set_marks_missing_optional_clip(tmp_path):
    dataset_root = tmp_path / "org_smoothed_mj"
    dataset_root.mkdir()
    for clip_id in ("1317", "1307", "969"):
        (dataset_root / f"{clip_id}.npz").write_bytes(b"")

    clip_set = resolve_clip_set(dataset_root=str(dataset_root), optional_eval_clip_ids=KFA_OPTIONAL_REGRESSION_CLIP_IDS)

    assert clip_set["train"].exists is True
    assert clip_set["eval"].exists is True
    optional_status = {item.clip_id: item.exists for item in clip_set["optional_eval"]}
    assert optional_status == {"969": True, "203": False}


def test_resolve_kfa_motion_path_requires_real_local_clip_id(tmp_path):
    dataset_root = tmp_path / "org_smoothed_mj"
    dataset_root.mkdir()
    (dataset_root / "203.npz").write_bytes(b"")

    resolved = resolve_kfa_motion_path("203", dataset_root=str(dataset_root), require_exists=True)

    assert resolved.endswith("/203.npz")


def test_build_parity_commands_include_expected_overrides(tmp_path):
    artifact_dir = build_parity_artifact_dir(str(tmp_path), "recovery", 2)
    training_command = build_training_command(
        preset_key="recovery",
        motion_file="/tmp/1317.npz",
        seed=2,
        artifact_dir=artifact_dir,
        recovery_dataset_path="/tmp/recovery.npz",
        logger_preset="disabled",
        training_num_envs=64,
        num_learning_iterations=5,
    )
    eval_command = build_eval_command(
        checkpoint_path="/tmp/model_1.pt",
        motion_file="/tmp/1307.npz",
        artifact_dir=artifact_dir,
        max_eval_steps=12,
    )

    assert "--training.seed=2" in training_command
    assert "--training.num-envs=64" in training_command
    assert "--algo.config.num-learning-iterations=5" in training_command
    assert "--command.setup_terms.motion_command.params.motion_config.recovery_init_dataset.enabled=True" in training_command
    assert "--training.max-eval-steps=12" in eval_command
    mujoco_commands = build_mujoco_commands("/tmp/model.onnx")
    assert mujoco_commands["simulator"][-1] == "robot:g1-29dof-kfa"
    assert "inference:g1-29dof-wbt" in mujoco_commands["policy"]


def test_parity_harness_manifest_fields_are_yaml_safe(tmp_path):
    dataset_root = tmp_path / "org_smoothed_mj"
    dataset_root.mkdir()
    for clip_id in ("1317", "1307", "969"):
        (dataset_root / f"{clip_id}.npz").write_bytes(b"")

    manifest = {
        "dataset_root": resolve_kfa_dataset_root(str(dataset_root), require_exists=True),
        "train_clip": {
            "clip_id": "1317",
            "path": resolve_kfa_motion_path("1317", str(dataset_root), require_exists=True),
        },
        "eval_clip": {
            "clip_id": "1307",
            "path": resolve_kfa_motion_path("1307", str(dataset_root), require_exists=True),
        },
    }
    dumped = yaml.safe_dump(manifest)
    loaded = yaml.safe_load(dumped)

    assert Path(loaded["dataset_root"]) == dataset_root.resolve()
    assert loaded["train_clip"]["path"].endswith("/1317.npz")
