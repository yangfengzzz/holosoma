from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.absolute()


def test_kfa_parity_harness_dry_run_writes_suite_manifest(tmp_path):
    dataset_root = tmp_path / "org_smoothed_mj"
    dataset_root.mkdir()
    for clip_id in ("1317", "1307", "969"):
        (dataset_root / f"{clip_id}.npz").write_bytes(b"")

    output_dir = tmp_path / "parity_ground"

    subprocess.check_call(
        [
            "python",
            f"{REPO_ROOT}/src/holosoma/holosoma/run_kfa_ground_parity.py",
            f"--dataset-root={dataset_root}",
            f"--output-dir={output_dir}",
            "--preset-keys",
            "stand",
            "recovery",
            "--seeds",
            "1",
            "2",
        ]
    )

    suite_manifest = output_dir / "suite_manifest.yaml"
    assert suite_manifest.exists()
    assert (output_dir / "stand" / "seed_1" / "run_manifest.yaml").exists()
    assert (output_dir / "recovery" / "seed_2" / "run_manifest.yaml").exists()
