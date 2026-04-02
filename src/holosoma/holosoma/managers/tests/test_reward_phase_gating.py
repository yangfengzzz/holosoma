from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.managers.reward.manager import RewardManager


def constant_tracking_reward(env, value: float) -> torch.Tensor:
    return torch.full((env.num_envs,), float(value), device=env.device)


def constant_recovery_reward(env, value: float) -> torch.Tensor:
    return torch.full((env.num_envs,), float(value), device=env.device)


def constant_always_on_reward(env, value: float) -> torch.Tensor:
    return torch.full((env.num_envs,), float(value), device=env.device)


def test_reward_manager_gates_phase_tagged_rewards_by_recovery_state():
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        max_episode_length_s=1.0,
        command_manager=SimpleNamespace(
            get_state=lambda name: SimpleNamespace(recovery_active_mask=lambda: torch.tensor([False, True]))
        ),
    )
    cfg = RewardManagerCfg(
        terms={
            "tracking_only": RewardTermCfg(
                func="holosoma.managers.tests.test_reward_phase_gating:constant_tracking_reward",
                params={"value": 2.0},
                weight=1.0,
                tags=["r_mtr"],
            ),
            "recovery_only": RewardTermCfg(
                func="holosoma.managers.tests.test_reward_phase_gating:constant_recovery_reward",
                params={"value": 5.0},
                weight=1.0,
                tags=["r_rc"],
            ),
            "always_on": RewardTermCfg(
                func="holosoma.managers.tests.test_reward_phase_gating:constant_always_on_reward",
                params={"value": 1.0},
                weight=1.0,
            ),
        }
    )

    reward_manager = RewardManager(cfg, env, device="cpu")
    reward = reward_manager.compute(dt=1.0)

    assert reward.tolist() == pytest.approx([3.0, 6.0])
    assert reward_manager.episode_sums_raw["tracking_only"].tolist() == pytest.approx([2.0, 0.0])
    assert reward_manager.episode_sums_raw["recovery_only"].tolist() == pytest.approx([0.0, 5.0])
    assert reward_manager.episode_sums_raw["always_on"].tolist() == pytest.approx([1.0, 1.0])
