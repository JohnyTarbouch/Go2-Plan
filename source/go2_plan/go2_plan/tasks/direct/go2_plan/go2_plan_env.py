# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .go2_plan_env_cfg import Go2PlanEnvCfg


class Go2PlanEnv(DirectRLEnv):
    cfg: Go2PlanEnvCfg

    def __init__(self, cfg: Go2PlanEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        # ground
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # optional: add ambient light
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

        # replicate environments (even if just one)
        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        pass

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        return {"policy": torch.zeros((self.num_envs, 1), device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), time_out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)