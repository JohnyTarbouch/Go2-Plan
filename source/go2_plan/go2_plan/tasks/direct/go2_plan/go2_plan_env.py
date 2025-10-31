# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .go2_plan_env_cfg import Go2PlanEnvCfg


class Go2PlanEnv(DirectRLEnv):
    cfg: Go2PlanEnvCfg

    def __init__(self, cfg: Go2PlanEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        # Spawn ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        walls_usd_path = "C:\\Users\\johnn\\Desktop\\IsaacLab\\go2_plan\\source\\go2_plan\\go2_plan\\tasks\\direct\\go2_plan\\assets\\walls.usd"
        walls_cfg = sim_utils.UsdFileCfg(
            usd_path=walls_usd_path,
            visible=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        )
        walls_cfg.func(prim_path="/World/walls", cfg=walls_cfg)
    
        # Spawn Unitree Go2 robot
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=4000.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)

        # Clone environments (even if single)
        self.scene.clone_environments(copy_from_source=False)
        

    def _pre_physics_step(self, actions: torch.Tensor):
        """Prepare actions before physics step."""
        self.actions = actions.clone()
        self._apply_action()

    def _get_observations(self):
        """Return basic observations (root position for now)."""
        obs = self.robot.data.root_pos_w.clone()
        return {"policy": obs}

    def _apply_action(self):
        """Apply zero effort (no control yet)."""
        self.robot.set_joint_effort_target(torch.zeros_like(self.robot.data.joint_pos))

    def _get_rewards(self):
        """No rewards yet."""
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        """Timeout handling."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), time_out

    def _reset_idx(self, env_ids):
        """Reset environment indices."""
        super()._reset_idx(env_ids)
        
    
