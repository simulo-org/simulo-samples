"""Define a walking task for a 21-joint humanoid, following the standard humanoid recipe.

The task uses the recipe's 75-value observation, dense walking reward, per-joint effort
scales, 15-second episode, and fall termination. Joint positions are measured from their
defaults instead of being scaled by joint limits, because the Simulo robot API does not
expose joint limits.
"""

from __future__ import annotations

from typing import List, Tuple

import simulo

# The humanoid robot: a validated, version-pinned global-catalog asset.
humanoid = simulo.Asset.from_registry("simulo/robot/humanoid:v1")

# Advanced: pick a different Simulo runtime with
# App("name", runtime=simulo.Runtime.from_registry("simulo/gpu-rl:2026.06"));
# see the Runtimes docs.
app = simulo.App("humanoid")

# The ONE module-level heavy import, deferred under the runtime guard so discovery
# records it as a remote import instead of resolving it.
with app.runtime.imports():
    import torch  # noqa: F401  (resolved only in execution mode)


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate one per-environment vector by one wxyz quaternion."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return torch.stack([rx, ry, rz], dim=-1)


def _normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _heading_features(
    root_quat: torch.Tensor,
    to_target: torch.Tensor,
    up_vec: torch.Tensor,
    heading_vec: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return yaw, roll, target angle, up projection, and heading projection."""
    heading_world = _quat_rotate(root_quat, heading_vec)
    up_world = _quat_rotate(root_quat, up_vec)
    target_angle = torch.atan2(to_target[:, 2], to_target[:, 0])
    flat_target = to_target.clone()
    flat_target[:, 2] = 0.0
    target_direction = flat_target / torch.clamp(
        torch.linalg.vector_norm(flat_target, dim=-1, keepdim=True), min=1e-6
    )

    w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (
        _normalize_angle(yaw),
        _normalize_angle(roll),
        _normalize_angle(target_angle - yaw),
        up_world[:, 2],
        torch.sum(heading_world * target_direction, dim=-1),
    )


@app.runtime.torch_jit
def _compute_rewards(
    up_weight: float,
    heading_weight: float,
    actions_cost_scale: float,
    energy_cost_scale: float,
    dof_vel_scale: float,
    alive_reward_scale: float,
    death_cost: float,
    progress: torch.Tensor,
    heading_projection: torch.Tensor,
    up_projection: torch.Tensor,
    joint_vel: torch.Tensor,
    prev_actions: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    """The recipe's dense walking reward, without its joint-limit cost."""
    heading_reward = torch.where(
        heading_projection > 0.8,
        torch.full_like(heading_projection, heading_weight),
        heading_weight * heading_projection / 0.8,
    )
    up_reward = torch.where(
        up_projection > 0.93,
        torch.full_like(up_projection, up_weight),
        torch.zeros_like(up_projection),
    )
    energy = torch.sum(torch.abs(prev_actions * joint_vel * dof_vel_scale), dim=-1)
    action_norm = torch.sum(torch.square(prev_actions), dim=-1)
    reward = (
        progress
        + alive_reward_scale
        + up_reward
        + heading_reward
        - actions_cost_scale * action_norm
        - energy_cost_scale * energy
    )
    return torch.where(reset_terminated, torch.full_like(reward, death_cost), reward).view(-1)


class HumanoidTask(simulo.Task):
    """Walk a 21-DOF bipedal humanoid forward while staying upright.

    Observation (75-dim): torso height, base-frame linear and angular velocities,
    yaw, roll, target angle, up and heading projections, joint positions and velocities,
    and previous actions. Action (21-dim): per-joint efforts scaled by ``joint_gears``.
    """

    observation_dim = 75
    action_dim = 21

    episode_length_s = 15.0
    action_scale = 1.0

    # Joint gear (torque scale) ratios: the classic humanoid locomotion values, in
    # USD joint-declaration order (lower_waist x2, upper_arms x4, pelvis, lower_arms
    # x2, thighs x6, knees x2, feet x4).
    joint_gears: List[float] = [
        67.5,
        67.5,
        67.5,
        67.5,
        67.5,
        67.5,
        67.5,
        45.0,
        45.0,
        45.0,
        135.0,
        45.0,
        45.0,
        135.0,
        45.0,
        90.0,
        90.0,
        22.5,
        22.5,
        22.5,
        22.5,
    ]

    heading_weight = 0.5
    up_weight = 0.1
    energy_cost_scale = 0.05
    actions_cost_scale = 0.01
    alive_reward_scale = 2.0
    death_cost = -1.0

    dof_vel_scale = 0.1
    angular_velocity_scale = 0.25

    termination_height = 0.8  # [m]: torso height below this counts as "fallen"

    # Framework-injected at runtime by ``simulo.core.Task`` / ``LearningEnv`` (declared
    # here only so the type checker sees the names the methods read; PEP 563 strings).
    device: str
    num_envs: int
    max_episode_length: int
    episode_length_buf: torch.Tensor
    reset_terminated: torch.Tensor

    def __init__(self, with_camera: bool = False):
        super().__init__()
        self._with_camera = with_camera

    def build(self, scene: simulo.Scene) -> None:
        scene.add(simulo.Terrain.plane(name="ground"), at="/World", per_environment=False)
        scene.add(
            simulo.Light.dome(name="light", intensity=2000.0, color=(0.75, 0.75, 0.75)),
            at="/World",
            per_environment=False,
        )
        self.robot = simulo.Robot(asset=humanoid, initial_pose=simulo.Pose.from_xyz(0.0, 0.0, 1.34))
        scene.add(self.robot, at="/World/Robot")
        if self._with_camera:
            scene.add(
                simulo.Camera(
                    width=640,
                    height=480,
                    data_types=["rgb"],
                    update_period=1.0 / 30.0,
                    offset=simulo.SensorOffset.look_at(pos=(8.0, 6.0, 4.0), target=(2.0, 0.0, 1.0)),
                ),
                at="/World/play_camera",
                per_environment=False,
            )

    def on_start(self, env: simulo.LearningEnv) -> None:
        self._joint_gears = torch.tensor(self.joint_gears, device=self.device, dtype=torch.float32)
        self._prev_actions = torch.zeros(
            (self.num_envs, self.action_dim), device=self.device, dtype=torch.float32
        )

        self._up_vec = torch.zeros((self.num_envs, 3), device=self.device)
        self._up_vec[:, 2] = 1.0
        self._heading_vec = torch.zeros((self.num_envs, 3), device=self.device)
        self._heading_vec[:, 0] = 1.0
        self._targets = env.scene.env_origins.clone()
        self._targets[:, 0] += 1000.0
        self._potential = self._potential_at(self.robot.state.pose[:, :3])

    def _potential_at(self, position: torch.Tensor) -> torch.Tensor:
        return (
            -torch.linalg.vector_norm((self._targets - position)[:, :2], dim=-1) / self.physics_dt
        )

    def get_observations(self) -> torch.Tensor:
        state = self.robot.state
        yaw, roll, target_angle, up_projection, heading_projection = _heading_features(
            state.pose[:, 3:7],
            self._targets - state.pose[:, :3],
            self._up_vec,
            self._heading_vec,
        )
        joint_pos_rel = state.joint_positions - self.robot.default_joint_positions

        return torch.cat(
            [
                state.pose[:, 2:3],
                state.linear_velocity_in_base_frame,
                state.angular_velocity_in_base_frame * self.angular_velocity_scale,
                yaw.unsqueeze(-1),
                roll.unsqueeze(-1),
                target_angle.unsqueeze(-1),
                up_projection.unsqueeze(-1),
                heading_projection.unsqueeze(-1),
                joint_pos_rel,
                state.joint_velocities * self.dof_vel_scale,
                self._prev_actions,
            ],
            dim=-1,
        )

    def get_rewards(self) -> torch.Tensor:
        state = self.robot.state
        potential = self._potential_at(state.pose[:, :3])
        _, _, _, up_projection, heading_projection = _heading_features(
            state.pose[:, 3:7],
            self._targets - state.pose[:, :3],
            self._up_vec,
            self._heading_vec,
        )
        rewards = _compute_rewards(
            self.up_weight,
            self.heading_weight,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.dof_vel_scale,
            self.alive_reward_scale,
            self.death_cost,
            potential - self._potential,
            heading_projection,
            up_projection,
            state.joint_velocities,
            self._prev_actions,
            self.reset_terminated,
        )
        self._potential.copy_(potential)
        return rewards

    def get_dones(self) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        # robot.state.pose is [x, y, z, qw, qx, qy, qz]; z (height) is column 2.
        terminated = {"robot fell over": self.robot.state.pose[:, 2] < self.termination_height}
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def apply_actions(self, actions: torch.Tensor) -> None:
        # Trained policies can emit values outside the range. Clamp before scaling
        # so the robot is not over-driven.
        actions = torch.clamp(actions, -1.0, 1.0)
        self._prev_actions = actions.clone()
        scaled_efforts = self.action_scale * actions * self._joint_gears
        self.robot.set_joint_effort_target(scaled_efforts)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        num_resets = len(env_ids)
        if num_resets == 0:
            return
        self.robot.reset(env_ids)
        joint_pos = self.robot.default_joint_positions[env_ids]
        joint_vel = self.robot.default_joint_velocities[env_ids]
        self.robot.set_joint_state(joint_pos, joint_vel, env_ids=env_ids)
        self._prev_actions[env_ids] = 0.0
        self._potential[env_ids] = -1000.0 / self.physics_dt
