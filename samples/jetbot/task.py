"""Define a direction-following task for a two-wheeled JetBot.

The policy controls the left and right wheel speeds to move in a commanded
direction.
"""

from __future__ import annotations

import math
from typing import Tuple

import simulo

# The JetBot robot - a validated, version-pinned global-catalog asset.
jetbot = simulo.Asset.from_registry("simulo/robot/jetbot:v1")

# Advanced: pick a different Simulo runtime with
# App("name", runtime=simulo.Runtime.from_registry("simulo/gpu-rl:2026.06"));
# see the Runtimes docs.
app = simulo.App("jetbot")

# The ONE module-level heavy import - deferred under the runtime guard so discovery
# records it as a remote import instead of resolving it.
with app.runtime.imports():
    import torch  # noqa: F401  (resolved only in execution mode)


def _quat_to_forward(quat: torch.Tensor) -> torch.Tensor:
    """Rotate the unit X vector ``[1, 0, 0]`` by a per-env quaternion (wxyz).

    Plain typed module-level helper (not itself decorated) - ``torch.jit.script``
    recursively compiles it when it is called from the ``@app.runtime.torch_jit`` reward
    kernel below, and it also runs eagerly when called directly from
    ``JetbotTask.get_observations``.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    forward_x = 1.0 - 2.0 * (y * y + z * z)
    forward_y = 2.0 * (x * y + w * z)
    forward_z = 2.0 * (x * z - w * y)
    return torch.stack([forward_x, forward_y, forward_z], dim=-1)


@app.runtime.torch_jit
def _compute_rewards(
    rew_scale_alignment: float,
    rew_scale_velocity: float,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    commands: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled alignment + velocity reward kernel.

    ``@app.runtime.torch_jit`` is a no-op marker at submit and real ``torch.jit.script`` on
    execution, so this lives at module level and is torch-free to *define* during
    discovery (its body never runs at submit). ``_quat_to_forward`` is compiled
    transitively - TorchScript recursively scripts plain typed functions it calls.
    """
    forward = _quat_to_forward(root_quat)
    alignment = (forward * commands).sum(dim=-1)
    velocity_in_cmd_dir = (root_lin_vel[:, :2] * commands[:, :2]).sum(dim=-1)
    reward = rew_scale_alignment * alignment + rew_scale_velocity * velocity_in_cmd_dir
    # ``.view(-1)`` is harmless belt-and-braces here - nothing above ``.squeeze()``s,
    # so ``reward`` is already (num_envs,) even when num_envs == 1 - but it locks in
    # the per-env reward contract explicitly rather than relying on that being true.
    return reward.view(-1)


class JetbotTask(simulo.Task):
    """Drive a two-wheeled Jetbot in a commanded direction, as fast as possible.

    Observation (6-dim): forward direction unit vector (3) + commanded direction unit
    vector (3, XY-plane). Action (2-dim): left / right wheel angular velocity.
    """

    observation_dim = 6
    action_dim = 2

    episode_length_s = 5.0
    velocity_scale = 10.0  # [rad/s] wheel angular velocity scale

    rew_scale_alignment = 1.0
    rew_scale_velocity = 0.5

    # Framework-injected at runtime by ``simulo.core.Task`` / ``LearningEnv`` (declared
    # here only so the type checker sees the names the methods read; PEP 563 strings).
    device: str
    num_envs: int
    max_episode_length: int
    episode_length_buf: torch.Tensor

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
        self.robot = simulo.Robot(asset=jetbot, initial_pose=simulo.Pose.identity())
        scene.add(self.robot, at="/World/Robot")
        if self._with_camera:
            scene.add(
                simulo.Camera(
                    width=640,
                    height=480,
                    data_types=["rgb"],
                    update_period=1.0 / 30.0,
                    offset=simulo.SensorOffset.look_at(pos=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0)),
                ),
                at="/World/play_camera",
                per_environment=False,
            )

    def on_start(self, env: simulo.LearningEnv) -> None:
        left = self.robot.find_joints("left_wheel_joint")
        right = self.robot.find_joints("right_wheel_joint")
        self._wheel_joint_ids = left + right

        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._randomize_commands(torch.arange(self.num_envs, device=self.device))

    def get_observations(self) -> torch.Tensor:
        # pose is [x, y, z, qw, qx, qy, qz]; the quaternion is the last four columns.
        forward = _quat_to_forward(self.robot.state.pose[:, 3:7])
        return torch.cat([forward, self.commands], dim=-1)

    def get_rewards(self) -> torch.Tensor:
        return _compute_rewards(
            self.rew_scale_alignment,
            self.rew_scale_velocity,
            self.robot.state.pose[:, 3:7],
            self.robot.state.linear_velocity,
            self.commands,
        )

    def get_dones(self) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = {}
        return terminated, truncated

    def apply_actions(self, actions: torch.Tensor) -> None:
        # Trained policies can emit values outside the range. Clamp before scaling
        # so the robot is not over-driven.
        actions = torch.clamp(actions, -1.0, 1.0)
        scaled_velocities = actions * self.velocity_scale
        self.robot.set_joint_velocity_target(scaled_velocities, joint_ids=self._wheel_joint_ids)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        num_resets = len(env_ids)
        if num_resets == 0:
            return
        self.robot.reset(env_ids)
        self._randomize_commands(env_ids)

    def _randomize_commands(self, env_ids: torch.Tensor) -> None:
        """Generate new random XY-plane unit-vector direction commands."""
        n = len(env_ids)
        angles = torch.rand(n, device=self.device) * 2 * math.pi
        self.commands[env_ids, 0] = torch.cos(angles)
        self.commands[env_ids, 1] = torch.sin(angles)
        self.commands[env_ids, 2] = 0.0
