"""Define the Cartpole task shared by preview, training, evaluation, and playback.

The policy moves a cart left or right to keep its pole upright.
"""

from __future__ import annotations

import math
from typing import Tuple

import simulo

# The cartpole robot - a validated, version-pinned global-catalog asset.
# Capturing it at module level lets submit resolve this exact version before the
# job runs; execution mounts it and the Task below consumes the same handle.
cartpole = simulo.Asset.from_registry("simulo/robot/cartpole:v1")

# Advanced: pick a different Simulo runtime with
# App("name", runtime=simulo.Runtime.from_registry("simulo/gpu-rl:2026.06"));
# see the Runtimes docs.
app = simulo.App("cartpole")

# The ONE module-level heavy import - deferred under the runtime guard so discovery
# records it as a remote import instead of resolving it.
with app.runtime.imports():
    import torch  # noqa: F401  (resolved only in execution mode)


@app.runtime.torch_jit
def _compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled reward kernel (the classic cartpole balance reward).

    ``@app.runtime.torch_jit`` is a no-op marker at submit (no ``torch`` locally) and
    real ``torch.jit.script`` during execution, so this lives at module level and is
    still torch-free to *define* during discovery (its body never runs at submit).
    """
    pole_pos = pole_pos.squeeze()
    pole_vel = pole_vel.squeeze()
    cart_pos = cart_pos.squeeze()
    cart_vel = cart_vel.squeeze()
    reset_terminated = reset_terminated.squeeze()

    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.square(pole_pos)
    rew_cart_vel = rew_scale_cart_vel * torch.abs(cart_vel)
    rew_pole_vel = rew_scale_pole_vel * torch.abs(pole_vel)

    reward: torch.Tensor = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    # Keep the (num_envs,) per-env reward contract even when num_envs == 1: the
    # squeezes above collapse a single-env batch to a 0-d scalar, which breaks
    # per-env consumers (e.g. the MCAP recorder's per-env /reward indexing).
    return reward.view(-1)


class CartpoleTask(simulo.Task):
    """Balance a pole on a cart by applying horizontal forces to the cart.

    Observation (4-dim): pole angle, pole angular velocity, cart position, cart
    velocity. Action (1-dim): scaled horizontal force on the cart.

    Defined at module level: ``simulo.Task`` is a torch-free contract stand-in at
    submit and the real ``simulo.core.Task`` during execution, so the same class
    authors lean and trains heavy.
    """

    observation_dim = 4
    action_dim = 1

    episode_length_s = 5.0
    action_scale = 100.0  # [N]

    max_cart_pos = 3.0  # [m]
    initial_pole_angle_range = (-0.25, 0.25)  # fraction of pi [rad]

    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005

    # Framework-injected at runtime by ``simulo.core.Task`` / ``LearningEnv``
    # (declared here only so the type checker sees the names the methods read; the
    # annotations are PEP 563 strings and never shadow the inherited values).
    device: str
    max_episode_length: int
    episode_length_buf: torch.Tensor
    reset_terminated: torch.Tensor

    def __init__(self, with_camera: bool = False):
        super().__init__()
        self._with_camera = with_camera

    def build(self, scene: simulo.Scene) -> None:
        scene.add(simulo.Terrain.plane(name="ground"), at="/", per_environment=False)
        scene.add(
            simulo.Light.dome(name="light", intensity=2000.0, color=(0.75, 0.75, 0.75)),
            at="/",
            per_environment=False,
        )
        self.robot = simulo.Robot(asset=cartpole, initial_pose=simulo.Pose.identity())
        scene.add(self.robot, at="/World/Robot")

        # Playback adds one world-frame camera. Training keeps the default off,
        # so its scene and render cost remain unchanged.
        if self._with_camera:
            scene.add(
                simulo.Camera(
                    width=640,
                    height=480,
                    data_types=["rgb"],
                    update_period=1.0 / 30.0,
                    offset=simulo.SensorOffset.look_at(pos=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0)),
                ),
                at="/World/overhead_cam",
                per_environment=False,
            )

    def on_start(self, env: simulo.LearningEnv) -> None:
        self._cart_dof_idx = self.robot.find_joints("slider_to_cart")
        self._pole_dof_idx = self.robot.find_joints("cart_to_pole")

    def get_observations(self) -> torch.Tensor:
        pole_idx = self._pole_dof_idx[0]
        cart_idx = self._cart_dof_idx[0]
        joint_pos = self.robot.state.joint_positions
        joint_vel = self.robot.state.joint_velocities
        pole_pos = joint_pos[:, pole_idx].view(-1, 1)
        pole_vel = joint_vel[:, pole_idx].view(-1, 1)
        cart_pos = joint_pos[:, cart_idx].view(-1, 1)
        cart_vel = joint_vel[:, cart_idx].view(-1, 1)
        return torch.cat((pole_pos, pole_vel, cart_pos, cart_vel), dim=-1)

    def get_rewards(self) -> torch.Tensor:
        joint_pos = self.robot.state.joint_positions
        joint_vel = self.robot.state.joint_velocities
        return _compute_rewards(
            self.rew_scale_alive,
            self.rew_scale_terminated,
            self.rew_scale_pole_pos,
            self.rew_scale_cart_vel,
            self.rew_scale_pole_vel,
            joint_pos[:, self._pole_dof_idx[0]],
            joint_vel[:, self._pole_dof_idx[0]],
            joint_pos[:, self._cart_dof_idx[0]],
            joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )

    def get_dones(self) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        pole_idx = self._pole_dof_idx[0]
        cart_idx = self._cart_dof_idx[0]
        joint_pos = self.robot.state.joint_positions
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        cart_out = torch.abs(joint_pos[:, cart_idx]) > self.max_cart_pos
        pole_fallen = torch.abs(joint_pos[:, pole_idx]) > math.pi / 2
        terminated = {"cart left the track": cart_out, "pole fell": pole_fallen}
        return terminated, truncated

    def apply_actions(self, actions: torch.Tensor) -> None:
        # Trained policies can emit values outside the range. Clamp before scaling
        # so the robot is not over-driven.
        actions = torch.clamp(actions, -1.0, 1.0)
        self.robot.set_joint_effort_target(
            self.action_scale * actions, joint_ids=self._cart_dof_idx
        )

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        num_resets = len(env_ids)
        if num_resets == 0:
            return
        self.robot.reset(env_ids)
        pole_idx = self._pole_dof_idx[0]
        # Start from the default joint positions for the selected environments.
        joint_pos = self.robot.default_joint_positions[env_ids]
        random_angles = torch.empty(num_resets, device=self.device).uniform_(
            self.initial_pole_angle_range[0] * math.pi,
            self.initial_pole_angle_range[1] * math.pi,
        )
        joint_pos[:, pole_idx] += random_angles
        # Apply the randomized starting positions and default velocities together.
        joint_vel = self.robot.default_joint_velocities[env_ids]
        self.robot.set_joint_state(joint_pos, velocities=joint_vel, env_ids=env_ids)
