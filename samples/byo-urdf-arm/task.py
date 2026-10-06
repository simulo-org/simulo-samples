"""Define a trainable reaching task for an organization-published URDF arm.

The task observes three joint positions, velocities, target positions, and commanded
positions. Each of the three actions names the angle one joint should hold: a value
from -1 to 1 becomes a joint target from -target_amplitude to +target_amplitude.
"""

from __future__ import annotations

from typing import Tuple

import simulo

# Your URDF-derived robot, published to your OWN org catalog. No publisher segment =>
# resolved against the caller's own org, not "simulo/...".
byo_arm = simulo.Asset.from_registry("robot/byo-urdf-arm:v1")

# Advanced: pick a different Simulo runtime with
# App("name", runtime=simulo.Runtime.from_registry("simulo/gpu-rl:2026.06")).
app = simulo.App("byo-urdf-arm")

# The ONE module-level heavy import, deferred under the runtime guard so discovery
# records it as a remote import instead of resolving it.
with app.runtime.imports():
    import torch  # noqa: F401  (resolved only in execution mode)


@app.runtime.torch_jit
def _compute_rewards(
    rew_scale_distance: float,
    rew_scale_velocity: float,
    rew_scale_settling: float,
    position_tolerance: float,
    settling_tolerance: float,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    target_pos: torch.Tensor,
) -> torch.Tensor:
    """Compute the arm reward from target accuracy, velocity, and settling."""
    pos_error = joint_pos - target_pos
    dist_sq = torch.sum(torch.square(pos_error), dim=-1)
    vel_sq = torch.sum(torch.square(joint_vel), dim=-1)
    within_position_tolerance = torch.amax(torch.abs(pos_error), dim=-1) < position_tolerance
    settling_quality = torch.exp(-vel_sq / (settling_tolerance * settling_tolerance))
    reward = (
        rew_scale_distance * dist_sq
        + rew_scale_velocity * vel_sq
        + rew_scale_settling * within_position_tolerance * settling_quality
    )
    # Keep the (num_envs,) per-env reward contract even when num_envs == 1.
    return reward.view(-1)


class ByoArmTask(simulo.Task):
    """Move a three-joint arm to a randomized joint-space target.

    Observation (12-dim): joint positions, velocities, target positions, and commanded
    positions, each ordered ``shoulder_pan``, ``shoulder_lift``, ``elbow``. Action
    (3-dim): the angle each of those joints should hold, from -1 to 1, scaled by
    ``target_amplitude``.
    """

    observation_dim = 12
    action_dim = 3

    episode_length_s = 5.0
    actuator_stiffness = 40.0  # [N m/rad]
    # The 5.0 value came from an empirical sweep of 1.1, 4.0, and 5.0 damping
    # with exact target commands. That sweep does not measure policy accuracy.
    # Catalog robots use these task drive gains where the execution environment applies a
    # task's configured joint stiffness and damping; otherwise they use asset damping.
    actuator_damping = 5.0  # [N m s/rad]

    #: Per-episode target sampled uniformly in [-target_amplitude, +target_amplitude]
    #: for each of the three joints, comfortably inside all three URDF joint limits
    #: (the narrowest is shoulder_lift at +-1.9 rad).
    target_amplitude = 1.0  # [rad]

    rew_scale_distance = -1.0
    rew_scale_velocity = -0.01
    rew_scale_settling = 0.1
    position_tolerance = 0.1  # [rad], matches the unchanged success rule
    settling_tolerance = 0.1  # [rad/s], matches the unchanged success rule

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
        scene.add(simulo.Terrain.plane(name="ground"), at="/World/ground", per_environment=False)
        scene.add(
            simulo.Light.dome(name="light", intensity=2000.0, color=(0.75, 0.75, 0.75)),
            at="/World/Light",
            per_environment=False,
        )
        # Your converted URDF arm (fixed base, bolted at the origin).
        self.robot = simulo.Robot(
            asset=byo_arm,
            initial_pose=simulo.Pose.identity(),
            actuator_gains={
                "all_joints": simulo.ActuatorGainsConfig(
                    stiffness=self.actuator_stiffness,
                    damping=self.actuator_damping,
                )
            },
        )
        scene.add(self.robot, at="/World/Arm")
        if self._with_camera:
            scene.add(
                simulo.Camera(
                    width=640,
                    height=480,
                    data_types=["rgb"],
                    update_period=1.0 / 30.0,
                    offset=simulo.SensorOffset.look_at(pos=(2.0, 2.0, 1.5), target=(0.0, 0.0, 0.5)),
                ),
                at="/World/play_camera",
                per_environment=False,
            )

    def on_start(self, env: simulo.LearningEnv) -> None:
        pan = self.robot.find_joints("shoulder_pan")
        lift = self.robot.find_joints("shoulder_lift")
        elbow = self.robot.find_joints("elbow")
        self._joint_ids = pan + lift + elbow

        self.target = torch.zeros((self.num_envs, 3), device=self.device)
        self._position_targets = self.robot.state.joint_positions[:, self._joint_ids].clone()
        self._randomize_target(torch.arange(self.num_envs, device=self.device))

    def get_observations(self) -> torch.Tensor:
        joint_pos = self.robot.state.joint_positions[:, self._joint_ids]
        joint_vel = self.robot.state.joint_velocities[:, self._joint_ids]
        return torch.cat([joint_pos, joint_vel, self.target, self._position_targets], dim=-1)

    def get_rewards(self) -> torch.Tensor:
        return _compute_rewards(
            self.rew_scale_distance,
            self.rew_scale_velocity,
            self.rew_scale_settling,
            self.position_tolerance,
            self.settling_tolerance,
            self.robot.state.joint_positions[:, self._joint_ids],
            self.robot.state.joint_velocities[:, self._joint_ids],
            self.target,
        )

    def get_dones(self) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = {}
        return terminated, truncated

    def apply_actions(self, actions: torch.Tensor) -> None:
        # Each action is the joint angle to hold, scaled so 1.0 means
        # +target_amplitude. The policy reaches a target by naming it, instead of
        # adding small steps that must stop at exactly the right place.
        self._position_targets = self.target_amplitude * torch.clamp(actions, -1.0, 1.0)
        self.robot.set_joint_position_target(self._position_targets, joint_ids=self._joint_ids)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        num_resets = len(env_ids)
        if num_resets == 0:
            return
        self.robot.reset(env_ids)
        self._position_targets[env_ids] = self.robot.default_joint_positions[env_ids][
            :, self._joint_ids
        ]
        self._randomize_target(env_ids)

    def _randomize_target(self, env_ids: torch.Tensor) -> None:
        """Sample each joint's target uniformly from -target_amplitude to +target_amplitude."""
        n = len(env_ids)
        self.target[env_ids] = (
            torch.rand((n, 3), device=self.device) * 2.0 - 1.0
        ) * self.target_amplitude
