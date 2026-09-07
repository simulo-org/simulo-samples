"""JetBot: train a two-wheeled robot to drive in a commanded direction.

A small differential-drive robot learns to turn toward a randomly chosen heading and
drive along it as fast as it can, by controlling the angular velocity of its left and
right wheels. Two action values map directly onto the two wheels, so the control
problem and the actuator layout match one to one.

The robot model is not bundled with this sample. The version-pinned catalog reference
is fetched when it is not already cached, so any job may have a quiet asset-preparation
phase before training output appears.

Task summary
------------
* Observation (6 values): the robot's forward direction as a unit vector (3) plus the
  commanded direction as a unit vector (3, in the XY plane, re-randomised each episode).
* Action (2 values): left and right wheel angular velocity, scaled by
  ``velocity_scale``.
* Reward: alignment (the dot product of the forward and commanded directions) plus
  forward speed in the commanded direction.
* Termination: none. The robot cannot really fail this task, so every episode
  truncates at the time limit.

What to expect from the defaults
--------------------------------
An earlier single-GPU run at the defaults reported a best reward near 190 at iteration
50 and near 343 from iteration 150 onward. ``best_reward`` is a mean completed-episode
return across 300 control steps, not a per-step reward. The alignment term can
contribute at most 300 per episode; the velocity term supplies the remainder. Read
those figures as the shape of the curve rather than as numbers to hit; training is not
bit-for-bit reproducible.

Run it
------
Sign in once with ``simulo login``, then::

    simulo run samples/jetbot/app.py --num-envs 512 --max-iterations 700

Use ``--num-envs 64 --max-iterations 2`` for a quick check that the job launches.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import simulo

# The JetBot robot: a version-pinned catalog reference.
jetbot = simulo.Asset.from_registry("simulo/robot/jetbot:v2")

# A named, durable, writable volume for the trained checkpoint. This line only
# declares metadata; nothing is created at packaging time. The job reads the
# volume's real directory through ``vol.path``, which resolves only inside a
# running job.
vol = simulo.Volume.from_name("jetbot-checkpoints", create_if_missing=True)

# Advanced: pick a different Simulo runtime with
# App("name", runtime=simulo.Runtime.from_registry("simulo/gpu-rl:2026.06"));
# see https://docs.simulo.ai/concepts/runtimes/.
app = simulo.App("jetbot", mounts={"/out": vol})

# The one heavy import, deferred: on your machine this block records the import
# instead of resolving it; in the cloud it is a plain import.
with app.runtime.imports():
    import torch  # noqa: F401  (resolved only when the job runs in the cloud)


def _quat_to_forward(quat: torch.Tensor) -> torch.Tensor:
    """Rotate the unit X vector ``[1, 0, 0]`` by a per-env quaternion (wxyz).

    A plain typed module-level helper, not itself decorated. ``torch.jit.script``
    compiles it transitively when the ``@app.runtime.torch_jit`` reward kernel below
    calls it, and it also runs eagerly when ``JetbotTask.get_observations`` calls it.
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

    ``@app.runtime.torch_jit`` is a marker on your machine and ``torch.jit.script`` in
    the cloud, so this can live at module level and its body never runs at submit.
    ``_quat_to_forward`` is compiled transitively: TorchScript scripts the plain typed
    functions a kernel calls.
    """
    forward = _quat_to_forward(root_quat)
    alignment = (forward * commands).sum(dim=-1)
    velocity_in_cmd_dir = (root_lin_vel[:, :2] * commands[:, :2]).sum(dim=-1)
    reward = rew_scale_alignment * alignment + rew_scale_velocity * velocity_in_cmd_dir
    # ``.view(-1)`` is belt and braces here: nothing above squeezes, so ``reward``
    # is already (num_envs,) even when num_envs == 1, but it locks in the per-env
    # reward contract explicitly rather than relying on that being true.
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

    # Set by the training base class when the job runs. Declared here only so a
    # type checker sees the names the methods read; the annotations are strings.
    device: str
    num_envs: int
    max_episode_length: int
    episode_length_buf: torch.Tensor

    def build(self, scene: simulo.Scene) -> None:
        scene.add(simulo.Terrain.plane(name="ground"), at="/World", per_environment=False)
        scene.add(
            simulo.Light.dome(name="light", intensity=2000.0, color=(0.75, 0.75, 0.75)),
            at="/World",
            per_environment=False,
        )
        self.robot = simulo.Robot(asset=jetbot, initial_pose=simulo.Pose.identity())
        scene.add(self.robot, at="/World/Robot")

    def on_start(self, env: simulo.LearningEnv) -> None:
        left = self.robot.find_joints("left_wheel_joint")
        right = self.robot.find_joints("right_wheel_joint")
        self._wheel_joint_ids = left + right

        self._commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._randomize_commands(torch.arange(self.num_envs, device=self.device))

    def get_observations(self) -> torch.Tensor:
        # robot.state is the supported, typed way to read live state. robot.internals
        # is the raw escape hatch; see
        # https://docs.simulo.ai/concepts/scene-robot-world/.
        # pose is [x, y, z, qw, qx, qy, qz]; the quaternion is the last four columns.
        forward = _quat_to_forward(self.robot.state.pose[:, 3:7])
        return torch.cat([forward, self._commands], dim=-1)

    def get_rewards(self) -> torch.Tensor:
        return _compute_rewards(
            self.rew_scale_alignment,
            self.rew_scale_velocity,
            self.robot.state.pose[:, 3:7],
            self.robot.state.linear_velocity,
            self._commands,
        )

    def get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(truncated)
        return terminated, truncated

    def apply_actions(self, actions: torch.Tensor) -> None:
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
        self._commands[env_ids, 0] = torch.cos(angles)
        self._commands[env_ids, 1] = torch.sin(angles)
        self._commands[env_ids, 2] = 0.0


# retries=2 is safe because ResumableCheckpoint saves every 50 iterations and
# resume defaults to "auto": a retried or preempted run picks up from the latest
# checkpoint instead of starting over.
@app.job(
    # Tier 1: T4 GPU, 16 GB VRAM. Run `simulo systems` for the full four-tier catalog.
    system=simulo.SystemType.TIER_1,
    timeout=8 * 60 * 60,
    retries=2,
    callbacks=[simulo.callbacks.ResumableCheckpoint(every=50)],
)
def train_jetbot(num_envs: int = 512, max_iterations: int = 700) -> dict[str, Any]:
    """Train the JetBot direction-following policy with PPO and save the checkpoint.

    Args:
        num_envs: Number of parallel environments to simulate. More environments give
            the trainer more experience per iteration and use more GPU memory.
        max_iterations: Number of PPO policy-update iterations. The default (700) is
            the setting behind the reward figures in the module docstring.

    Returns:
        A JSON-serialisable dict: the saved ``checkpoint`` path inside the volume plus
        training statistics such as ``iterations`` and ``best_reward``.
    """
    env = simulo.LearningEnv(
        task=JetbotTask(),
        num_envs=num_envs,
        device="cuda",
        dt=1.0 / 120.0,
        physics_steps_per_action=2,
        env_spacing=2.0,
        headless=True,
        seed=42,
    )
    trainer = simulo.RLTrainer(env=env, algorithm="PPO", device="cuda", seed=42)

    stats = trainer.train(max_iterations=max_iterations)

    checkpoint = f"{vol.path}/jetbot_final.pt"
    trainer.save(checkpoint)

    # Close the trainer before the environment so the RL library releases its resources.
    trainer.close()
    env.close()

    return {"checkpoint": checkpoint, "num_envs": num_envs, **stats}
