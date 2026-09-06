"""Bring your own robot: publish a URDF arm to your catalog and train it.

The robot this sample trains starts out as a plain ``.urdf`` file: mesh geometry,
link masses, inertia tensors, and joint limits, kept in ``assets/robot/byo-urdf-arm/``
at the root of this repository. Publishing that directory with ``simulo asset publish``
puts the arm in your own organization's catalog as ``robot/byo-urdf-arm:v1``, and the job
below then consumes it exactly the way the other samples consume a Simulo catalog robot.

Publish the asset before you run this app. ``README.md`` next to this file has the
command; without it the job fails when the scene asks for a robot the catalog does not
have.

What this sample shows
----------------------
* An asset reference with no publisher segment, ``robot/byo-urdf-arm:v1``. It resolves
  against the organization you are signed in as, rather than against the Simulo
  catalog, so the same file works for anyone who publishes their own arm under that
  name.
* A ``simulo.Task`` whose robot came out of your own files, with the same lifecycle
  every other training sample uses: ``build`` declares the scene, ``on_start`` resolves
  joint indices, and the per-step methods run the task.
* Joint-space reaching. Observation (9 values): the three joint angles, the three joint
  velocities, and the three target angles, ordered ``[shoulder_pan, shoulder_lift,
  elbow]``. Action (3 values): a scaled effort for each of those joints.
* A reward that is minus the squared distance to a target joint configuration, resampled
  every episode, plus a small velocity penalty so the arm settles on the target instead
  of swinging through it.
* No failure condition: every episode runs to the time limit, and the target changes on
  reset.

Run it
------
Sign in with ``simulo login``, publish ``assets/robot/byo-urdf-arm/``, then::

    simulo run samples/byo-urdf-arm/app.py --num-envs 256 --max-iterations 150

``--num-envs`` and ``--max-iterations`` are ``train``'s own parameters. Use
``--num-envs 16 --max-iterations 2`` for a quick check that the job launches.
"""

from __future__ import annotations

from typing import Any, Tuple

import simulo

# Your arm, published to your own organization's catalog. The reference carries no
# publisher segment, so it resolves against the organization you are signed in as
# rather than against the Simulo catalog.
byo_arm = simulo.Asset.from_registry("robot/byo-urdf-arm:v1")

# A named, durable, writable volume for the trained checkpoint. This line only
# declares metadata; nothing is created at packaging time. The job reads the
# volume's real directory through ``vol.path``, which resolves only inside a
# running job, never through the ``/out`` mount point declared below.
vol = simulo.Volume.from_name("byo-urdf-arm-checkpoints", create_if_missing=True)

# Advanced: pick a different Simulo runtime with
# App("name", runtime=simulo.Runtime.from_registry("simulo/gpu-rl:2026.06"));
# see https://docs.simulo.ai/concepts/runtimes/.
app = simulo.App("byo-urdf-arm", mounts={"/out": vol})

# The one heavy import, deferred: on your machine this block records the import
# instead of resolving it; in the cloud it is a plain import.
with app.runtime.imports():
    import torch  # noqa: F401  (resolved only when the job runs in the cloud)


@app.runtime.torch_jit
def _compute_rewards(
    rew_scale_distance: float,
    rew_scale_velocity: float,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    target_pos: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled distance and velocity reward kernel.

    ``@app.runtime.torch_jit`` is a marker on your machine and ``torch.jit.script`` in
    the cloud, so this can live at module level and its body never runs at submit.
    """
    pos_error = joint_pos - target_pos
    dist_sq = torch.sum(torch.square(pos_error), dim=-1)
    vel_sq = torch.sum(torch.square(joint_vel), dim=-1)
    reward = rew_scale_distance * dist_sq + rew_scale_velocity * vel_sq
    # Keep the (num_envs,) per-env reward contract even when num_envs == 1.
    return reward.view(-1)


class ByoArmTask(simulo.Task):
    """Reach a randomized target joint configuration with your own arm.

    Observation (9-dim): joint angles (3) + joint velocities (3) + target joint angles
    (3), ordered ``[shoulder_pan, shoulder_lift, elbow]``. Action (3-dim): scaled joint
    effort for the same three joints.
    """

    observation_dim = 9
    action_dim = 3

    episode_length_s = 5.0
    action_scale = 10.0  # [N*m], well inside each joint's 30 N*m effort limit

    #: Per-episode target sampled uniformly in [-target_amplitude, +target_amplitude]
    #: for each of the three joints, comfortably inside every joint limit the URDF
    #: declares (the narrowest is shoulder_lift at +-1.9 rad).
    target_amplitude = 1.0  # [rad]

    rew_scale_distance = -1.0
    rew_scale_velocity = -0.01

    # Set by the training base class when the job runs. Declared here only so a
    # type checker sees the names the methods read; the annotations are strings
    # and never shadow the inherited values.
    device: str
    num_envs: int
    max_episode_length: int
    episode_length_buf: torch.Tensor

    def build(self, scene: simulo.Scene) -> None:
        scene.add(simulo.Terrain.plane(name="ground"), at="/World/ground", per_environment=False)
        scene.add(
            simulo.Light.dome(name="light", intensity=2000.0, color=(0.75, 0.75, 0.75)),
            at="/World/Light",
            per_environment=False,
        )
        # Your arm, on a fixed base: bolted at the origin rather than free-floating.
        self.robot = simulo.Robot(asset=byo_arm, initial_pose=simulo.Pose.identity())
        scene.add(self.robot, at="/World/Arm")

    def on_start(self, env: simulo.LearningEnv) -> None:
        pan = self.robot.find_joints("shoulder_pan")
        lift = self.robot.find_joints("shoulder_lift")
        elbow = self.robot.find_joints("elbow")
        self._joint_ids = pan + lift + elbow

        self._target = torch.zeros((self.num_envs, 3), device=self.device)
        self._randomize_target(torch.arange(self.num_envs, device=self.device))

    def get_observations(self) -> torch.Tensor:
        # robot.state is the supported, typed way to read live state. robot.internals
        # is the raw escape hatch; see
        # https://docs.simulo.ai/concepts/scene-robot-world/.
        joint_pos = self.robot.state.joint_positions[:, self._joint_ids]
        joint_vel = self.robot.state.joint_velocities[:, self._joint_ids]
        return torch.cat([joint_pos, joint_vel, self._target], dim=-1)

    def get_rewards(self) -> torch.Tensor:
        return _compute_rewards(
            self.rew_scale_distance,
            self.rew_scale_velocity,
            self.robot.state.joint_positions[:, self._joint_ids],
            self.robot.state.joint_velocities[:, self._joint_ids],
            self._target,
        )

    def get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(truncated)
        return terminated, truncated

    def apply_actions(self, actions: torch.Tensor) -> None:
        self.robot.set_joint_effort_target(self.action_scale * actions, joint_ids=self._joint_ids)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        num_resets = len(env_ids)
        if num_resets == 0:
            return
        self.robot.reset(env_ids)
        self._randomize_target(env_ids)

    def _randomize_target(self, env_ids: torch.Tensor) -> None:
        """Sample a new target angle per joint, uniform in the amplitude band."""
        n = len(env_ids)
        self._target[env_ids] = (
            torch.rand((n, 3), device=self.device) * 2.0 - 1.0
        ) * self.target_amplitude


# retries=2 is safe because ResumableCheckpoint saves every 50 iterations and
# resume defaults to "auto": a retried or preempted run picks up from the latest
# checkpoint instead of starting over. The job body below needs no changes.
@app.job(
    gpu="L4",
    timeout=8 * 60 * 60,
    retries=2,
    callbacks=[simulo.callbacks.ResumableCheckpoint(every=50)],
)
def train(num_envs: int = 256, max_iterations: int = 150) -> dict[str, Any]:
    """Train the reaching policy with PPO and save the checkpoint.

    Everything here runs in the Simulo cloud: ``simulo.LearningEnv`` builds ``num_envs``
    parallel copies of ``ByoArmTask`` on the GPU and ``simulo.RLTrainer`` trains a PPO
    policy against them.

    Args:
        num_envs: Number of parallel environments to simulate.
        max_iterations: Number of PPO policy-update iterations.

    Returns:
        A JSON-serialisable dict: the saved ``checkpoint`` path inside the volume, the
        catalog reference the run trained against, and training statistics such as
        ``iterations`` and ``best_reward``.
    """
    env = simulo.LearningEnv(
        task=ByoArmTask(),
        num_envs=num_envs,
        device="cuda",
        dt=1.0 / 120.0,
        physics_steps_per_action=2,
        env_spacing=1.5,
        headless=True,
        seed=42,
    )
    trainer = simulo.RLTrainer(env=env, algorithm="PPO", device="cuda", seed=42)

    stats = trainer.train(max_iterations=max_iterations)

    # Save the trained policy into the durable volume (vol.path resolves inside the job).
    checkpoint = f"{vol.path}/byo_urdf_arm_final.pt"
    trainer.save(checkpoint)

    # Close the trainer before the environment so the RL library releases its resources.
    trainer.close()
    env.close()

    return {
        "checkpoint": checkpoint,
        "num_envs": num_envs,
        "robot_asset": "robot/byo-urdf-arm:v1",
        **stats,
    }
