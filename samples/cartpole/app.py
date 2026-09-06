"""Cartpole: train a pole-balancing policy with PPO.

A cart slides along a rail with a pole hinged on top. The policy learns to keep the
pole upright by pushing the cart left or right. This is the core Simulo training shape,
and every other training sample in this repository follows it: a ``simulo.Task``
describes the problem, ``simulo.LearningEnv`` runs thousands of copies of it in
parallel on a GPU, ``simulo.RLTrainer`` trains a policy, and the checkpoint lands in a
durable ``simulo.Volume``.

What this sample shows
----------------------
* A ``simulo.Task`` subclass with the full lifecycle: ``build`` declares the scene,
  ``on_start`` resolves joint indices, and ``get_observations``, ``get_rewards``,
  ``get_dones``, ``apply_actions``, and ``reset_idx`` run every step or reset.
* The robot uses a version-pinned catalog reference
  (``simulo/robot/cartpole:v1``), so every run resolves the same model version.
* ``simulo.callbacks.ResumableCheckpoint(every=50)`` saves a checkpoint every 50
  iterations. That is what makes ``retries=2`` safe: a retried or preempted run resumes
  from its latest checkpoint instead of starting over. It is also what publishes the
  ``best.pt`` and ``latest.pt`` files you download with ``simulo models``.
* Observation (4 values): pole angle, pole angular velocity, cart position, cart
  velocity. Action (1 value): a scaled horizontal force on the cart.

How the file is written, and why
--------------------------------
``simulo run`` imports this file on your machine, where no GPU and no ``torch`` are
installed, and only the Simulo cloud executes the job body. Four habits keep the file
importable in both places:

* ``from __future__ import annotations`` turns every annotation into a string, so
  ``-> torch.Tensor`` never needs ``torch`` at import time.
* The only heavy import, ``import torch``, sits inside ``with app.runtime.imports():``.
  On your machine that block records the import; in the cloud it runs for real.
* The reward kernel is a module-level function under ``@app.runtime.torch_jit``. On
  your machine the decorator is a marker; in the cloud it is ``torch.jit.script``.
* The task class is defined at module level. Its method bodies use ``torch`` and the
  live robot, and none of them run at submit time.

Run it
------
Sign in once with ``simulo login``, then::

    simulo run samples/cartpole/app.py --num-envs 4096 --max-iterations 200

``--num-envs`` and ``--max-iterations`` are ``train_cartpole``'s own parameters. Use
``--num-envs 64 --max-iterations 2`` for a quick check that the job launches.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import simulo

# The cartpole robot: a version-pinned catalog reference.
# Declaring it at module level lets `simulo run` record this exact version with
# the job; the cloud mounts it read-only and the task below uses the same handle.
cartpole = simulo.Asset.from_registry("simulo/robot/cartpole:v1")

# A named, durable, writable volume for the trained checkpoint. This line only
# declares metadata; nothing is created at packaging time. The job reads the
# volume's real directory through ``vol.path``, which resolves only inside a
# running job, never through the ``/out`` mount point declared below.
vol = simulo.Volume.from_name("cartpole-checkpoints", create_if_missing=True)

# Advanced: pick a different Simulo runtime with
# App("name", runtime=simulo.Runtime.from_registry("simulo/gpu-rl:2026.06"));
# see https://docs.simulo.ai/concepts/runtimes/.
app = simulo.App("cartpole", mounts={"/out": vol})

# The one heavy import, deferred: on your machine this block records the import
# instead of resolving it; in the cloud it is a plain import.
with app.runtime.imports():
    import torch  # noqa: F401  (resolved only when the job runs in the cloud)


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

    ``@app.runtime.torch_jit`` is a marker on your machine and ``torch.jit.script`` in
    the cloud, so this can live at module level and its body never runs at submit.
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

    # Set by the training base class when the job runs. Declared here only so a
    # type checker sees the names the methods read; the annotations are strings
    # and never shadow the inherited values.
    device: str
    max_episode_length: int
    episode_length_buf: torch.Tensor
    reset_terminated: torch.Tensor

    def build(self, scene: simulo.Scene) -> None:
        scene.add(simulo.Terrain.plane(name="ground"), at="/", per_environment=False)
        scene.add(
            simulo.Light.dome(name="light", intensity=2000.0, color=(0.75, 0.75, 0.75)),
            at="/",
            per_environment=False,
        )
        self.robot = simulo.Robot(asset=cartpole, initial_pose=simulo.Pose.identity())
        scene.add(self.robot, at="/World/Robot")

    def on_start(self, env: simulo.LearningEnv) -> None:
        self._cart_dof_idx = self.robot.find_joints("slider_to_cart")
        self._pole_dof_idx = self.robot.find_joints("cart_to_pole")
        # robot.state is the supported, typed way to read live state. robot.internals
        # is the raw escape hatch; see
        # https://docs.simulo.ai/concepts/scene-robot-world/.
        self._joint_pos = self.robot.state.joint_positions
        self._joint_vel = self.robot.state.joint_velocities

    def get_observations(self) -> torch.Tensor:
        pole_idx = self._pole_dof_idx[0]
        cart_idx = self._cart_dof_idx[0]
        pole_pos = self._joint_pos[:, pole_idx].view(-1, 1)
        pole_vel = self._joint_vel[:, pole_idx].view(-1, 1)
        cart_pos = self._joint_pos[:, cart_idx].view(-1, 1)
        cart_vel = self._joint_vel[:, cart_idx].view(-1, 1)
        return torch.cat((pole_pos, pole_vel, cart_pos, cart_vel), dim=-1)

    def get_rewards(self) -> torch.Tensor:
        return _compute_rewards(
            self.rew_scale_alive,
            self.rew_scale_terminated,
            self.rew_scale_pole_pos,
            self.rew_scale_cart_vel,
            self.rew_scale_pole_vel,
            self._joint_pos[:, self._pole_dof_idx[0]],
            self._joint_vel[:, self._pole_dof_idx[0]],
            self._joint_pos[:, self._cart_dof_idx[0]],
            self._joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )

    def get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._joint_pos = self.robot.state.joint_positions
        self._joint_vel = self.robot.state.joint_velocities
        pole_idx = self._pole_dof_idx[0]
        cart_idx = self._cart_dof_idx[0]
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        cart_out = torch.abs(self._joint_pos[:, cart_idx]) > self.max_cart_pos
        pole_fallen = torch.abs(self._joint_pos[:, pole_idx]) > math.pi / 2
        terminated = cart_out | pole_fallen
        return terminated, truncated

    def apply_actions(self, actions: torch.Tensor) -> None:
        self.robot.set_joint_effort_target(
            self.action_scale * actions, joint_ids=self._cart_dof_idx
        )

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        num_resets = len(env_ids)
        if num_resets == 0:
            return
        self.robot.reset(env_ids)
        pole_idx = self._pole_dof_idx[0]
        # robot.state has no default-joint-value equivalent, so this stays on the
        # internals escape hatch (there is nothing unstable about reading it here,
        # just no supported, typed name for it yet).
        joint_pos = self.robot.internals.default_joint_pos[env_ids].clone()
        random_angles = torch.empty(num_resets, device=self.device).uniform_(
            self.initial_pole_angle_range[0] * math.pi,
            self.initial_pole_angle_range[1] * math.pi,
        )
        joint_pos[:, pole_idx] += random_angles
        # set_joint_state writes positions and velocities in one call and refreshes
        # the buffers behind robot.state in place. self._joint_pos and
        # self._joint_vel are those same buffers, so they are already current after
        # this call and nothing needs to be written into them directly.
        joint_vel = self.robot.internals.default_joint_vel[env_ids]
        self.robot.set_joint_state(joint_pos, velocities=joint_vel, env_ids=env_ids)


# retries=2 is safe because ResumableCheckpoint saves every 50 iterations and
# resume defaults to "auto": a retried or preempted run picks up from the latest
# checkpoint instead of starting over. The job body below needs no changes.
@app.job(
    gpu="L4",
    timeout=8 * 60 * 60,
    retries=2,
    # Each save updates latest.pt in place; keep_last=N also retains numbered
    # iter_<n>.pt copies.
    callbacks=[simulo.callbacks.ResumableCheckpoint(every=50)],
)
def train_cartpole(num_envs: int = 4096, max_iterations: int = 200) -> dict[str, Any]:
    """Train a cartpole-balancing policy with PPO and save the checkpoint.

    Everything here runs in the Simulo cloud: ``simulo.LearningEnv`` builds ``num_envs``
    parallel copies of ``CartpoleTask`` on the GPU and ``simulo.RLTrainer`` trains a PPO
    policy against them.

    Args:
        num_envs: Number of parallel environments to simulate.
        max_iterations: Number of PPO policy-update iterations.

    Returns:
        A JSON-serialisable dict: the saved ``checkpoint`` path inside the volume plus
        training statistics such as ``iterations`` and ``best_reward``.
    """
    env = simulo.LearningEnv(
        task=CartpoleTask(),
        num_envs=num_envs,
        device="cuda",
        dt=1.0 / 120.0,
        physics_steps_per_action=2,
        env_spacing=4.0,
        headless=True,
        seed=42,
    )
    trainer = simulo.RLTrainer(env=env, algorithm="PPO", device="cuda", seed=42)

    stats = trainer.train(max_iterations=max_iterations)

    # Save the trained policy into the durable volume (vol.path resolves inside the job).
    checkpoint = f"{vol.path}/cartpole_final.pt"
    trainer.save(checkpoint)

    # Close the trainer before the environment so the RL library releases its resources.
    trainer.close()
    env.close()

    return {"checkpoint": checkpoint, "num_envs": num_envs, **stats}
