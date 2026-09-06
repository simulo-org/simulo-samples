"""Humanoid: train a 21-joint biped to stay upright and move forward.

Where ``cartpole`` trains a 4-value observation and a 1-value action, this sample runs
the same loop on a much heavier body: a simple bipedal humanoid with 21 controlled
joints, learning to walk forward while staying upright. The task is the classic
humanoid locomotion problem written out in plain tensor operations, so every term of
the observation and the reward is visible in this file.

Task summary
------------
* Observation (75 values): base linear velocity (3, body frame), base angular velocity
  (3, body frame, scaled), projected gravity (3, body frame), a fixed "walk forward"
  velocity command (3), joint positions relative to default (21), joint velocities
  (21, scaled), and the previous actions (21).
* Action (21 values): per-joint effort targets, each scaled by a fixed gear ratio.
* Reward: heading alignment plus upright posture, minus energy and action penalties,
  plus an alive bonus, with a death penalty on termination.
* Termination: the torso drops below ``termination_height`` (the robot fell).
  Truncation: the episode time limit. Keeping the two apart lets the trainer treat a
  fall differently from running out of time.

What to expect from the defaults
--------------------------------
An earlier single-GPU run with ``num_envs=1024`` and ``max_iterations=600`` reported
best rewards of about -7200 at iteration 50, -2200 at 150, -770 at 300, -450 at 450,
and -350 at 600. That is a steady climb rather than a walking gait: the defaults are a
short run, and walking needs the larger budget below. Read the figures as the shape of
the curve rather than as numbers to hit; training is not bit-for-bit reproducible. The
defaults collect 9,830,400 transitions: 1024 environments times 600 policy updates
times the trainer's 16-step rollout. For scale, 4096 environments for 500 to 1000
updates would collect 32,768,000 to 65,536,000 transitions.

Run it
------
Sign in once with ``simulo login``, then::

    simulo run samples/humanoid/app.py --num-envs 1024 --max-iterations 600

Use ``--num-envs 64 --max-iterations 2`` for a quick check that the job launches.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import simulo

# The humanoid robot: a version-pinned catalog reference.
humanoid = simulo.Asset.from_registry("simulo/robot/humanoid:v1")

# A named, durable, writable volume for the trained checkpoint. This line only
# declares metadata; nothing is created at packaging time. The job reads the
# volume's real directory through ``vol.path``, which resolves only inside a
# running job.
vol = simulo.Volume.from_name("humanoid-checkpoints", create_if_missing=True)

# Advanced: pick a different Simulo runtime with
# App("name", runtime=simulo.Runtime.from_registry("simulo/gpu-rl:2026.06"));
# see https://docs.simulo.ai/concepts/runtimes/.
app = simulo.App("humanoid", mounts={"/out": vol})

# The one heavy import, deferred: on your machine this block records the import
# instead of resolving it; in the cloud it is a plain import.
with app.runtime.imports():
    import torch  # noqa: F401  (resolved only when the job runs in the cloud)


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a per-env vector by a per-env quaternion (both wxyz, shape (num_envs, *)).

    A plain typed module-level helper, not itself decorated. ``torch.jit.script``
    compiles it transitively when the ``@app.runtime.torch_jit`` reward kernel below
    calls it, and it also runs eagerly when ``HumanoidTask.get_observations`` calls it
    through ``_quat_rotate_inverse``.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return torch.stack([rx, ry, rz], dim=-1)


def _quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by the conjugate of ``quat``: world frame to body frame."""
    quat_conj = quat.clone()
    quat_conj[:, 1:4] = -quat_conj[:, 1:4]
    return _quat_rotate(quat_conj, vec)


@app.runtime.torch_jit
def _compute_rewards(
    heading_weight: float,
    up_weight: float,
    energy_cost_scale: float,
    actions_cost_scale: float,
    alive_reward_scale: float,
    death_cost: float,
    root_quat: torch.Tensor,
    up_vec: torch.Tensor,
    heading_vec: torch.Tensor,
    joint_vel: torch.Tensor,
    prev_actions: torch.Tensor,
    joint_gears: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled locomotion reward kernel (the classic humanoid locomotion reward).

    ``@app.runtime.torch_jit`` is a marker on your machine and ``torch.jit.script`` in
    the cloud, so this can live at module level and its body never runs at submit.
    ``_quat_rotate`` is compiled transitively: TorchScript scripts the plain typed
    functions a kernel calls.
    """
    heading_world = _quat_rotate(root_quat, heading_vec)
    heading_alignment = heading_world[:, 0]

    up_world = _quat_rotate(root_quat, up_vec)
    upright_alignment = up_world[:, 2]

    energy = torch.sum(torch.abs(prev_actions * joint_gears * joint_vel), dim=-1)
    action_norm = torch.sum(torch.square(prev_actions), dim=-1)
    terminated = reset_terminated.float()

    reward = (
        heading_weight * heading_alignment
        + up_weight * upright_alignment
        - energy_cost_scale * energy
        - actions_cost_scale * action_norm
        + alive_reward_scale * (1.0 - terminated)
        + death_cost * terminated
    )
    # ``.view(-1)`` is belt and braces here: nothing above squeezes, so ``reward``
    # is already (num_envs,) even when num_envs == 1, but it locks in the per-env
    # reward contract explicitly rather than relying on that being true.
    return reward.view(-1)


class HumanoidTask(simulo.Task):
    """Walk a 21-DOF bipedal humanoid forward while staying upright.

    Observation (75-dim): base linear velocity (3, body frame), base angular velocity
    (3, body frame, scaled), projected gravity (3, body frame), a fixed forward
    velocity command (3), joint positions relative to default (21), joint velocities
    (21, scaled), previous actions (21). Action (21-dim): per-joint effort targets,
    scaled by ``joint_gears``.
    """

    observation_dim = 75
    action_dim = 21

    episode_length_s = 15.0
    action_scale = 1.0

    # Joint gear (torque scale) ratios: the classic humanoid locomotion values, in
    # the robot model's joint declaration order (lower_waist x2, upper_arms x4,
    # pelvis, lower_arms x2, thighs x6, knees x2, feet x4).
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

    # Set by the training base class when the job runs. Declared here only so a
    # type checker sees the names the methods read; the annotations are strings.
    device: str
    num_envs: int
    max_episode_length: int
    episode_length_buf: torch.Tensor
    reset_terminated: torch.Tensor

    def build(self, scene: simulo.Scene) -> None:
        scene.add(simulo.Terrain.plane(name="ground"), at="/World", per_environment=False)
        scene.add(
            simulo.Light.dome(name="light", intensity=2000.0, color=(0.75, 0.75, 0.75)),
            at="/World",
            per_environment=False,
        )
        self.robot = simulo.Robot(asset=humanoid, initial_pose=simulo.Pose.identity())
        scene.add(self.robot, at="/World/Robot")

    def on_start(self, env: simulo.LearningEnv) -> None:
        self._joint_gears = torch.tensor(self.joint_gears, device=self.device, dtype=torch.float32)
        self._prev_actions = torch.zeros(
            (self.num_envs, self.action_dim), device=self.device, dtype=torch.float32
        )

        # Fixed "walk forward" command: [vx, vy, heading_rate] = [1, 0, 0].
        self._commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._commands[:, 0] = 1.0

        # Per-env up / heading reference vectors (world frame), pre-broadcast to
        # (num_envs, 3) so ``_quat_rotate`` stays branch-free and JIT-friendly.
        self._up_vec = torch.zeros((self.num_envs, 3), device=self.device)
        self._up_vec[:, 2] = 1.0
        self._heading_vec = torch.zeros((self.num_envs, 3), device=self.device)
        self._heading_vec[:, 0] = 1.0

    def get_observations(self) -> torch.Tensor:
        # Stays on the internals escape hatch: this reads `default_joint_pos`
        # (robot.state has no default-joint-value equivalent) AND the BODY-FRAME
        # root velocities `root_ang_vel_b` / `root_lin_vel_b`. robot.state's
        # `angular_velocity` / `linear_velocity` are the WORLD-frame members, so
        # substituting them would silently change the observation's frame, not
        # just its spelling.
        data = self.robot.internals
        projected_gravity = _quat_rotate_inverse(data.root_quat_w, self._up_vec)
        joint_pos_rel = data.joint_pos - data.default_joint_pos
        joint_vel_scaled = data.joint_vel * self.dof_vel_scale
        ang_vel_scaled = data.root_ang_vel_b * self.angular_velocity_scale

        return torch.cat(
            [
                data.root_lin_vel_b,
                ang_vel_scaled,
                projected_gravity,
                self._commands,
                joint_pos_rel,
                joint_vel_scaled,
                self._prev_actions,
            ],
            dim=-1,
        )

    def get_rewards(self) -> torch.Tensor:
        # Unlike get_observations above, this reads only the root quaternion and
        # the joint velocities, both covered by robot.state (robot.state.pose is
        # [x, y, z, qw, qx, qy, qz]; [:, 3:7] is the w-first root quaternion).
        return _compute_rewards(
            self.heading_weight,
            self.up_weight,
            self.energy_cost_scale,
            self.actions_cost_scale,
            self.alive_reward_scale,
            self.death_cost,
            self.robot.state.pose[:, 3:7],
            self._up_vec,
            self._heading_vec,
            self.robot.state.joint_velocities,
            self._prev_actions,
            self._joint_gears,
            self.reset_terminated,
        )

    def get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # robot.state.pose is [x, y, z, qw, qx, qy, qz]; z (height) is column 2.
        terminated = self.robot.state.pose[:, 2] < self.termination_height
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def apply_actions(self, actions: torch.Tensor) -> None:
        self._prev_actions = actions.clone()
        scaled_efforts = self.action_scale * actions * self._joint_gears
        self.robot.set_joint_effort_target(scaled_efforts)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        num_resets = len(env_ids)
        if num_resets == 0:
            return
        self.robot.reset(env_ids)
        self._prev_actions[env_ids] = 0.0


# retries=2 is safe because ResumableCheckpoint saves every 50 iterations and
# resume defaults to "auto": a retried or preempted run picks up from the latest
# checkpoint instead of starting over.
@app.job(
    gpu="L4",
    timeout=8 * 60 * 60,
    retries=2,
    callbacks=[simulo.callbacks.ResumableCheckpoint(every=50)],
)
def train_humanoid(num_envs: int = 1024, max_iterations: int = 600) -> dict[str, Any]:
    """Train the humanoid-walking policy with PPO and save the checkpoint.

    Args:
        num_envs: Number of parallel environments to simulate. More environments give
            the trainer more experience per iteration and use more GPU memory.
        max_iterations: Number of PPO policy-update iterations. The default (600) is
            the setting behind the reward curve in the module docstring, which climbs
            steadily without reaching a walking gait.

    Returns:
        A JSON-serialisable dict: the saved ``checkpoint`` path inside the volume plus
        training statistics such as ``iterations`` and ``best_reward``.
    """
    env = simulo.LearningEnv(
        task=HumanoidTask(),
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

    checkpoint = f"{vol.path}/humanoid_final.pt"
    trainer.save(checkpoint)

    # Close the trainer before the environment so the RL library releases its resources.
    trainer.close()
    env.close()

    return {"checkpoint": checkpoint, "num_envs": num_envs, **stats}
