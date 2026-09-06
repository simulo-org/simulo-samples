"""Franka reach: task-space reaching with a differential IK controller.

The first manipulation sample. Where ``cartpole`` and ``humanoid`` learn in joint
space, this one learns in task space: the policy does not command joints at all. It
nudges a target point in front of a Franka Panda arm, and a
``simulo.DifferentialIKController`` works out the seven joint angles that put the hand
there. The reward is simply how close the hand ends up to a goal that moves every
episode.

What this sample shows
----------------------
* Task-space control. ``simulo.DifferentialIKController`` is constructed against the
  already-built ``simulo.Robot`` in ``on_start`` and driven every step with
  ``move_to(...)``. It computes and applies the joint commands itself, so this task
  never calls ``set_joint_position_target``; the two would fight over the same joints.
* A 3-value action for a 7-joint arm. The policy emits a small Cartesian delta and the
  controller expands it into joint space. The action space matches the task, not the
  hardware.
* Reading the hand back. ``robot.get_body_pose_in_base_frame("panda_hand")`` returns
  the hand's position and orientation relative to the arm's own base, one row per
  environment. Both the reward and the observation are built from it.
* Bounded targets. The commanded point is clamped to a configured box intended to
  stay within the arm's working region, so large actions cannot move it without bound.

Sizing and reward scale
-----------------------
The
per-step reward is at most 1.5 before the action penalty, and each episode has 240
control steps, giving a no-penalty episode-return ceiling of 360. Logged
``best_reward`` values are mean completed-episode returns, not per-step rewards.

Run it
------
Sign in once with ``simulo login``, then::

    simulo run samples/franka-reach/app.py --num-envs 2048 --max-iterations 300

Use ``--num-envs 64 --max-iterations 2`` for a quick check that the job launches.
"""

from __future__ import annotations

from typing import Any, Tuple

import simulo

# The Franka Panda arm: a version-pinned catalog reference.
# Declared at module level so `simulo run` records this exact version with the job.
franka = simulo.Asset.from_registry("simulo/robot/franka-panda:v1")

# A named, durable, writable volume for the trained checkpoint.
vol = simulo.Volume.from_name("franka-reach-checkpoints", create_if_missing=True)

app = simulo.App("franka-reach", mounts={"/out": vol})

# The one heavy import, deferred until the job runs in the cloud.
with app.runtime.imports():
    import torch  # noqa: F401  (resolved only when the job runs in the cloud)


@app.runtime.torch_jit
def _compute_rewards(
    rew_scale_distance: float,
    rew_scale_fine: float,
    rew_scale_action: float,
    distance_std: float,
    fine_std: float,
    ee_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled reward kernel for task-space reaching.

    Two nested distance terms plus an action penalty:

    * a wide ``tanh`` shell that pulls the hand across the workspace toward
      the goal from anywhere,
    * a narrow one that only pays out in the last few centimetres, so the
      policy keeps improving after the wide term has saturated,
    * a small penalty on action magnitude, which stops the policy from
      thrashing the target point around once it is already on the goal.
    """
    distance = torch.norm(goal_pos - ee_pos, dim=-1)
    rew_coarse = rew_scale_distance * (1.0 - torch.tanh(distance / distance_std))
    rew_fine = rew_scale_fine * (1.0 - torch.tanh(distance / fine_std))
    rew_action = rew_scale_action * torch.sum(torch.square(actions), dim=-1)

    reward: torch.Tensor = rew_coarse + rew_fine + rew_action
    # Keep the (num_envs,) per-env reward contract even when num_envs == 1.
    return reward.view(-1)


class FrankaReachTask(simulo.Task):
    """Move a Franka Panda's hand onto a goal point that moves every episode.

    Observation (23-dim): 7 joint positions (relative to the arm's rest pose),
    7 joint velocities, the hand position, the goal position, and the vector
    from hand to goal, with all positions in the arm's own base frame.

    Action (3-dim): a Cartesian delta applied to the IK target point. The
    controller turns that into joint commands.
    """

    observation_dim = 23
    action_dim = 3

    episode_length_s = 4.0

    # How far one full-scale action moves the IK target point, per step [m].
    action_scale = 0.05

    # The link the controller drives and the joints it is allowed to move.
    # `panda_hand` is the wrist plate; the two finger joints are deliberately
    # NOT in this list: this task has no gripper.
    end_effector = "panda_hand"
    arm_joint_pattern = "panda_joint.*"

    # Goal sampling box, in the arm's base frame [m]. Sized to sit inside the arm's
    # working region.
    goal_x_range = (0.35, 0.60)
    goal_y_range = (-0.25, 0.25)
    goal_z_range = (0.20, 0.50)

    # Where the commanded point starts every episode: the centre of the goal
    # box, in the arm's base frame [m]. Fixed and identical for every episode,
    # so the policy's job (walk the point from here onto the goal) is a real
    # one and the first IK request of an episode is always a modest, bounded
    # move rather than a jump from wherever the last episode ended.
    initial_target = (0.475, 0.0, 0.35)

    # The IK target is clamped to this configured box so large actions cannot
    # move the commanded point without bound.
    target_x_range = (0.25, 0.70)
    target_y_range = (-0.40, 0.40)
    target_z_range = (0.10, 0.65)

    # Hand orientation held for the whole episode: pointing straight down,
    # as a w-first quaternion. Reaching is a position task; pinning the
    # orientation keeps the arm in a sane posture without adding 3 more
    # action dimensions.
    ee_orientation = (0.0, 1.0, 0.0, 0.0)

    rew_scale_distance = 1.0
    rew_scale_fine = 0.5
    rew_scale_action = -0.01
    distance_std = 0.20
    fine_std = 0.04

    # Framework-injected at runtime by the training base (declared here only so
    # the type checker sees the names the methods read; the annotations are
    # PEP 563 strings and never shadow the inherited values).
    device: str
    num_envs: int
    max_episode_length: int
    episode_length_buf: torch.Tensor
    reset_terminated: torch.Tensor

    def build(self, scene: simulo.Scene) -> None:
        scene.add(simulo.Terrain.plane(name="ground"), at="/", per_environment=False)
        scene.add(
            simulo.Light.dome(name="light", intensity=2500.0, color=(0.75, 0.75, 0.75)),
            at="/",
            per_environment=False,
        )
        self.robot = simulo.Robot(asset=franka, initial_pose=simulo.Pose.identity())
        scene.add(self.robot, at="/World/Robot")

    def on_start(self, env: simulo.LearningEnv) -> None:
        self._arm_dof_idx = self.robot.find_joints(self.arm_joint_pattern)

        # The controller is constructed against the ALREADY-BUILT robot, here
        # in on_start, because it reads the robot's joints and bodies, which do not
        # exist during build(). `command_type="pose"` makes its command a
        # 7-vector: the hand's target position in the base frame plus the
        # w-first orientation quaternion held below.
        self.ik = simulo.DifferentialIKController(
            robot=self.robot,
            end_effector=self.end_effector,
            joints=self._arm_dof_idx,
            ik_method="dls",
            command_type="pose",
        )

        # robot.state has no default-joint-value equivalent, so this stays on
        # the internals escape hatch (there is nothing unstable about reading
        # it here, just no supported, typed name for it yet).
        self._default_joint_pos = self.robot.internals.default_joint_pos.clone()

        zeros = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_pos = zeros.clone()
        self._ee_pos = zeros.clone()
        self._actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._orientation = torch.tensor(self.ee_orientation, device=self.device).repeat(
            self.num_envs, 1
        )
        self._initial_target = torch.tensor(self.initial_target, device=self.device).repeat(
            self.num_envs, 1
        )
        self._target_pos = self._initial_target.clone()

        self._sample_goals(torch.arange(self.num_envs, device=self.device))

    # -- helpers ----------------------------------------------------------

    def _sample_goals(self, env_ids: torch.Tensor) -> None:
        """Draw a fresh goal point for each environment being reset."""
        count = len(env_ids)
        for axis, (low, high) in enumerate(
            (self.goal_x_range, self.goal_y_range, self.goal_z_range)
        ):
            self._goal_pos[env_ids, axis] = torch.empty(count, device=self.device).uniform_(
                low, high
            )

    def _read_ee_position(self) -> torch.Tensor:
        """Hand position in the arm's base frame, shape ``(num_envs, 3)``.

        ``get_body_pose_in_base_frame`` is the supported, typed readback; it
        returns ``(position, orientation)`` and this task only needs the
        position. Before the runtime attaches the robot it returns
        ``(None, None)``, so the zeroes below are the honest pre-attach value.
        """
        position, _ = self.robot.get_body_pose_in_base_frame(self.end_effector)
        if position is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        return position

    def _clamp_target(self) -> None:
        for axis, (low, high) in enumerate(
            (self.target_x_range, self.target_y_range, self.target_z_range)
        ):
            self._target_pos[:, axis] = self._target_pos[:, axis].clamp(low, high)

    # -- the Task contract ------------------------------------------------

    def get_observations(self) -> torch.Tensor:
        joint_pos = self.robot.state.joint_positions[:, self._arm_dof_idx]
        joint_vel = self.robot.state.joint_velocities[:, self._arm_dof_idx]
        joint_pos_rel = joint_pos - self._default_joint_pos[:, self._arm_dof_idx]
        return torch.cat(
            (
                joint_pos_rel,
                joint_vel,
                self._ee_pos,
                self._goal_pos,
                self._goal_pos - self._ee_pos,
            ),
            dim=-1,
        )

    def get_rewards(self) -> torch.Tensor:
        return _compute_rewards(
            self.rew_scale_distance,
            self.rew_scale_fine,
            self.rew_scale_action,
            self.distance_std,
            self.fine_std,
            self._ee_pos,
            self._goal_pos,
            self._actions,
        )

    def get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Refresh the hand readback once per step, here, before the reward and
        # the next observation both read it.
        self._ee_pos = self._read_ee_position()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(truncated)
        return terminated, truncated

    def apply_actions(self, actions: torch.Tensor) -> None:
        # This ALIASES the trainer's own action tensor: nothing between the
        # trainer and this method copies it defensively, and the trainer stores
        # that same tensor as the transition's action after the step returns.
        # So: read `self._actions`, never write into it. An in-place write here
        # (or anywhere downstream) lands in a transition PPO is about to record
        # against a real reward.
        self._actions = actions
        self._target_pos = self._target_pos + self.action_scale * actions
        self._clamp_target()
        # A (num_envs, 7) pose command: position + the fixed w-first
        # orientation. The controller computes the joint targets and writes
        # them to the robot itself.
        self.ik.move_to(torch.cat((self._target_pos, self._orientation), dim=-1))

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.robot.reset(env_ids)

        # Back to the arm's rest pose with a little joint noise, so every
        # episode starts from a slightly different posture. The noise goes on
        # the SEVEN ARM JOINTS only: the Franka's two finger joints are
        # prismatic with 0.04 m of total travel, and +/-0.05 m of noise placed
        # them outside their own limits at the start of an episode. They are
        # not part of this task either -- its action is a 3-dim Cartesian
        # nudge, and it declares no gripper.
        joint_pos = self.robot.internals.default_joint_pos[env_ids].clone()
        arm = self._arm_dof_idx
        joint_pos[:, arm] += torch.empty_like(joint_pos[:, arm]).uniform_(-0.05, 0.05)
        joint_vel = self.robot.internals.default_joint_vel[env_ids]
        self.robot.set_joint_state(joint_pos, velocities=joint_vel, env_ids=env_ids)

        self.ik.reset()
        self._sample_goals(env_ids)
        self._target_pos[env_ids] = self._initial_target[env_ids]

        # Refresh the hand readback here as well as in `get_dones`. `env.reset()`
        # runs `reset_idx` and then `get_observations` -- it never calls
        # `get_dones` -- so without this the FIRST observation of a run reports
        # the hand at the origin and a goal-relative vector measured from it.
        self._ee_pos = self._read_ee_position()

        # `self._actions[env_ids] = 0.0` used to live here. It was dead for
        # this task's arithmetic (`_actions` is read only by `get_rewards`,
        # which runs earlier in the same `env.step`, and `apply_actions`
        # rebinds it before the next read) and actively harmful otherwise: it
        # wrote zeros into the trainer's own tensor. See `apply_actions`.


@app.job(
    gpu="L4",
    timeout=4 * 60 * 60,
    retries=2,
    callbacks=[simulo.callbacks.ResumableCheckpoint(every=50)],
)
def train_franka_reach(num_envs: int = 2048, max_iterations: int = 300) -> dict[str, Any]:
    """Train a task-space reaching policy with PPO and save the checkpoint.

    Args:
        num_envs: Number of parallel environments to simulate.
        max_iterations: Number of PPO policy-update iterations.

    Returns:
        A JSON-serialisable dict: the saved ``checkpoint`` path inside the volume plus
        training statistics such as ``iterations`` and ``best_reward``.

    Raises:
        ValueError: If ``num_envs`` or ``max_iterations`` is not a positive
            integer. Checked here, at the argument boundary, because the
            alternative is a GPU job that starts the simulation and then fails
            somewhere less legible.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be a positive integer, got {num_envs}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be a positive integer, got {max_iterations}")

    env = simulo.LearningEnv(
        task=FrankaReachTask(),
        num_envs=num_envs,
        device="cuda",
        dt=1.0 / 120.0,
        physics_steps_per_action=2,
        env_spacing=2.5,
        headless=True,
        seed=42,
    )
    trainer = simulo.RLTrainer(env=env, algorithm="PPO", device="cuda", seed=42)

    stats = trainer.train(max_iterations=max_iterations)

    checkpoint = f"{vol.path}/franka_reach_final.pt"
    trainer.save(checkpoint)

    # Close the trainer before the environment so the RL library releases its
    # resources first.
    trainer.close()
    env.close()

    return {"checkpoint": checkpoint, "num_envs": num_envs, **stats}
