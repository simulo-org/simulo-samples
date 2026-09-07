# Franka reach

## What this shows

A Franka Panda arm learns to put its hand on a goal point that moves every episode. The policy
never commands joints: it emits a small x, y, z nudge to a target point in front of the arm, and a
`simulo.DifferentialIKController` works out the seven joint angles that put the hand there. The
reward is how close the hand ends up to the goal.

You will learn task-space control (a 3-value action driving a 7-joint arm through an IK
controller), how to read a link's pose back in the robot's own base frame so observations and
goals share a coordinate system, how to keep an IK target inside a configured box intended to
stay within the arm's working region, and how to reset only the environments that finished, with
a fresh goal for each.

## Prerequisites

- Python 3.11 or newer and the Simulo client: `python -m pip install "simulo>=0.26.0,<0.27"`.
- A Simulo account, signed in once with `simulo login`.
- A clone of this repository. The commands below run from its root.
- No GPU on your machine. The job asks for a Tier 1 GPU (T4) in the Simulo cloud and is
  billed to your account. Hardware describes the job's own request, not queue priority:
  it waits on the same shared GPU fleet as every other job.

## Assets

- `simulo/robot/franka-panda:v1`: a version-pinned catalog reference for the arm.

## Files and APIs

- `app.py`: the reward kernel, the task, and the `train_franka_reach` job.
- `.simuloignore`: files `simulo run` leaves out of the uploaded package.

Simulo names it uses, beyond those in the [Cartpole](../cartpole/) sample:

- `simulo.DifferentialIKController`, constructed in `on_start` against the built robot:

  ```python
  simulo.DifferentialIKController(
      robot=self.robot,
      end_effector="panda_hand",
      joints=self._arm_dof_idx,
      ik_method="dls",
      command_type="pose",
  )
  ```

  The task drives it with `ik.move_to(...)` every step and calls `ik.reset()` on episode reset.
- `robot.get_body_pose_in_base_frame("panda_hand")`: the hand's position and orientation relative
  to the arm's base, one row per environment.
- `robot.find_joints("panda_joint.*")`: the seven arm joints by name pattern, leaving the two
  finger joints out of the task.
- `robot.state.joint_positions` and `robot.state.joint_velocities`, sliced to the arm joints.

## Run it

```bash
simulo login
simulo run samples/franka-reach/app.py --num-envs 2048 --max-iterations 300
```

For a quick check that the job launches, use `--num-envs 64 --max-iterations 2`. Add `--detach`
to submit without waiting for the log.

The job has one configured 4-hour execution budget shared by the initial attempt and its two
retries. Dependency and asset preparation happens before that execution deadline, so this is not
an absolute billing ceiling. Run `simulo cancel <job-id>` to stop a queued or running job.

## What to expect

The per-step reward is at most 1.5 before the action penalty: 1.0 from the coarse distance term and
0.5 from the fine term at zero hand-to-goal distance. An episode has 240 control steps, so the
no-penalty episode-return ceiling is 360. The logged `best_reward` is a mean completed-episode
return, not a per-step reward. Read it as a trend: it climbs as the hand finishes closer to the
goal more often.

The result names the checkpoint the job saved into its volume, plus `iterations`, `best_reward`,
and checkpoint bookkeeping fields. A run takes about three minutes at the defaults once capacity is free; a first run
can take longer while the cloud prepares the runtime.

## Inspecting results

```bash
simulo jobs                           # status and job IDs
simulo logs <job-id> --follow         # iteration and checkpoint lines
simulo result <job-id>                # checkpoint, num_envs, iterations, best_reward
simulo models <job-id>                # best.pt and latest.pt
simulo models <job-id> best.pt        # download one, digest-verified
simulo export <job-id>                # best checkpoint as a portable ONNX bundle
simulo cancel <job-id>                # stop a queued or running job
```

To watch the arm, submit with `--viewstream` and open `simulo view` while it runs; streaming can
slow training, so use it to look.

`<job-id>` is printed by `simulo run` and listed by `simulo jobs`. Stopping a log-follow session
only detaches from the stream; it does not cancel the job.

## Troubleshooting

- `simulo run` prints "This wrote a local package only": you are not signed in.
- The job fails immediately with a `ValueError` about `num_envs` or `max_iterations`: both must be
  positive integers. The job checks them before starting the simulation.
- The job stays `queued`: the cloud is waiting for GPU capacity.
- The arm thrashes after you edit the task: check that nothing else writes joint targets. The IK
  controller applies the joint commands itself, and a `set_joint_position_target` call alongside
  it fights over the same joints.
- The first observation looks wrong after you edit `reset_idx`: the hand position is refreshed
  both in `get_dones` and at the end of `reset_idx`, because a reset is followed by an observation
  and not by a done check. Keep both refreshes.

## Extending it

- Goal region: `goal_x_range`, `goal_y_range`, and `goal_z_range` on `FrankaReachTask` set where
  goals are sampled; `target_*_range` clamps the commanded point.
- Step size: `action_scale` is how far one full-scale action moves the target per step.
- Hand orientation: `ee_orientation` is held for the whole episode. Add it to the action to learn
  orientation as well, at the cost of three more action values.
- Reward shape: `distance_std` and `fine_std` set how wide the two distance terms are.
- Next step: give the arm something to pick up. A `simulo.Prop` exposes an object's live pose,
  and a `simulo.ParallelGripperActuator` turns one action value into open, close, or hold.

## Assets, licensing, attribution

The sample code is available under the [MIT License](../../LICENSE).

This sample installs no third-party packages.

### Attribution

Sample code: Copyright (c) 2026 Simulo LLC.
