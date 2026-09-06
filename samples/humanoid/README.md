# Humanoid

## What this shows

The `cartpole` training loop applied to a much heavier body: a simple bipedal humanoid with 21
controlled joints, learning to walk forward while staying upright. The task is the classic
humanoid locomotion problem written out in plain tensor operations, so every term of the 75-value
observation and of the reward is visible in one file.

You will learn how a locomotion observation is assembled from body-frame velocities, projected
gravity, a velocity command, joint state, and the previous action; how per-joint gear ratios
scale a policy's actions into effort targets; and why a fall (termination) and the time limit
(truncation) are reported separately.

## Prerequisites

- Python 3.11 or newer and the Simulo client: `python -m pip install "simulo>=0.23.1,<0.25"`.
- A Simulo account, signed in once with `simulo login`.
- A clone of this repository. The commands below run from its root.
- No GPU on your machine. The job asks for an L4-class GPU in the Simulo cloud and is billed to
  your account.

## Assets

- `simulo/robot/humanoid:v1`: a version-pinned catalog reference for the bipedal humanoid.

## Files and APIs

- `app.py`: two quaternion helpers, the reward kernel, the task, and the `train_humanoid` job.
- `.simuloignore`: files `simulo run` leaves out of the uploaded package.

Simulo names it uses, beyond those in the `cartpole` sample:

- `robot.state.pose` (position and w-first quaternion per environment) and
  `robot.state.joint_velocities` for the reward and the fall check.
- `robot.internals` for the two reads `robot.state` does not offer: the default joint positions
  and the body-frame root velocities. The comment in `get_observations` says why those two frames
  matter.
- `robot.set_joint_effort_target(...)` on all 21 joints at once.

## Run it

```bash
simulo login
simulo run samples/humanoid/app.py --num-envs 1024 --max-iterations 600
```

For a quick check that the job launches, use `--num-envs 64 --max-iterations 2`. Add `--detach`
to submit without waiting for the log.

The job has one configured 8-hour execution budget shared by the initial attempt and its two
retries. Dependency and asset preparation happens before that execution deadline, so this is not
an absolute billing ceiling. Run `simulo cancel <job-id>` to stop a queued or running job.

## What to expect

The reward on this task is a large negative number at the start (energy and action penalties
dominate a random policy) and climbs toward zero as the policy learns. An earlier single-GPU run
of this code reported a best reward near -9500 at the random-policy floor, about -7200 at iteration
50, -2200 at 150, -770 at 300, -450 at 450, and -350 at 600. Read that as the shape of the curve
rather than as numbers to hit; training is not bit-for-bit reproducible. `best_reward` is a mean
completed-episode return across 900 control steps, not a per-step reward.

The defaults collect 9,830,400 transitions: 1024 environments multiplied by 600 policy updates
and the trainer's 16-step rollout. The cited run did not learn a walking gait. For scale, 4096
environments for 500 to 1000 updates would collect 32,768,000 to 65,536,000 transitions.

The result names the checkpoint the job saved into its volume, plus `iterations`, `best_reward`,
and checkpoint bookkeeping fields. A run takes about five minutes at the defaults once capacity is free; a first run
can take longer while the cloud prepares the runtime.

## Inspecting results

```bash
simulo jobs                           # status and job IDs
simulo logs <job-id> --follow         # iteration and checkpoint lines
simulo result <job-id>                # checkpoint, num_envs, iterations, best_reward
simulo models <job-id>                # best.pt and latest.pt
simulo models <job-id> best.pt        # download one, digest-verified
simulo cancel <job-id>                # stop a queued or running job
```

Watch the "Best checkpoint saved" lines in the log: their reward values are the curve described
above. To see the robot, submit with `--viewstream` and open `simulo view` while it runs;
streaming can slow training, so use it to look.

`<job-id>` is printed by `simulo run` and listed by `simulo jobs`. Stopping a log-follow session
only detaches from the stream; it does not cancel the job.

## Troubleshooting

- `simulo run` prints "This wrote a local package only": you are not signed in.
- The job stays `queued`: the cloud is waiting for GPU capacity.
- The reward is negative at the end: that is normal on this task, where the reward climbs
  toward zero. See [What to expect](#what-to-expect).
- The job failed after training started: `simulo logs` prints the platform's reason code and
  detail after its header.

## Extending it

- Increase experience: use a larger `--num-envs` or `--max-iterations`, or continue a finished
  run with `--from <job-id>` and a higher `--max-iterations` total. The scale example above shows
  how to calculate the transition count.
- Reshape the reward: `heading_weight`, `up_weight`, `energy_cost_scale`, `actions_cost_scale`,
  `alive_reward_scale`, and `death_cost` on `HumanoidTask`.
- Change the command: `on_start` builds a fixed "walk forward" command; sample it per environment
  in `reset_idx` to train one policy across directions, as the [JetBot](../jetbot/) sample does.
- Change what counts as a fall: `termination_height`.

## Assets, licensing, attribution

The sample code is available under the [MIT License](../../LICENSE).

This sample installs no third-party packages.

### Attribution

Sample code: Copyright (c) 2026 Simulo LLC.
