# Franka reach

## What this shows

Learn task-space control by training a Franka Panda arm to move its hand to a changing goal. The
focused lesson is how a three-value Cartesian policy action drives seven arm joints through
`simulo.DifferentialIKController`.

## Prerequisites

- Python 3.11 or newer with the Simulo client installed.
- A Simulo account and `simulo login`.
- Run every command below from the repository root.

## Assets

The task uses the version-pinned catalog asset `simulo/robot/franka-panda:v1`. You do not need
to publish an asset for this sample.

## Files and APIs

- `task.py` defines `FrankaReachTask`, goal sampling, hand pose readback, and inverse kinematics.
  The policy observes the arm's joints, the hand, the goal, and the target point it commands.
- `preview.py` checks the arm, joints, and action mapping before training.
- `train.py` declares the `train_franka_reach` PPO job.
- `eval.py` measures the hand's final distance from the goal.
- `play.py` plays a saved checkpoint with a camera and records the rollout.
- `.simuloignore` excludes local files from submitted packages.

The task reads the hand with `hand_position()` and sends Cartesian targets through
`simulo.DifferentialIKController`.

## Run it

### Preview

```bash
simulo run samples/franka-reach/preview.py
simulo recordings <job-id>
```

### Train

```bash
simulo run samples/franka-reach/train.py --num-envs 2048 --max-iterations 1500
simulo policy list
```

### Evaluate

```bash
simulo run samples/franka-reach/eval.py --policy <policy-id>:best
```

An episode succeeds when the hand finishes within 5 cm of its goal.

### Play

```bash
simulo run samples/franka-reach/play.py --policy <policy-id>:best
simulo recordings <job-id>
```

#### Try this

- Run `simulo run samples/franka-reach/preview.py --checks action-map` to inspect how three
  Cartesian action values move the arm's seven joints through inverse kinematics.
- Tighten the 5 cm radius in `eval.py` to 2 cm and evaluate the same policy without retraining.
  Restore the original rule, then compare checkpoints with
  `simulo run samples/franka-reach/eval.py --policy <policy-id>:best --compare <policy-id>:latest`.

## What to expect

_Measured on staging on October 6, 2026 with the published `simulo` 0.32.0 client. These are
measured runs, not a guarantee._

Preview (job `job_sandy-saturation-7b0qhd`) passed 39 of its 40 checks and warned on one.
Sweeping `panda_joint2` toward +1.41 rad, the joint stalled short of its target, with 1.046 rad
tail mean error. Settle, random moves, the training-loop checks, and all three Cartesian action
checks passed. Its recording was verified with 18,981 messages.

The default training command ran for 12.4 minutes on Tier 1 (job `job_misty-riff-01b1eb`).
Policy `policy_misty-riff-01b1eb` saved its best checkpoint at iteration 1400 and its latest
checkpoint at iteration 1500.

Evaluating the best checkpoint (job `job_beveled-modem-gn4dj8`) reported 100 of 100 episodes
succeeded (likely 96% to 100%): the hand finished within 5 cm of its goal every time.

Evaluate `best`, not `latest`. On this task the last checkpoint scores lower than the best one:
the latest checkpoint of the same policy (job `job_ambitious-kettle-ydx3b9`) reached the goal in
only 59 of 100 episodes.

Five more trainings changed the seed in `train.py` or the GPU tier: seed 42 on Tier 2, seeds 1
and 2 on Tier 1, and seeds 3 and 4 on Tier 2. The best checkpoint of every one scored 100 of 100.

Playing the best checkpoint (job `job_milky-apex-96sp2y`) completed 3 episodes and 717 steps
with mean reward 264.84 +/- 23.54. Its recording was verified with 5,747 messages. Playback warned
that `task.py` had changed since the policy was trained, because the policy came from a copy of
this file with the same code and different comments.

## Inspecting results

```bash
simulo jobs
simulo logs <job-id> --follow
simulo result <job-id>
simulo policy list
simulo recordings <job-id>
```

Use the training job's policy ID with either `:best` or `:latest`. Evaluation reports each
hand-to-goal check separately.

## Troubleshooting

If the preview reports a scene, joint, or action problem, follow the public
[scene troubleshooting guide](https://docs.simulo.ai/guides/troubleshoot-a-scene-with-preview/)
before training. When adapting the task, do not write joint targets separately from the inverse
kinematics controller because both would control the same joints.

## Extending it

Change the goal ranges or Cartesian step size in `FrankaReachTask`. You can also extend the
action with an orientation target, but update preview and evaluation so they still describe the
same task.

## Assets, licensing, attribution

The catalog robot is supplied by Simulo. The sample code is available under the
[MIT License](../../LICENSE).

Sample code: Copyright (c) 2026 Simulo LLC.
