# Bring your own URDF

## What this shows

Publish a three-joint arm from a URDF, then train a policy to move its joints to a randomly
sampled target. This sample keeps the publishing step and the policy lifecycle together: preview
the task, train, evaluate a saved policy, and play it back as a recording.

The policy has one action for each joint: `shoulder_pan`, `shoulder_lift`, and `elbow`, in that
order. Each action is a value from -1 to 1 that names the angle its joint should hold.
`apply_actions` in `task.py` clamps the value to that range and multiplies it by
`target_amplitude`, which is 1.0 rad. An action of -1 holds the joint at -1.0 rad, 0 holds it at
zero, and 1 holds it at +1.0 rad. Every joint uses the same mapping, centred on the zero angle
defined in the URDF. The policy reaches a target by naming it rather than nudging the joint in
small steps.

## Prerequisites

- Python 3.11 or newer with the Simulo client installed.
- A Simulo account and `simulo login`.
- Run the commands below from this repository's root.

## Assets

- `assets/robot/byo-urdf-arm/` contains the URDF and four meshes to publish.
- `robot/byo-urdf-arm:v1` is the organization-scoped catalog reference the jobs use.

Publish and inspect the arm before submitting a job:

```bash
simulo asset publish assets/robot/byo-urdf-arm --kind robot --name byo-urdf-arm --entry robot.urdf
simulo asset inspect robot/byo-urdf-arm:v1
```

If `simulo asset inspect robot/byo-urdf-arm:v1` already shows the arm, skip the publish.
Publishing again creates `:v2`, and the jobs in this sample keep using `:v1`.

The reference has no publisher segment, so it resolves in the organization you are signed in to.

## Files and APIs

- `task.py` defines `ByoArmTask`, the three-joint observation and action mapping, and its reward.
- `preview.py`, `train.py`, `eval.py`, and `play.py` declare each stage of the lifecycle.
- `simulo.Asset.from_registry`, `simulo.LearningEnv`, and `simulo.RLTrainer` connect the catalog
  robot, task, and policy training job.

## Run it

### Preview

```bash
simulo run samples/byo-urdf-arm/preview.py
simulo recordings <job-id>
```

### Train

```bash
simulo run samples/byo-urdf-arm/train.py --num-envs 256 --max-iterations 1200
simulo policy list
```

### Evaluate

```bash
simulo run samples/byo-urdf-arm/eval.py --policy <policy-id>:best
```

### Play

```bash
simulo run samples/byo-urdf-arm/play.py --policy <policy-id>:best
simulo recordings <job-id>
```

#### Try this

- Loosen the target-accuracy tolerance in `eval.py` from `0.1` to `0.2` radians, then evaluate
  the same policy again without retraining.
- After swapping in your own URDF and publishing it, run
  `simulo run samples/byo-urdf-arm/preview.py --checks joint-sweep,action-map` to inspect its
  joints and action mapping before training.

## What to expect

_Measured on staging on October 6, 2026 with the published `simulo` 0.32.0 client. These are
measured runs, not a guarantee._

Preview (job `job_affable-bound-ca7yb4`) passed all 22 checks: settle, nine joint sweeps, random
moves, eight training-loop checks, and three action checks. Action 0 moved `shoulder_pan`, action
1 moved `shoulder_lift`, and action 2 moved `elbow`. The runtime reported that it applied the
task's actuator gains to all joints. Its recording was verified with 10,950 messages.

The default training command ran for about 7 minutes on Tier 1. Policy `policy_novel-credits-a1h2j1`
saved its best checkpoint at iteration 700 with reward -4.64 and its latest checkpoint at
iteration 1200.

Evaluating the best checkpoint (job `job_humorous-turntable-7j79qd`) reported 98 of 100 episodes
succeeded (likely 92% to 100%). In those 98, every joint was within 0.1 rad of its target and the
arm settled below 0.1 rad/s. All 100 episodes settled; the two failures missed only the target.

Three more trainings changed the seed in `train.py`. Seed 1 and seed 3 ran on Tier 2 and both
scored 97 of 100. Seed 2 ran on Tier 1 and scored 80 of 100. Every failure in those runs missed
only the target; the arm settled in every episode.

Playing the best checkpoint (job `job_humorous-bounce-z5axwe`) completed 3 episodes and 897 steps.
Mean reward varied between about 12 and 17 across two measured runs (2026-10-06). Its recording
was verified with 7,187 messages.

The settling check passes whenever the arm stops; a policy that does nothing also settles. The
meaningful result is whether the policy reaches its target as well as settling.

## Inspecting results

```bash
simulo jobs
simulo logs <job-id> --follow
simulo result <job-id>
simulo policy list
simulo recordings <job-id>
```

Use the training job's policy ID with either `:best` or `:latest` when evaluating or playing it.

## Troubleshooting

If the arm cannot be resolved, publish it in the organization you are signed in to and inspect
`robot/byo-urdf-arm:v1`. If preview cannot find a joint after you replace the URDF, the joint
names in the code do not match your robot; see the next section.

## Extending it

To train your own arm, replace the URDF and meshes with your robot, publish a new catalog
version, and update the asset reference in `task.py`. Then change these to match your robot:

- The joint names. They appear twice: in `ByoArmTask.on_start` in `task.py`, which sets the
  joints the actions drive, and in `completed_stably` in `eval.py`, which scores them.
- `target_amplitude` in `ByoArmTask`, if any joint cannot move 1 rad each way from zero. The same
  value sets both the action range and the targets sampled each episode, so lower it to fit the
  narrowest joint. The mapping is centred on zero, so a joint with a one-sided range, such as 0 to
  1.5 rad, also needs an offset added in `apply_actions` and `_randomize_target`.
- `observation_dim`, `action_dim`, and the joint count of 3 in `on_start` and
  `_randomize_target`, if your robot has a different number of joints.

You can also change the reward weights in `ByoArmTask` to explore a different reaching task.

## Assets, licensing, attribution

The sample code and asset files are available under the [MIT License](../../LICENSE).

Sample code and assets: Copyright (c) 2026 Simulo LLC.
