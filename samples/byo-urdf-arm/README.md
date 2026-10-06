# Bring your own URDF

## What this shows

Publish a three-joint arm from a URDF, then train a policy to move its joints to a randomly
sampled target. The arm's three actions control `shoulder_pan`, `shoulder_lift`, and `elbow`.
This sample keeps the publishing step and the policy lifecycle together: preview the task, train,
evaluate a saved policy, and play it back as a recording.

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

_Measured in one staging run on October 5, 2026 (America/Los_Angeles) with the published
`simulo` 0.32.0 client._

Preview passed settle, all nine joint sweeps, random moves, training-loop checks, and all three
action checks. The runtime reported that it applied the task actuator gains to all joints. The
recording was verified with 11,598 messages. The focused `--checks joint-sweep,action-map`
experiment against the newly published bundled asset also passed: action 0 moved
`shoulder_pan`, action 1 moved `shoulder_lift`, and action 2 moved `elbow`.

Training completed in 427 seconds. Policy `policy_black-relaxation-wwzw1d` saved its best
checkpoint at iteration 700 with reward -18.3558 and its latest checkpoint at iteration 1200.

The best checkpoint succeeded in 70 of 100 episodes (likely 60% to 79%): every joint was within
0.1 rad of its target in 70 of 100, while the arm settled below 0.1 rad/s in 100 of 100. Loosening
the target tolerance to 0.2 rad raised success to 97 of 100 (likely 91% to 99%), with settling
still 100 of 100.

Playback completed 3 episodes and 897 steps with mean reward -11.50 +/- 4.70. Its recording was
verified with 7,187 messages.

The settling check passes whenever the arm stops; a policy that does nothing also settles. The
meaningful result is whether the policy reaches its target as well as settling.

The task's joint stiffness and damping are applied to the robot when the job runs, and preview
reports it.

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
`robot/byo-urdf-arm:v1`. If preview cannot find the three named joints after you replace the
URDF, update the names in `ByoArmTask.on_start` and keep the observation and action dimensions in
sync with the robot.

## Extending it

Replace the URDF and meshes with your own robot, publish a new catalog version, then update the
asset reference and joint names in `task.py`. You can also change `target_amplitude` or the reward
weights to explore a different reaching task.

## Assets, licensing, attribution

The sample code and asset files are available under the [MIT License](../../LICENSE).

Sample code and assets: Copyright (c) 2026 Simulo LLC.
