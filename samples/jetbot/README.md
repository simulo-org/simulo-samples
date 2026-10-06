# JetBot

## What this shows

Learn differential-drive control by training a two-wheeled robot to follow a changing commanded
direction. The focused lesson is the direct mapping from two policy actions to left and right
wheel velocities.

## Prerequisites

- Python 3.11 or newer with the Simulo client installed.
- A Simulo account and `simulo login`.
- Run every command below from the repository root.

## Assets

The task uses the version-pinned catalog asset `simulo/robot/jetbot:v1`. The job fetches the
robot from the catalog when it starts; you do not need to publish it.

## Files and APIs

- `task.py` defines `JetbotTask`, command sampling, wheel actions, and the reward.
- `preview.py` checks both driven wheels and the training loop before training.
- `train.py` declares the `train_jetbot` PPO job.
- `eval.py` checks motion in the commanded direction.
- `play.py` plays a saved checkpoint with a camera and records the rollout.
- `.simuloignore` excludes local files from submitted packages.

The task reads `robot.state` and writes wheel targets with
`robot.set_joint_velocity_target(...)`.

## Run it

### Preview

```bash
simulo run samples/jetbot/preview.py
simulo recordings <job-id>
```

### Train

```bash
simulo run samples/jetbot/train.py
simulo policy list
```

### Evaluate

```bash
simulo run samples/jetbot/eval.py --policy <policy-id>:best
```

The success rule uses the robot's velocity and current command.

### Play

```bash
simulo run samples/jetbot/play.py --policy <policy-id>:best --episodes 3
simulo recordings <job-id>
```

#### Try this

- Run `simulo run samples/jetbot/preview.py --checks joint-sweep,action-map` to see which action
  drives each wheel.
- Compare the saved checkpoints with
  `simulo run samples/jetbot/eval.py --policy <policy-id>:best --compare <policy-id>:latest`.

## What to expect

One staging run on 2026-10-06 with `simulo 0.32.0` produced the preview, training, and
evaluation results below. Preview (job `job_convex-phase-7c2hv8`) passed all checks.

The default training command (job `job_clever-glacier-9nz091`) created 16 environments, matching
the evaluation run's capacity, and ran for about 4 minutes. The returned best reward was about
335; both `best` and `latest` were saved at iteration 700 in `policy_clever-glacier-9nz091`.

Evaluation at the training spacing (job `job_closed-resolution-4bwe8n`) reported
`100 of 100 episodes succeeded`, with all episodes moving at least `0.1 m/s` in the commanded
direction. The comparison (job `job_favorite-gallery-5w2m7a`) found no clear difference: `best`
and `latest` were each 100 of 100, with zero changed outcomes.

A second run of the default training command on 2026-10-06 (job `job_district-lintel-ghnqfs`)
saved the same best reward at the same iteration. Playing its best checkpoint
(job `job_mighty-stager-6kb9r3`) completed 3 episodes and 897 steps with mean reward
`267.20 +/- 27.43`. Its recording was verified with 7,187 messages.

An earlier staging run on 2026-10-05, also with `simulo 0.32.0`, trained its own policy
(job `job_eminent-commit-26xr5e`). Its focused preview (job `job_nippy-command-czp2bg`)
confirmed action 0 drove the left wheel and action 1 drove the right wheel; both wheel sweeps
passed. Its full preview recording (job `job_sapphire-texture-tgrvnp`) verified as 7,846 MCAP
messages and included Lichtblick and Foxglove layout files.

## Inspecting results

```bash
simulo jobs
simulo logs <job-id> --follow
simulo result <job-id>
simulo policy list
simulo recordings <job-id>
```

The evaluation report shows whether final motion agrees with each episode's command. Use the
recording to distinguish heading errors from a wheel-action mapping problem.

## Troubleshooting

Use the public
[scene troubleshooting guide](https://docs.simulo.ai/guides/troubleshoot-a-scene-with-preview/)
when preview reports a joint or action failure. A quiet job may still be preparing its catalog
asset before the simulation starts.

## Extending it

Change how commands are sampled or adjust `velocity_scale`, then rerun preview before training.
Keep the success rule based on independent robot state and the task's command.

## Assets, licensing, attribution

The catalog robot is supplied by Simulo. The sample code is available under the
[MIT License](../../LICENSE).

Sample code: Copyright (c) 2026 Simulo LLC.
