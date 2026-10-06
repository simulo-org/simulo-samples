# Humanoid

## What this shows

Learn how a locomotion task is built and trained by teaching a 21-joint humanoid to walk forward
without falling. The focused lesson is a dense walking reward paired with observation
normalisation: the trainer keeps observations and value estimates on a steady scale as it learns,
which lets the default training learn a reliable gait.

## Prerequisites

- Python 3.11 or newer with the Simulo client installed.
- A Simulo account and `simulo login`.
- Run every command below from the repository root.

## Assets

The task uses the version-pinned catalog asset `simulo/robot/humanoid:v1`. You do not need to
publish an asset for this sample.

## Files and APIs

- `task.py` defines `HumanoidTask`, its 75-value observation, 21 effort actions, walking reward,
  fall condition, and play camera.
- `preview.py` checks the task before training and creates no policy.
- `train.py` declares the `train_humanoid` PPO job and its PPO settings.
- `eval.py` defines the evaluation job and the `walked_forward` success rule.
- `play.py` plays a saved checkpoint with a camera and records the rollout.
- `.simuloignore` excludes local files from submitted packages.

The task scales each action by its joint's gear ratio in `joint_gears` and applies it with
`robot.set_joint_effort_target(...)`. `train.py` passes its PPO settings to `simulo.RLTrainer`,
including skrl's `RunningStandardScaler` as the state and value preprocessor, which normalises
observations and value estimates during training.

The reward, observation, and PPO settings follow the standard humanoid walking recipe, with these
differences:

- Joint positions are measured from their defaults instead of scaled by joint limits, and there
  is no joint-limit cost, because the robot API does not expose joint limits.
- The ground keeps the default friction of 0.5; the recipe uses 1.0.
- The policy and value use separate, smaller networks, and each update collects 16 rollouts
  instead of 32.
- Training runs 1,024 environments for 2,500 iterations instead of 4,096 environments.

## Run it

### Preview

```bash
simulo run samples/humanoid/preview.py
simulo recordings <job-id>
```

The preview checks the scene, action mapping, and training loop on 16 environments without
creating a policy.

### Train

```bash
simulo run samples/humanoid/train.py
simulo policy list
```

The defaults are 1,024 environments, 2,500 iterations, and seed 42. The training job asks for a
Tier 2 GPU (A10G). Use the policy ID printed by training in the remaining commands.

### Evaluate

```bash
simulo run samples/humanoid/eval.py --policy <policy-id>:best
```

An episode succeeds when the robot stays upright to the 15-second time limit and ends at least
1 m forward of its start.

### Play

```bash
simulo run samples/humanoid/play.py --policy <policy-id>:best
simulo recordings <job-id>
```

#### Try this

- Train an independent run with another seed, for example
  `simulo run samples/humanoid/train.py --seed 1`, and evaluate its `best` checkpoint.
- Compare the saved checkpoints with
  `simulo run samples/humanoid/eval.py --policy <policy-id>:best --compare <policy-id>:latest`.

## What to expect

Staging runs on 2026-10-06 with `simulo 0.32.0` on a Tier 2 GPU produced the training and
evaluation results below. These are measured runs, not a guarantee that every run learns to walk.

The default training command ran for 17.2 minutes (`job_fortunate-fermata-new9vp`) and saved its
best checkpoint at iteration 2,050 of 2,500.

Evaluation of `best` reported `99 of 100 episodes succeeded` (`job_deft-journal-scvjjx`). One
robot fell at 14.67 seconds, and all 100 moved at least 1 m forward. `latest` scored 100 of 100
(`job_lavender-mile-7pbwme`). The commands above evaluate `best`, the checkpoint with the
highest training reward.

Three more trainings with seeds 1, 2, and 3 also learned to walk. Their best checkpoints scored
96, 100, and 100 of 100.

Preview and play ran on staging on 2026-10-06 with `simulo 0.32.0`.

Preview (job `job_matching-cellulose-yz6qbb`) passed with 64 warnings and no failures. The
warnings say that some joints kept moving with zero actions and that no joint held a position
target. Every joint on this robot is torque-driven, so none can hold a pose on its own, which is
expected for this robot. Random moves and every training-loop check passed, including all 21
action checks. Its recording was verified with 18,017 messages.

Playing the best checkpoint (job `job_sociable-factorial-zqk9yp`) completed 3 episodes and 2,697
steps, with a mean episode reward of about 9,100 that varied by about 100. Its recording was
verified with 21,587 messages.

## Inspecting results

```bash
simulo jobs
simulo logs <job-id> --follow
simulo result <job-id>
simulo policy list
simulo policy get <policy-id>:best
simulo recordings <job-id>
```

`simulo export` cannot convert this policy yet, because the policy carries the observation
normalisation it learned during training. Use `simulo policy get` to download the checkpoint, and
`eval.py` and `play.py` to run it on Simulo.

## Troubleshooting

If a preview check fails, use the public
[scene troubleshooting guide](https://docs.simulo.ai/guides/troubleshoot-a-scene-with-preview/)
before spending time on training. A training job that stays queued is waiting for GPU capacity.
If `simulo run` only writes a local package, run `simulo login` and submit again.

## Extending it

Adjust the reward weights on `HumanoidTask`, such as `heading_weight`, `up_weight`, and
`energy_cost_scale`, or the fall height in `termination_height`, then preview the task again
before retraining. Keep evaluation independent by reading the robot's pose in `eval.py`.

## Assets, licensing, attribution

The catalog robot is supplied by Simulo. The sample code is available under the
[MIT License](../../LICENSE).

Sample code: Copyright (c) 2026 Simulo LLC.
