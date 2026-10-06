# Cartpole

## What this shows

Learn the basic policy lifecycle by training a cart to keep a hinged pole upright. The focused
lesson is how one `simulo.Task` is previewed, trained, evaluated with a task-specific success
rule, and played back from a saved policy.

## Prerequisites

- Python 3.11 or newer with the Simulo client installed.
- A Simulo account and `simulo login`.
- Run every command below from the repository root.

## Assets

The task uses the version-pinned catalog asset `simulo/robot/cartpole:v1`. You do not need to
publish an asset for this sample.

## Files and APIs

- `task.py` defines `CartpoleTask`, its observations, actions, reward, resets, and camera.
- `preview.py` checks the task before training and creates no policy.
- `train.py` declares the `train` PPO job.
- `eval.py` defines the evaluation job and `balanced` success rule.
- `play.py` plays one saved checkpoint and records the rollout.
- `.simuloignore` excludes local files from submitted packages.

The main APIs are `simulo.preview`, `simulo.LearningEnv`, `simulo.RLTrainer`,
`simulo.evaluate`, and `simulo.RLPlayer`.

## Run it

### Preview

```bash
simulo run samples/cartpole/preview.py
simulo recordings <job-id>
```

The preview checks the scene, action mapping, and training loop without creating a policy.

### Train

```bash
simulo run samples/cartpole/train.py --num-envs 4096 --max-iterations 200
simulo policy list
```

Use the policy ID printed by training in the remaining commands.

### Evaluate

```bash
simulo run samples/cartpole/eval.py --policy <policy-id>:best
```

An episode succeeds when the pole remains within 10 degrees of upright until the time limit.

### Play

```bash
simulo run samples/cartpole/play.py --policy <policy-id>:best
simulo recordings <job-id>
```

#### Try this

- Run `simulo run samples/cartpole/preview.py --checks action-map` to isolate how the cart action
  moves the driven joint while the pole remains passive.
- Change the 10-degree limit in `eval.py` to 5 degrees and evaluate the same policy again without
  retraining. Restore the original rule, then compare checkpoints with
  `simulo run samples/cartpole/eval.py --policy <policy-id>:best --compare <policy-id>:latest`.

## What to expect

One staging run on 2026-10-05 with `simulo 0.32.0` produced the following results. Preview
passed, with expected warnings that the passive pole was still swinging and was not driven. Its
recording verified as 11,576 MCAP messages and included Lichtblick and Foxglove layout files.

Training ran for 94.580 seconds. The returned best reward was `295.0882`; both `best` and
`latest` were saved at iteration 200 in `policy_modern-pepato-75dpmx`.

The standard evaluation reported `100 of 100 episodes succeeded`, with both the time-limit and
10-degree checks at 100 of 100. Tightening the rule to 5 degrees also produced 100 of 100. The
restored 10-degree comparison found no clear difference: `best` and `latest` were each 100 of
100, with zero changed outcomes.

Playback completed 3 episodes and 897 steps with mean reward `294.89 +/- 3.44`. The verified
MCAP contained 7,187 messages, including 897 packets on each overhead-camera video channel.

## Inspecting results

```bash
simulo jobs
simulo logs <job-id> --follow
simulo result <job-id>
simulo policy list
simulo recordings <job-id>
```

Training creates one policy with `best` and `latest` checkpoint names when those checkpoints are
available. Preview and play recordings use the job ID printed by `simulo run`.

## Troubleshooting

If a preview check fails, use the public
[scene troubleshooting guide](https://docs.simulo.ai/guides/troubleshoot-a-scene-with-preview/)
to inspect the failure before spending time on training. If `simulo run` only writes a local
package, run `simulo login` and submit again.

## Extending it

Adjust the reward weights or initial pole angle in `CartpoleTask`, then preview the task again
before retraining. Keep evaluation independent by reading the robot's final state in `eval.py`.

## Assets, licensing, attribution

The catalog robot is supplied by Simulo. The sample code is available under the
[MIT License](../../LICENSE).

Sample code: Copyright (c) 2026 Simulo LLC.
