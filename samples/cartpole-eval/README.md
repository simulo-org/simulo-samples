# Cartpole Eval

## What this shows

The [Cartpole](../cartpole/) sample trains a policy and stops. This sample carries a policy through its whole
life in one application, with three jobs that share named volumes and run in order:

1. `train` runs a short PPO training and saves two files: the trainer's own checkpoint and a
   standalone TorchScript policy exported with `RLTrainer.export_policy`.
2. `evaluate` reloads the checkpoint, replays it for several rounds of several episodes, and
   reduces the rounds to reward percentiles and a success rate.
3. `rollout` plays the exported policy with `simulo.RLPlayer`, with no trainer involved, and
   records every step, plus a side-view camera video, to an MCAP flight recording you download.

You will learn how several jobs share volumes, the difference between a trainer checkpoint and an
exported policy, how a rollout is recorded, and how to attach a camera so the recording carries
video.

## Prerequisites

- Python 3.11 or newer and the Simulo client: `python -m pip install "simulo>=0.26.0,<0.27"`.
- A Simulo account, signed in once with `simulo login`.
- A clone of this repository. The commands below run from its root.
- No GPU on your machine. All three jobs ask for a Tier 1 GPU (T4) in the Simulo cloud and
  are billed to your account. Hardware describes the job's own request, not queue
  priority: it waits on the same shared GPU fleet as every other job.
- To open the recording: Foxglove or Lichtblick, both free desktop applications, or the `mcap`
  Python package, which the Simulo client already depends on.

## Assets

- `simulo/robot/cartpole:v1`: a catalog reference pinned to version 1.

## Files and APIs

- `app.py`: the task (with an optional camera), the reward kernel, a shared environment factory,
  and the three jobs.
- `.simuloignore`: files `simulo run` leaves out of the uploaded package.

Simulo names it uses, beyond those in the [Cartpole](../cartpole/) sample:

- Two volumes, `cartpole-eval-checkpoints` and `cartpole-eval-reports`, mounted on one `App`.
- `RLTrainer.export_policy(path)`: writes the deterministic policy as a standalone TorchScript
  file. `RLTrainer.evaluate(checkpoint=..., num_episodes=...)`: replays the trainer checkpoint.
- `simulo.RLPlayer(env=..., checkpoint=...)` and `player.play(num_steps=..., record=...)`.
- `simulo.RecordConfig(output_path=..., include_video=True, video_fps=30, ...)`.
- `simulo.Camera`, `simulo.SensorOffset.look_at(...)`, `simulo.CameraSpawnConfig`, and
  `robot.add_sensor(..., attach_to="slider/side_cam")`, plus `enable_cameras=True` on
  `LearningEnv` for the rollout job.
- `simulo.Entity.primitive.cuboid`, `simulo.Material.surface`, and `simulo.Visual.sphere` for the
  decorative prop and the marker that `on_post_physics_step` moves each step.

## Run it

```bash
simulo login
simulo run samples/cartpole-eval/app.py --job train --num-envs 512 --max-iterations 20
simulo run samples/cartpole-eval/app.py --job evaluate --num-episodes 10 --num-rounds 5
simulo run samples/cartpole-eval/app.py --job rollout --num-steps 200
```

Run each stage after the previous one completes. `--job` is required because the file declares
three jobs; `simulo run samples/cartpole-eval/app.py -h` lists them, and
`--job train -h` lists one job's own flags.

The configured execution budgets are eight hours for `train`, shared across its first attempt and
up to two retries, two hours for `evaluate`, and one hour for `rollout`: 11 graphics-accelerated
execution hours for the full sequence. Runtime preparation happens before those budgets, so this is
not a billing ceiling. Pressing Ctrl-C while following logs only detaches your terminal. Stop any
queued or running stage explicitly with its job id:

```bash
simulo cancel <job-id>
```

## What to expect

`train` is deliberately short (20 iterations). Its result names both files it saved into the
checkpoint volume, `checkpoint` and `policy`, alongside `best_reward` and `iterations`. The
complete three-job sequence takes about two minutes once capacity is free; a first run
can take longer while the cloud prepares the runtime.

`evaluate` returns the aggregated report and also writes it to `eval_report.json` in the reports
volume. The fields below are the statistics the job returns (alongside the `checkpoint` and
`report` paths); the numbers are illustrative:

```json
{
  "report": "<reports-volume-path>/eval_report.json",
  "checkpoint": "<checkpoints-volume-path>/cartpole_eval_final.pt",
  "num_rounds": 5,
  "num_episodes": 10,
  "reward_target": 75.0,
  "mean_reward": 68.2,
  "std_reward": 4.1,
  "reward_p25": 65.0,
  "reward_p50": 68.9,
  "reward_p75": 71.3,
  "mean_length": 240.5,
  "success_rate": 0.2
}
```

The two path prefixes above are job-resolved volume paths, not paths on your machine. The
success rate is the fraction of rounds whose mean reward cleared `reward_target`. The JSON numbers
are illustrative rather than a distribution to expect, and the default target of 75 is a starting
point to tune. With five rounds, `success_rate` can be `0.0`, `0.2`, `0.4`, `0.6`, `0.8`, or
`1.0`.

`rollout` plays 200 steps in a single environment and returns playback statistics, including
`messages_written` and `recording_complete`, alongside the recording path. Its recording is the
file you download next.

## Inspecting results

```bash
simulo jobs                  # all three jobs, most recent first
simulo logs --follow         # the stage currently running
simulo result                # the most recent stage's returned dictionary
simulo result <job-id>       # a specific stage
simulo recordings            # download rollout.mcap from the rollout job, verified
simulo models <train-job-id> # best.pt and latest.pt from the training stage
simulo cancel <job-id>       # stop a queued or running stage
```

Open `rollout.mcap` in Foxglove or Lichtblick. Add an Image panel on
`/sensors/camera/side_cam/video_foxglove` to watch the cart and pole, and a Plot panel on
`/reward.total` to see the reward per step. The JSON reports stay in the reports volume, but the
same numbers are in each job's result.

## Troubleshooting

- `simulo run` refuses with "declares no @app.entrypoint and 3 jobs": pass `--job NAME`.
- `evaluate` or `rollout` fails with "Checkpoint not found": run `--job train` first. Both read
  files that only `train` writes, and the volumes start empty.
- `success_rate` is `0.0` or `1.0`: either extreme can occur and is not by itself a broken
  evaluation. See [What to expect](#what-to-expect); with five rounds, the rate changes in
  increments of 0.2.
- The recording has no video: the camera must be attached to the robot before
  `scene.add(self.robot, ...)`, as `build` does. A camera attached afterwards is never seen and the
  recorder reports `VIDEO_NO_CAMERA_FOUND`.
- `simulo recordings` says the job has several recordings: name one, or pass `--all`.
- The job stays `queued`: the cloud is waiting for GPU capacity.

## Extending it

- Train for longer: raise `--max-iterations` on `train`; the other two stages pick up whatever the
  checkpoint volume holds.
- Move the evaluation bar: give the app a runtime with an environment layer, preserving its two
  existing mounts:

  ```python
  app = simulo.App(
      "cartpole-eval",
      mounts={"/checkpoints": checkpoints, "/reports": reports},
      runtime=simulo.Runtime.from_registry("simulo/gpu-rl:2026.06").env(
          {"SIMULO_EVAL_REWARD_TARGET": "60"}
      ),
  )
  ```

  `evaluate` then scores rounds against 60 instead of 75. See
  [Runtimes](https://docs.simulo.ai/concepts/runtimes/).
- Record more: `RecordConfig` accepts a `profile` and video settings; `num_steps` on `rollout`
  sets how long the recording is.
- Change the shot: `SensorOffset.look_at(pos=..., target=...)` in `build` places the camera.

## Assets, licensing, attribution

The sample code is available under the [MIT License](../../LICENSE).

This sample installs no third-party packages. `numpy`, which `evaluate` imports, ships with the
Simulo runtime.

### Attribution

Sample code: Copyright (c) 2026 Simulo LLC.
