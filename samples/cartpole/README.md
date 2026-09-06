# Cartpole

## What this shows

A cart slides along a rail with a pole hinged on top, and a policy learns to keep the pole upright
by pushing the cart left or right. This is the core Simulo training loop, and every other training
sample in this repository has the same shape: a `simulo.Task` describes the problem,
`simulo.LearningEnv` runs thousands of copies of it in parallel on a GPU, `simulo.RLTrainer`
trains a PPO policy, and the checkpoint lands in a durable `simulo.Volume`.

You will learn the Task lifecycle (`build`, `on_start`, and the per-step methods), how a
training job is declared and sized, how a catalog robot is pinned to an exact version, and why a
file that trains on a GPU can still be imported on a laptop with no `torch` installed.

## Prerequisites

- Python 3.11 or newer and the Simulo client: `python -m pip install "simulo>=0.23.1,<0.25"`.
- A Simulo account, signed in once with `simulo login`. Without a sign-in, `simulo run` writes a
  package to a `.simulo/` directory next to `app.py` and runs nothing.
- A clone of this repository. The commands below run from its root.
- No GPU on your machine. The job asks for an L4-class GPU in the Simulo cloud (`gpu="L4"` on the
  job) and is billed to your account like any other job.

## Assets

- `simulo/robot/cartpole:v1`: a catalog reference pinned to version 1. `simulo run` records the
  exact version with the job; nothing is downloaded to your machine.

## Files and APIs

- `app.py`: the task, the reward kernel, and the `train_cartpole` job.
- `.simuloignore`: files `simulo run` leaves out of the uploaded package.

Simulo names it uses:

- `simulo.Asset.from_registry(...)`: the catalog robot handle.
- `simulo.Volume.from_name(...)` and `App(mounts=...)`: a durable volume for the checkpoint. Its
  real directory is `vol.path`, which resolves only inside the running job.
- `simulo.Task` with `build`, `on_start`, `get_observations`, `get_rewards`, `get_dones`,
  `apply_actions`, and `reset_idx`.
- `simulo.Scene`, `simulo.Terrain.plane`, `simulo.Light.dome`, `simulo.Robot`, `simulo.Pose`.
- `robot.state` for live joint positions and velocities; `robot.find_joints`,
  `robot.set_joint_effort_target`, `robot.set_joint_state`, `robot.reset`.
- `app.runtime.imports()` around `import torch` and `@app.runtime.torch_jit` on the reward
  kernel: both are inert on your machine and real in the cloud.
- `simulo.LearningEnv`, `simulo.RLTrainer` (`train`, `save`, `close`).
- `simulo.callbacks.ResumableCheckpoint(every=50)` on the job, with `retries=2`.

## Run it

```bash
simulo login
simulo run samples/cartpole/app.py --num-envs 4096 --max-iterations 200
```

`--num-envs` and `--max-iterations` are `train_cartpole`'s own parameters. For a quick check that
the job launches, use `--num-envs 64 --max-iterations 2`. Add `--detach` to submit without
waiting for the log.

The configured execution budget is eight hours total across the first attempt and up to two
retries; retries share that budget. Runtime preparation happens before that budget, so eight hours
is not a billing ceiling. Pressing Ctrl-C while following logs only detaches your terminal. Stop a
queued or running job explicitly with the id printed by `simulo run`:

```bash
simulo cancel <job-id>
```

## What to expect

`simulo run` uploads the application, creates a job, and follows its log. After the simulation
starts, the log shows an iteration counter and, every 50 iterations, a checkpoint line and, when
the mean episode reward improved, a "Best checkpoint saved" line with the reward. Reward rises
quickly on this task. A balanced, motionless step earns at most 1.0. At 60 actions per second for a
five-second episode, the undiscounted episode-return ceiling is about 300, so a `best_reward` in the
high two hundreds means the policy balanced for most of an episode. The example in
[Train on the catalog cartpole](https://docs.simulo.ai/guides/train-cartpole/) reports 294.65; that
figure is illustrative, and the exact numbers vary because GPU training is not bit-for-bit
reproducible even with the fixed seed. Treat it as a trend rather than a target.

The result names the checkpoint the job saved into its volume, plus statistics such as
`iterations` and `best_reward`:

```json
{"checkpoint": "<runtime-volume-path>/cartpole_final.pt", "num_envs": 4096, "iterations": 200, "best_reward": 294.65}
```

The `checkpoint` prefix above stands for the job-resolved `vol.path`; the application mount
label `/out` is not a runtime path and will not appear in the result. The reward is illustrative.
A run takes about two minutes at the defaults once capacity is free; a first run can take longer while the cloud prepares the runtime.

## Inspecting results

```bash
simulo jobs                           # status and job IDs
simulo logs <job-id> --follow         # iteration and checkpoint lines
simulo result <job-id>                # the returned dictionary
simulo models <job-id>                # best.pt and latest.pt
simulo models <job-id> best.pt        # download one, digest-verified
simulo export <job-id>                # convert the best checkpoint to ONNX
simulo cancel <job-id>                # stop a queued or running job
```

The `checkpoint` path in the result is inside the job's volume, which later jobs of the same
application can read; it is not a file on your machine. The files you download come from
`ResumableCheckpoint`, which uploads `best.pt` (the highest-reward policy seen) and `latest.pt`
(for resuming) as the run's models.

`<job-id>` is printed by `simulo run` and listed by `simulo jobs`.

## Troubleshooting

- `simulo run` prints "This wrote a local package only": you are not signed in. Run
  `simulo login` and submit again.
- The job stays `queued`: the cloud is waiting for GPU capacity. `simulo logs --follow` attaches
  when it starts.
- The first log lines do not appear immediately after the job starts: the simulation must boot
  first.
- `simulo result` says the job has not completed: results exist only for completed jobs.
- `simulo models` lists nothing for a very short run: a checkpoint is written every 50
  iterations and at the end of training, so a two-iteration smoke run still uploads
  `latest.pt`, but a run that was cancelled before its first checkpoint has nothing to list.
- The job failed: `simulo logs` prints the platform's reason code and detail after its header,
  even when the job produced no output.

## Extending it

- Tune the reward: the `rew_scale_*` class attributes on `CartpoleTask` weight the alive bonus,
  the termination penalty, and the pole and cart penalties.
- Change the episode: `episode_length_s`, `max_cart_pos`, and `initial_pole_angle_range` set how
  long an episode lasts, when the cart is out of bounds, and how far the pole starts from
  vertical.
- Keep training the same policy:
  `simulo run samples/cartpole/app.py --max-iterations 400 --from <job-id>` continues from a
  finished job's best checkpoint, with `--max-iterations` read as the new total.
- Watch it: add `--viewstream` to the run and open `simulo view`. Streaming slows training
  noticeably, so use it to look, not for routine runs.
- Carry the policy through evaluation and a recorded rollout: see
  [Cartpole Eval](../cartpole-eval/).

## Assets, licensing, attribution

The sample code is available under the [MIT License](../../LICENSE).

This sample installs no third-party packages.

### Attribution

Sample code: Copyright (c) 2026 Simulo LLC.
