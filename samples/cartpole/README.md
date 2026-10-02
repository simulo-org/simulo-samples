# Cartpole

## What this shows

A cart slides along a rail with a pole hinged on top, and a policy learns to keep the pole upright
by pushing the cart left or right. This is the core Simulo training loop, and every other training
sample in this repository has the same shape: a `simulo.Task` describes the problem,
`simulo.LearningEnv` runs thousands of copies of it in parallel on a GPU, `simulo.RLTrainer`
trains a PPO policy, and the training job saves the policy's checkpoints on its own.

You will learn the Task lifecycle (`build`, `on_start`, and the per-step methods), how a
training job is declared and sized, how a catalog robot is pinned to an exact version, how a
sample splits its task and its job across two files, and why a file that trains on a GPU can
still be imported on a laptop with no `torch` installed.

## Prerequisites

- Python 3.11 or newer and the Simulo client: `python -m pip install --upgrade simulo`.
- A Simulo account, signed in once with `simulo login`. Without a sign-in, `simulo run` writes a
  package to a `.simulo/` directory next to `train.py` and runs nothing.
- A clone of this repository. The commands below run from its root.
- No GPU on your machine. The job asks for a Tier 1 GPU (T4) in the Simulo cloud
  (`system=simulo.SystemType.TIER_1` on the job) and is billed to your account like any
  other job. Hardware describes the job's own request, not queue priority: it waits on
  the same shared GPU fleet as every other job.

## Assets

- `simulo/robot/cartpole:v1`: a catalog reference pinned to version 1. `simulo run` records the
  exact version with the job; nothing is downloaded to your machine.

## Files and APIs

- `task.py`: the application, the catalog robot, the reward kernel, and `CartpoleTask`. It
  declares no job.
- `train.py`: the one job you submit, `train`. It imports `app` and `CartpoleTask` from
  `task.py`.
- `preview.py`: a `preview` job that checks the robot inside `CartpoleTask` before you train.
  It makes no policy.
- `.simuloignore`: files `simulo run` leaves out of the uploaded package.

Simulo names it uses:

- `simulo.Asset.from_registry(...)`: the catalog robot handle.
- `simulo.Task` with `build`, `on_start`, `get_observations`, `get_rewards`, `get_dones`,
  `apply_actions`, and `reset_idx`. `get_dones` returns `(terminated, truncated)`: `terminated`
  is a dictionary mapping each early-end condition name to a boolean tensor, and `truncated` is
  one boolean tensor.
- `simulo.Scene`, `simulo.Terrain.plane`, `simulo.Light.dome`, `simulo.Robot`, `simulo.Pose`.
- `robot.state` for live joint positions and velocities; `robot.find_joints`,
  `robot.set_joint_effort_target`, `robot.set_joint_state`, `robot.reset`.
- `app.runtime.imports()` around `import torch` and `@app.runtime.torch_jit` on the reward
  kernel: both are inert on your machine and real in the cloud.
- `@app.job(type="train", ...)`: a training job, which saves the policy's `best` and `latest`
  checkpoints automatically. `retries=2` reruns it after a failure, starting from its own
  latest checkpoint.
- `simulo.LearningEnv`, `simulo.RLTrainer` (`train`, `close`).

## Run it

```bash
simulo login
simulo run samples/cartpole/train.py --num-envs 4096 --max-iterations 200
```

`--num-envs` and `--max-iterations` are `train`'s own parameters. For a quick check that
the job launches, use `--num-envs 64 --max-iterations 2`. Add `--detach` to submit without
waiting for the log. `simulo run samples/cartpole/task.py` is refused, because that file
declares no job.

The configured execution budget is eight hours total across the first attempt and up to two
retries; retries share that budget. Runtime preparation happens before that budget, so eight hours
is not a billing ceiling. Pressing Ctrl-C while following logs only detaches your terminal. Stop a
queued or running job explicitly with the id printed by `simulo run`:

```bash
simulo cancel <job-id>
```

Before a long training run you can preview the task. The preview job lets the cartpole settle,
sweeps its driven joint, runs random actions through the real training loop, and prints a report
with a recording. It makes no policy:

```bash
simulo run samples/cartpole/preview.py
```

### Evaluate

From this sample directory, use `simulo policy list` to find the policy id, then run:

```bash
simulo run eval.py --policy policy_<id>:best
```

The rule checks that the pole lasts to the time limit and finishes within 10 degrees of upright.

<!-- REPORT EXCERPT: filled after the GPU run -->
```text
```

## What to expect

`simulo run` uploads the application, creates a job, and follows its log. After the simulation
starts, the log shows an iteration counter and, every 50 iterations, a "Checkpoint written" line
and, when the mean episode reward improved, a "Best checkpoint saved" line with the reward. Reward rises
quickly on this task. A balanced, motionless step earns at most 1.0. At 60 actions per second for a
five-second episode, the undiscounted episode-return ceiling is about 300, so a `best_reward` in the
high two hundreds means the policy balanced for most of an episode. The example in
[Train on the catalog cartpole](https://docs.simulo.ai/guides/train-cartpole/) reports 294.65; that
figure is illustrative, and the exact numbers vary because GPU training is not bit-for-bit
reproducible even with the fixed seed. Treat it as a trend rather than a target.

The result holds training statistics such as `iterations` and `best_reward`, shown here
abbreviated:

```json
{"num_envs": 4096, "iterations": 200, "best_reward": 294.65}
```

The reward is illustrative.
A run takes about two minutes at the defaults once capacity is free; a first run can take longer while the cloud prepares the runtime.

## Inspecting results

```bash
simulo jobs                                  # status, job IDs, and each job's policy
simulo logs <job-id> --follow                # iteration and checkpoint lines
simulo result <job-id>                       # the returned dictionary
simulo policy list                           # your policies and their checkpoints
simulo policy get <policy-id>:best           # download the best checkpoint, digest-verified
simulo export <policy-id>:best               # convert the best checkpoint to ONNX
simulo cancel <job-id>                       # stop a queued or running job
```

Every training job makes one policy, whose ID is the job's ID with a `policy_` prefix in place
of `job_`. `simulo run` prints both. The policy holds at most two checkpoints: `latest`, the
most recent save, and `best`, the save with the highest mean episode reward. When a training
job completes, `simulo run` also prints which checkpoints it saved and the commands to try
next. A checkpoint is training state for Simulo; to run the policy elsewhere, export it.

`<job-id>` and `<policy-id>` are printed by `simulo run` and listed by `simulo jobs`.

## Troubleshooting

- `simulo run` prints "This wrote a local package only": you are not signed in. Run
  `simulo login` and submit again.
- The job stays `queued`: the cloud is waiting for GPU capacity. `simulo logs --follow` attaches
  when it starts.
- The first log lines do not appear immediately after the job starts: the simulation must boot
  first.
- `simulo result` says the job has not completed: results exist only for completed jobs.
- The policy has only a `latest` checkpoint after a very short run: a checkpoint is saved
  every 50 iterations and at the end of training, but a save becomes `best` only when an
  episode finished since the previous save. A training job that ends without saving any
  checkpoint is marked `failed` with the reason `no_checkpoint_saved`.
- The job failed: `simulo logs` prints the platform's reason code and detail after its header,
  even when the job produced no output.

## Extending it

- Tune the reward: the `rew_scale_*` class attributes on `CartpoleTask` weight the alive bonus,
  the termination penalty, and the pole and cart penalties.
- Change the episode: `episode_length_s`, `max_cart_pos`, and `initial_pole_angle_range` set how
  long an episode lasts, when the cart is out of bounds, and how far the pole starts from
  vertical.
- Keep training the same policy:
  `simulo run samples/cartpole/train.py --max-iterations 400 --from <policy-id>:best` starts a
  new job from that checkpoint, with `--max-iterations` read as the new total. The new job
  makes its own policy; the one you started from never changes.
- Watch it: add `--viewstream` to the run and open `simulo view`. Streaming slows training
  noticeably, so use it to look, not for routine runs.
- Add a second job, for example one that trains a variant of the task, as its own file next
  to `train.py` that imports from `task.py`. Each file holds exactly one job.

## Assets, licensing, attribution

The sample code is available under the [MIT License](../../LICENSE).

This sample installs no third-party packages.

### Attribution

Sample code: Copyright (c) 2026 Simulo LLC.
