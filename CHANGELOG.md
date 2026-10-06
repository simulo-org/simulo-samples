# Changelog

All notable changes to this repository are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Releases are
tagged by calendar version, `vYYYY.MM.N`, because this repository publishes samples rather
than an interface: there is no API here whose compatibility a semantic version could describe.

## [Unreleased]

### Added

- `simulo install samples` as a documented way to get this repository, alongside
  `git clone`, with options to choose a destination and a Git branch, tag, or commit.
- `preview.py`, `eval.py`, and `play.py` beside `train.py` in every sample. Preview checks the
  task before training, evaluation scores a saved checkpoint against a sample-specific success
  rule, and play records a trained policy.
- Document the preview, train, evaluate, and play lifecycle.

### Changed

- Every sample uses the current client's one-job-per-file layout. Each file declares one job, and
  you submit the training job with `simulo run samples/<slug>/train.py`; `--job` no longer exists.
  Every sample keeps its application and task in `task.py`, which the other files import. The
  Cartpole training job is now named `train`.
- Training jobs save their policy's `best` and `latest` checkpoints automatically, so the
  samples no longer declare checkpoint volumes. The READMEs use `simulo policy list`,
  `simulo policy get <policy-id>:best`, and `--policy <policy-id>:best` in place of
  `simulo models` and `--from <job-id>`.
- Every task returns named end conditions as the dictionary half of `get_dones()`; tasks
  without an early end condition return `{}`.
- Clamp actions before they reach each task.
- Make evaluation spacing match training, and check that every sample keeps it matched.
- The structure check requires `task.py`, `preview.py`, `train.py`, `eval.py`, and `play.py` in
  every sample and exactly one job in each catalog row, and discovery compares that job with
  the one the client packages from `train.py`.
- Removed the declared Simulo client compatibility range. The samples are written for the
  latest client, and a scheduled weekly check packages every declared job with the latest
  published client.
- Bring back Humanoid, with normalised observations. With the default training, the robot
  stays upright and ends at least 1 m forward in 99 of 100 evaluation episodes, and seeds 1, 2,
  and 3 also pass that rule. The rule does not check for stepping, so the motion can look like a
  shuffle or a drag rather than a walk.
- Bring back Franka reach. It trains longer by default so it reaches the goal reliably, and
  the arm now observes its own target.
- Bring back Bring your own URDF. Each action now names the angle a joint should hold, and the
  default training reaches every joint target in 98 of 100 evaluation episodes.
- Bring back Install a PyPI dependency. Its reward is now computed by Shapely for all robots
  at once, and a stronger reward for reaching the target zone lets the default training
  learn to arrive.
- Update the Cartpole, JetBot, and F1TENTH results from a staging run on 2026-10-06 with
  simulo 0.32.0.

### Removed

- Hello. Its job printed greetings, trained nothing, and saved no checkpoint.
- Cartpole Eval. `eval.py` and `play.py` beside every sample now evaluate and play a policy.
- The `rollout` job of Bring your own F1TENTH-compatible car. Its `play.py` now plays and
  records a trained policy.

### Fixed

- Every job requests its GPU with a `system=simulo.SystemType` tier instead of the removed
  `@app.job(gpu=...)` parameter, which the current client no longer accepts.

## [2026.09.1] - 2026-09-06

### Added

- Repository scaffold and structural checks for the sample catalog.
- Hello: submit a job, follow its log, and read its result. Requests no GPU.
- Cartpole: train a balancing policy and read the reward trend and saved checkpoint.
- Cartpole Eval: train, score over several rounds, then play the policy back and record it.
- Humanoid: train a 21-joint biped to stay upright and read the reward climbing toward zero.
- JetBot: train a two-wheeled robot to drive in a commanded direction.
- Franka reach: train an arm onto a moving goal with a differential IK controller.
- Install a PyPI dependency: compute a reward with a third-party library installed into
  the job's runtime.
- Bring your own URDF: publish a robot you wrote to your organization's catalog, then
  train a reaching policy on it.
- A repository-level `assets/` tree holding the files a sample tells the reader to
  publish to their own catalog. A directory's path is the reference it publishes as:
  `assets/robot/byo-urdf-arm/` becomes `robot/byo-urdf-arm:v1`.
- Bring your own F1TENTH-compatible car: publish a USD race car to your organization's
  catalog, then train a drifting policy around a stadium-shaped track and record a lap.
- Samples declare compatibility with Simulo client versions `>=0.23.1,<0.25`.
