# Changelog

All notable changes to this repository are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Releases are
tagged by calendar version, `vYYYY.MM.N`, because this repository publishes samples rather
than an interface: there is no API here whose compatibility a semantic version could describe.

## [Unreleased]

### Added

- `simulo install samples` as a documented way to get this repository, alongside
  `git clone`, on Simulo client `0.25.0` or newer.

### Changed

- `tools/validate_samples.py --discover` now reads the installed client's own reported
  version (`simulo --version`) and fails if it falls outside the declared compatibility
  range, instead of only proving that some unspecified client packages every job. A
  green `discovery` run is now evidence about which release it actually exercised.
- Widened the declared Simulo client compatibility range to `>=0.23.1,<0.26` to admit
  `simulo` 0.25.0, published after this repository's samples were validated. No sample was
  re-run against the new release: `runtime_minutes`, `published`, and each README's "What
  to expect" prose still describe the run recorded during the original validation window,
  and this widening is evidence only that every sample still packages and still declares
  the jobs its row claims against the new release.
- Raised the declared Simulo client compatibility range's floor, to `>=0.26.0,<0.27`, to
  require `simulo` 0.26.0. This is a floor raise, not a widening: `simulo.SystemType`,
  which every sample now imports (see Fixed, below), does not exist before `simulo`
  0.26.0, so any version the prior range admitted below it would fail with
  `AttributeError` at import time. Same evidence boundary as the prior change:
  `runtime_minutes`, `published`, and each README's "What to expect" prose are unchanged,
  and this is evidence only that every sample still packages and still declares the jobs
  its row claims against the new release.

### Fixed

- Migrated every GPU-requesting sample off the removed `@app.job(gpu=...)` parameter to
  `system=simulo.SystemType.TIER_1`. `simulo` 0.26.0 removed `gpu=` outright, so every
  sample would have raised `TypeError` at import time under the compatibility range
  raise above without this change.
- Corrected the generated "Hardware" claim from an L4-class GPU to a Tier 1 GPU (T4, 16 GB
  VRAM): the tier every sample's job actually requests, and the tier this repository's
  samples were actually validated against. Nothing in this repository requests or
  provisions an L4.
- Reworded the "Hardware" field and nearby Quick start / Prerequisites prose so "no GPU
  requested" no longer reads as "starts sooner": every sample, GPU-requesting or not,
  queues on the same single, shared GPU fleet, and the Hardware field
  describes only a job's own resource request, not queue priority.

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
- Samples declare compatibility with Simulo client versions `>=0.23.1,<0.26`.
