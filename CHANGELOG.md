# Changelog

All notable changes to this repository are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Releases are
tagged by calendar version, `vYYYY.MM.N`, because this repository publishes samples rather
than an interface: there is no API here whose compatibility a semantic version could describe.

## [Unreleased]

### Added

- `simulo install samples` as a documented way to get this repository, alongside
  `git clone`, with options to choose a destination and a Git branch, tag, or commit.

### Changed

- Removed the declared Simulo client compatibility range. The samples are written for the
  latest client, installation upgrades to it, and a scheduled weekly check packages every
  declared job with the latest published client. The runtimes and results each sample
  describes were recorded with an earlier client.

### Fixed

- Migrated every GPU-requesting sample off the removed `@app.job(gpu=...)` parameter to
  `system=simulo.SystemType.TIER_1`. The current client no longer accepts `gpu=`, so the
  old form would raise `TypeError` at import time.
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
- Samples declare compatibility with Simulo client versions `>=0.23.1,<0.25`.
