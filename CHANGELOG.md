# Changelog

All notable changes to this repository are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Releases are
tagged by calendar version, `vYYYY.MM.N`, because this repository publishes samples rather
than an interface: there is no API here whose compatibility a semantic version could describe.

## [Unreleased]

- Add preview and play files to every sample.
- Make evaluation spacing match training.
- Update Humanoid so it walks.
- Clamp actions before they reach each task.
- Document the preview, train, evaluate, and play lifecycle.
- Bring back Humanoid. It now walks reliably, using normalised observations.
- Bring back Franka reach. It trains longer by default so it reaches the goal reliably, and
  the arm now observes its own target.
- Bring back Bring your own URDF. Each action now names the angle a joint should hold, and the
  default training reaches every joint target in 98 of 100 evaluation episodes.
- Update the Cartpole, JetBot, and F1TENTH results from a staging run on 2026-10-06 with
  simulo 0.32.0.
- Bring back Install a PyPI dependency. Its reward is now computed by Shapely for all robots
  at once, and a stronger reward for reaching the target zone lets the default training
  learn to arrive.

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
