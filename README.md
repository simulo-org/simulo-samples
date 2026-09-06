# Simulo samples

This repository collects runnable Simulo robotics projects for engineers learning how to
define cloud simulation and reinforcement learning jobs. Each sample is a small application
that you submit with the Simulo client.

## Prerequisites and compatibility

Install Python 3.11 or newer and the Simulo client. Every sample works with Simulo client
versions `>=0.23.1,<0.25`.

```bash
python -m pip install "simulo>=0.23.1,<0.25"
```

Samples run in the Simulo cloud. Sign in before submitting a job:

```bash
simulo login
```

## Get the samples

Clone this repository:

```bash
git clone https://github.com/simulo-org/simulo-samples.git
cd simulo-samples
```

## Quick start

Start with [Hello](samples/hello/): it is deterministic, requests no GPU, uses no catalog
assets, and has an exact expected result.

```bash
simulo run samples/hello/app.py --name robot --repeat 5
```

Each sample README explains its prerequisites, assets, expected results, and runtime.

## Sample index

<!-- BEGIN INDEX -->
### By learning goal

#### Submit a job and read its output

- [Hello](samples/hello/): No robot. Print a few greeting lines from a job and return a small JSON result. Concepts: App and jobs, job flags, logs and results. Assets: none. Hardware: cloud CPU; no GPU requested. Runtime: about 1 minute.

#### Train a policy

- [Cartpole](samples/cartpole/): Cartpole. Train a PPO policy that balances a pole on a sliding cart. Concepts: Task lifecycle, LearningEnv and RLTrainer, Volumes, ResumableCheckpoint. Assets: `simulo/robot/cartpole:v1`. Hardware: L4-class cloud GPU; no local GPU required. Runtime: about 2 minutes.
- [Humanoid](samples/humanoid/): Bipedal humanoid. Train a 21-joint biped to stay upright and move forward. Concepts: Locomotion, body-frame observations, termination versus truncation. Assets: `simulo/robot/humanoid:v1`. Hardware: L4-class cloud GPU; no local GPU required. Runtime: about 5 minutes.
- [JetBot](samples/jetbot/): JetBot. Train a two-wheeled robot to drive in a commanded direction. Concepts: Differential drive, randomised commands, network-fetched asset. Assets: `simulo/robot/jetbot:v2`. Hardware: L4-class cloud GPU; no local GPU required. Runtime: about 5 minutes.
- [Franka reach](samples/franka-reach/): Franka Panda arm. Train an arm to put its hand on a moving goal, with a differential IK controller. Concepts: Task-space control, DifferentialIKController, body pose readback. Assets: `simulo/robot/franka-panda:v1`. Hardware: L4-class cloud GPU; no local GPU required. Runtime: about 3 minutes.

#### Evaluate and record a policy

- [Cartpole Eval](samples/cartpole-eval/): Cartpole. Train a policy, score it over several rounds, then play it back and record the rollout. Concepts: Multiple jobs, shared volumes, policy export, RLPlayer, MCAP recording, camera video. Assets: `simulo/robot/cartpole:v1`. Hardware: L4-class cloud GPU; no local GPU required. Runtime: about 2 minutes.

#### Bring your own dependency

- [Install a PyPI dependency](samples/pip-install-shapely/): JetBot. Compute a training reward with a third-party PyPI library installed into the job's runtime. Concepts: Runtime.pip_install, Runtime.env, third-party rewards. Assets: `simulo/robot/jetbot:v2`, `simulo/gpu-rl:2026.06`. Hardware: L4-class cloud GPU; no local GPU required. Runtime: about 3 minutes.

### By difficulty

#### Introductory

- [Hello](samples/hello/): No robot. Print a few greeting lines from a job and return a small JSON result. Concepts: App and jobs, job flags, logs and results. Assets: none. Hardware: cloud CPU; no GPU requested. Runtime: about 1 minute.
- [Cartpole](samples/cartpole/): Cartpole. Train a PPO policy that balances a pole on a sliding cart. Concepts: Task lifecycle, LearningEnv and RLTrainer, Volumes, ResumableCheckpoint. Assets: `simulo/robot/cartpole:v1`. Hardware: L4-class cloud GPU; no local GPU required. Runtime: about 2 minutes.
- [JetBot](samples/jetbot/): JetBot. Train a two-wheeled robot to drive in a commanded direction. Concepts: Differential drive, randomised commands, network-fetched asset. Assets: `simulo/robot/jetbot:v2`. Hardware: L4-class cloud GPU; no local GPU required. Runtime: about 5 minutes.

#### Intermediate

- [Cartpole Eval](samples/cartpole-eval/): Cartpole. Train a policy, score it over several rounds, then play it back and record the rollout. Concepts: Multiple jobs, shared volumes, policy export, RLPlayer, MCAP recording, camera video. Assets: `simulo/robot/cartpole:v1`. Hardware: L4-class cloud GPU; no local GPU required. Runtime: about 2 minutes.
- [Humanoid](samples/humanoid/): Bipedal humanoid. Train a 21-joint biped to stay upright and move forward. Concepts: Locomotion, body-frame observations, termination versus truncation. Assets: `simulo/robot/humanoid:v1`. Hardware: L4-class cloud GPU; no local GPU required. Runtime: about 5 minutes.
- [Franka reach](samples/franka-reach/): Franka Panda arm. Train an arm to put its hand on a moving goal, with a differential IK controller. Concepts: Task-space control, DifferentialIKController, body pose readback. Assets: `simulo/robot/franka-panda:v1`. Hardware: L4-class cloud GPU; no local GPU required. Runtime: about 3 minutes.
- [Install a PyPI dependency](samples/pip-install-shapely/): JetBot. Compute a training reward with a third-party PyPI library installed into the job's runtime. Concepts: Runtime.pip_install, Runtime.env, third-party rewards. Assets: `simulo/robot/jetbot:v2`, `simulo/gpu-rl:2026.06`. Hardware: L4-class cloud GPU; no local GPU required. Runtime: about 3 minutes.
<!-- END INDEX -->

## Assets

Most samples train against robots the Simulo catalog already carries, named with a
publisher: `simulo/robot/cartpole:v1`. A sample can also use a robot, world, or prop that
this repository ships and you publish to your own organization's catalog, named without
one: `robot/byo-urdf-arm:v1`. Those files live under [`assets/`](assets/), one directory
per asset, and that directory's path is the reference it publishes as.

## Update a clone

Pull the latest sample catalog and files from your existing clone:

```bash
git pull
```

## Support

Open an issue in this repository for bugs and sample requests. The documentation site at
[docs.simulo.ai](https://docs.simulo.ai) covers the client and cloud workflow. Pull requests
are not accepted yet.

## Licensing and attribution

Sample code is available under the [MIT License](LICENSE). Copyright (c) 2026 Simulo LLC.
