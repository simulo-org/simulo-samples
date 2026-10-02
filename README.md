# Simulo samples

This repository collects runnable Simulo robotics projects for engineers learning how to
define cloud simulation and reinforcement learning jobs. Each sample is a small application
that you submit with the Simulo client. A sample's `train.py` declares the one training job
you submit, and its task lives in `task.py` next to it. Every
training sample also has an `eval.py` that measures a saved policy against that sample's rule.

## Prerequisites and compatibility

Install Python 3.11 or newer and the Simulo client. These samples are written for the latest
Simulo client. If a sample does not work with the latest client, open an issue and include
the output of `simulo --version`.

```bash
python -m pip install --upgrade simulo
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

`simulo install samples` does the same clone for you:

```bash
simulo install samples
cd simulo-samples
```

Choose a different destination:

```bash
simulo install samples --directory robotics-examples
```

The destination must not exist, and its parent directory must already exist. The destination
appears only after the clone and checkout succeed. Add `--ref` with a branch, tag, or full
commit ID to check out that revision instead of the default branch. A checkout made with
`--ref` is detached at the selected commit.

## Quick start

Start with [Cartpole](samples/cartpole/): it trains a pole-balancing policy on a catalog
robot, and every other training sample here has the same shape. A short run checks that
the job launches:

```bash
simulo run samples/cartpole/train.py --num-envs 64 --max-iterations 2
```

Run it without the flags for the full training run. Every job waits on the same shared GPU
fleet, so a first run can sit `queued` for a while. A training job saves its policy's
`best` and `latest` checkpoints on its own. `simulo policy list` shows them, and
`simulo run samples/cartpole/train.py --max-iterations 400 --from <policy-id>:best` continues
training from one, with `--max-iterations` read as the new total.

After training, run `simulo policy list` from the repository root to find the policy id, then
evaluate its saved checkpoint with `simulo run samples/cartpole/eval.py --policy <policy-id>:best`.

Each sample README explains its prerequisites, assets, expected results, and runtime.

## Sample index

Hardware below describes only what a sample's own job code requests, not how quickly it
starts. There is one shared GPU fleet: a job that requests no GPU still queues for that
same fleet, on the same basis as a job that does.

<!-- BEGIN INDEX -->
### By learning goal

#### Train a policy

- [Cartpole](samples/cartpole/): Cartpole. Train a PPO policy that balances a pole on a sliding cart. Concepts: Task lifecycle, LearningEnv and RLTrainer, automatic checkpoints, continuing with --from. Assets: `simulo/robot/cartpole:v1`. Hardware: Tier 1 GPU job (T4, 16 GB VRAM); no local GPU required. Runtime: about 2 minutes.
- [Humanoid](samples/humanoid/): Bipedal humanoid. Train a 21-joint biped to stay upright and move forward. Concepts: Locomotion, body-frame observations, termination versus truncation. Assets: `simulo/robot/humanoid:v1`. Hardware: Tier 1 GPU job (T4, 16 GB VRAM); no local GPU required. Runtime: about 5 minutes.
- [JetBot](samples/jetbot/): JetBot. Train a two-wheeled robot to drive in a commanded direction. Concepts: Differential drive, randomised commands, network-fetched asset. Assets: `simulo/robot/jetbot:v2`. Hardware: Tier 1 GPU job (T4, 16 GB VRAM); no local GPU required. Runtime: about 5 minutes.
- [Franka reach](samples/franka-reach/): Franka Panda arm. Train an arm to put its hand on a moving goal, with a differential IK controller. Concepts: Task-space control, DifferentialIKController, body pose readback. Assets: `simulo/robot/franka-panda:v1`. Hardware: Tier 1 GPU job (T4, 16 GB VRAM); no local GPU required. Runtime: about 3 minutes.

#### Bring your own dependency

- [Install a PyPI dependency](samples/pip-install-shapely/): JetBot. Compute a training reward with a third-party PyPI library installed into the job's runtime. Concepts: Runtime.pip_install, Runtime.env, third-party rewards. Assets: `simulo/robot/jetbot:v2`, `simulo/gpu-rl:2026.06`. Hardware: Tier 1 GPU job (T4, 16 GB VRAM); no local GPU required. Runtime: about 3 minutes.

#### Bring your own robot

- [Bring your own URDF](samples/byo-urdf-arm/): Your own three-joint arm. Publish a robot you wrote as a URDF to your organization's catalog, then train a reaching policy on it. Concepts: simulo asset publish, organization catalog, URDF robots, joint-space reaching. Assets: `robot/byo-urdf-arm:v1`. Hardware: Tier 1 GPU job (T4, 16 GB VRAM); no local GPU required. Runtime: about 2 minutes.
- [Bring your own F1TENTH-compatible car](samples/byo-f1tenth-drift/): Your own F1TENTH-compatible race car. Publish an F1TENTH-compatible USD race car to your organization's catalog, then train a drifting policy around a stadium-shaped track. Concepts: simulo asset publish, organization catalog, USD robots, 4WD steering, multi-term rewards, best checkpoints. Assets: `robot/f1tenth:v1`. Hardware: Tier 1 GPU job (T4, 16 GB VRAM); no local GPU required. Runtime: about 5 minutes.

### By difficulty

#### Introductory

- [Cartpole](samples/cartpole/): Cartpole. Train a PPO policy that balances a pole on a sliding cart. Concepts: Task lifecycle, LearningEnv and RLTrainer, automatic checkpoints, continuing with --from. Assets: `simulo/robot/cartpole:v1`. Hardware: Tier 1 GPU job (T4, 16 GB VRAM); no local GPU required. Runtime: about 2 minutes.
- [JetBot](samples/jetbot/): JetBot. Train a two-wheeled robot to drive in a commanded direction. Concepts: Differential drive, randomised commands, network-fetched asset. Assets: `simulo/robot/jetbot:v2`. Hardware: Tier 1 GPU job (T4, 16 GB VRAM); no local GPU required. Runtime: about 5 minutes.

#### Intermediate

- [Humanoid](samples/humanoid/): Bipedal humanoid. Train a 21-joint biped to stay upright and move forward. Concepts: Locomotion, body-frame observations, termination versus truncation. Assets: `simulo/robot/humanoid:v1`. Hardware: Tier 1 GPU job (T4, 16 GB VRAM); no local GPU required. Runtime: about 5 minutes.
- [Franka reach](samples/franka-reach/): Franka Panda arm. Train an arm to put its hand on a moving goal, with a differential IK controller. Concepts: Task-space control, DifferentialIKController, body pose readback. Assets: `simulo/robot/franka-panda:v1`. Hardware: Tier 1 GPU job (T4, 16 GB VRAM); no local GPU required. Runtime: about 3 minutes.
- [Install a PyPI dependency](samples/pip-install-shapely/): JetBot. Compute a training reward with a third-party PyPI library installed into the job's runtime. Concepts: Runtime.pip_install, Runtime.env, third-party rewards. Assets: `simulo/robot/jetbot:v2`, `simulo/gpu-rl:2026.06`. Hardware: Tier 1 GPU job (T4, 16 GB VRAM); no local GPU required. Runtime: about 3 minutes.
- [Bring your own URDF](samples/byo-urdf-arm/): Your own three-joint arm. Publish a robot you wrote as a URDF to your organization's catalog, then train a reaching policy on it. Concepts: simulo asset publish, organization catalog, URDF robots, joint-space reaching. Assets: `robot/byo-urdf-arm:v1`. Hardware: Tier 1 GPU job (T4, 16 GB VRAM); no local GPU required. Runtime: about 2 minutes.
- [Bring your own F1TENTH-compatible car](samples/byo-f1tenth-drift/): Your own F1TENTH-compatible race car. Publish an F1TENTH-compatible USD race car to your organization's catalog, then train a drifting policy around a stadium-shaped track. Concepts: simulo asset publish, organization catalog, USD robots, 4WD steering, multi-term rewards, best checkpoints. Assets: `robot/f1tenth:v1`. Hardware: Tier 1 GPU job (T4, 16 GB VRAM); no local GPU required. Runtime: about 5 minutes.
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

If you installed with `--ref`, the checkout is detached; run `git switch main` first.

## Support

Open an issue in this repository for bugs and sample requests. The documentation site at
[docs.simulo.ai](https://docs.simulo.ai) covers the client and cloud workflow. Pull requests
are not accepted yet.

## Licensing and attribution

Sample code is available under the [MIT License](LICENSE). Copyright (c) 2026 Simulo LLC.
