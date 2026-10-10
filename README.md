# Simulo samples

Runnable Simulo robotics projects for learning how to define and submit cloud simulation
and reinforcement learning jobs.

Last tested end to end with simulo `0.32.0` on 2026-10-06.

## Install

Use the Simulo client to install the samples:

```bash
simulo install samples
cd simulo-samples
```

To choose a destination, use `simulo install samples --directory robotics-examples`.
The destination must not already exist. Add `--ref` with a branch, tag, or full commit ID to
check out a specific revision.

You can also clone the repository directly:

```bash
git clone https://github.com/simulo-org/simulo-samples.git
cd simulo-samples
```

## Sign in

Install Python 3.11 or newer and the latest Simulo client:

```bash
python -m pip install --upgrade simulo
```

Sign in before submitting a job:

```bash
simulo login
```

## Common lifecycle

Each sample follows the same path. Replace `<sample>` with a folder from the index and use the
policy ID printed by training.

```bash
simulo run samples/<sample>/preview.py
simulo recordings <preview-job-id>

simulo run samples/<sample>/train.py
simulo policy list

simulo run samples/<sample>/eval.py --policy policy_<stem>:best
simulo run samples/<sample>/play.py --policy policy_<stem>:best
simulo recordings <play-job-id>
```

Preview checks the task before training. Evaluate checks the saved policy against the sample's
rule. Play makes the policy easy to inspect and records the run.

## Choose a sample

The index is generated from `samples.toml`. Each row lists its one training job's lesson and
the assets it needs.

<!-- BEGIN INDEX -->
| Sample | Difficulty | Main lesson | Assets |
| --- | --- | --- | --- |
| [Cartpole](samples/cartpole/) | Introductory | Train a policy | `simulo/robot/cartpole:v1` |
| [JetBot](samples/jetbot/) | Introductory | Train a policy | `simulo/robot/jetbot:v1` |
| [Franka reach](samples/franka-reach/) | Intermediate | Train a policy | `simulo/robot/franka-panda:v1` |
| [Humanoid](samples/humanoid/) | Intermediate | Train a policy | `simulo/robot/humanoid:v1` |
| [Install a PyPI dependency](samples/pip-install-shapely/) | Intermediate | Bring your own dependency | `simulo/robot/jetbot:v1`, `simulo/gpu-rl:2026.06` |
| [Bring your own F1TENTH-compatible car](samples/byo-f1tenth-drift/) | Intermediate | Bring your own robot | `robot/f1tenth:v1` |
| [Bring your own URDF](samples/byo-urdf-arm/) | Intermediate | Bring your own robot | `robot/byo-urdf-arm:v1` |
| [Franka reach from camera](samples/franka-reach-from-camera/) | Advanced | Find a reaching target through a camera | `simulo/robot/franka-panda:v2` |
<!-- END INDEX -->

## Bring your own robot

Some samples include robot files under [`assets/`](assets/). Publish those files to your own
organization before running the sample. A directory's path is the catalog reference.

## Update a clone

Pull the latest sample catalog and files:

```bash
git pull
```

If you installed with `--ref`, the checkout is detached. Run `git switch main` before pulling.

## Support

Open an issue in this repository for bugs and sample requests. The documentation site at
[docs.simulo.ai](https://docs.simulo.ai) covers the client and cloud workflow. Pull requests
are not accepted yet.

## License

Sample code is available under the [MIT License](LICENSE). Copyright (c) 2026 Simulo LLC.
