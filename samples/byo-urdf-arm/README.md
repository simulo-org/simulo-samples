# Bring your own URDF

## What this shows

The other training samples here use robots from the Simulo catalog. This one uses a robot
you bring. [`assets/robot/byo-urdf-arm/`](../../assets/robot/byo-urdf-arm/) holds a
three-joint desktop arm written as an ordinary URDF: four meshes, real link masses and
inertia tensors, and revolute joints with limits and damping. You send that directory to
your organization's catalog with one command, and from then on the arm behaves like any
other catalog robot: version-pinned, resolved when you submit, and mounted read-only
while the job runs.

You will learn how `simulo asset publish` turns a directory of robot files into a
version-pinned catalog reference, why a reference with no publisher segment resolves
against your own organization rather than the Simulo catalog, and how to write a
joint-space reaching task against a robot whose joint names you chose yourself.

## Prerequisites

- Python 3.11 or newer and the Simulo client: `python -m pip install --upgrade simulo`.
- A Simulo account, signed in once with `simulo login`.
- A clone of this repository. The commands below run from its root.
- The arm published to your organization's catalog. [Run it](#run-it) does that first;
  the training job cannot start before it.
- No GPU on your machine. The job asks for a Tier 1 GPU (T4) in the Simulo cloud and is
  billed to your account. Hardware describes the job's own request, not queue priority:
  it waits on the same shared GPU fleet as every other job.

## Assets

- `assets/robot/byo-urdf-arm/robot.urdf` and `assets/robot/byo-urdf-arm/meshes/*.stl`:
  the arm itself, in this repository. These files are the input to
  `simulo asset publish`. They sit outside this sample directory, so `simulo run` never
  uploads them; the arm reaches the job from the catalog instead. The
  [assets tree](../../assets/) explains how its directories are named.
- `robot/byo-urdf-arm:v1`: the arm once you have published it, pinned to version 1. The
  reference carries no publisher segment, so it resolves against whichever organization
  you are signed in as. A reference like `simulo/robot/cartpole:v1` names the Simulo
  catalog instead.

The arm has three revolute joints, `shoulder_pan`, `shoulder_lift`, and `elbow`, with
limits of ±2.6, ±1.9, and ±2.2 radians and a 30 N·m effort limit each. `train.py` looks
those names up, so a robot of your own with different joint names needs the names in
`on_start` changed to match.

## Files and APIs

- `task.py`: the application, reward kernel, and three-joint arm task.
- `train.py`: the `train` job, declared with `@app.job(type="train", ...)`, which saves the
  policy's `best` and `latest` checkpoints automatically.
- `eval.py`: the evaluation job and its joint-target success rule.
- `.simuloignore`: files `simulo run` leaves out of the uploaded package.
- `../../assets/robot/byo-urdf-arm/robot.urdf`: the robot description. Its
  `<mesh filename="meshes/...">` references resolve relative to the URDF.
- `../../assets/robot/byo-urdf-arm/meshes/`: the four meshes the URDF names, one per
  link.

Simulo names it uses, beyond those in the [Cartpole](../cartpole/) sample:

- `simulo.Asset.from_registry("robot/byo-urdf-arm:v1")`: an organization-scoped
  reference, written without the leading publisher segment a Simulo catalog reference
  carries.
- `robot.find_joints("shoulder_pan")` and its two siblings, resolving the joint names the
  URDF declares. The three results are concatenated into one index list, so observations,
  actions, and the reward all use the same joint order.
- `robot.set_joint_effort_target(..., joint_ids=...)` on exactly those three joints.

## Run it

Publish the arm first, from the repository root:

```bash
simulo login
simulo asset publish assets/robot/byo-urdf-arm \
  --kind robot --name byo-urdf-arm --entry robot.urdf
```

`--kind robot` says what you are publishing, and the command never guesses it.
`--name byo-urdf-arm` fixes the catalog name `train.py` expects. `--entry robot.urdf` says which file to start from,
which matters because the directory holds meshes as well. The command uploads the
directory, fills in the physics a URDF does not carry, such as drive gains, friction, and
collision settings, and reports every value it chose on your behalf rather than applying
it silently. It then publishes the arm as version 1. Add `--report report.json` to keep
the full report, or `--json` for a machine-readable summary.

A robot published from a URDF gets a fixed base unless you say otherwise, which is what
this arm wants: bolted at the origin, like a manipulator on a bench. A mobile or legged
robot needs `--base floating`.

Check what the catalog recorded:

```bash
simulo asset inspect robot/byo-urdf-arm:v1
```

Then train:

```bash
simulo run samples/byo-urdf-arm/train.py --num-envs 256 --max-iterations 150
```

`--num-envs` and `--max-iterations` are `train`'s own parameters. For a quick check that
the job launches, use `--num-envs 16 --max-iterations 2`. Add `--detach` to submit
without waiting for the log.

The job stops itself after eight hours of execution and retries up to twice if it fails.
Run `simulo cancel <job-id>` to stop a queued or running job.

### Evaluate

From the repository root, use `simulo policy list` to find the policy id, then run:

```bash
simulo run samples/byo-urdf-arm/eval.py --policy <policy-id>:best
```

The rule checks that every joint is within 0.1 rad of its target and that the arm settles below
0.1 rad/s.

The default run is short, so expect its policy to miss the 0.1 rad bar on most or all episodes;
train for more iterations to see the success rate rise. The start of a real report, from
evaluating a policy trained with this sample's default settings (256 environments, 150 iterations):

```text
RESULT
  0 of 100 episodes succeeded. Likely range: 0% to 4%.
  Each check:  every joint within 0.1 rad of its target 0 of 100 · arm settled below 0.1 rad/s 3 of 100
  Among 100 failures: 97 missed both checks, 3 missed only "every joint within 0.1 rad of its target".

WHAT WAS TESTED
  Policy     policy_flat-inlet-fwawnt:best, saved at iteration 150 (training reward -284.3)
  Task       ByoArmTask in task.py, unchanged since the policy was trained
  Episodes   0 to 99, the same starting positions every time
  Actions    the policy's best-guess action, without training's random variation
  Success    completed_stably() in eval.py, with checks: every joint within 0.1 rad of its target, arm settled below 0.1 rad/s
```

## What to expect

Publishing is a one-time step. `simulo asset publish` reports what it did as it goes, and
from then on every run resolves `robot/byo-urdf-arm:v1` from the catalog and uploads
nothing.

The training job then behaves like the other training samples. Once the simulation
starts, the log shows an iteration counter and, every 50 iterations, a "Checkpoint written" line,
plus a "Best checkpoint saved" line whenever the mean episode reward improved. The reward
here is negative and climbs toward zero: it is minus the squared joint-space distance to
the target, minus a small velocity penalty. A policy that parks the arm on the target and
holds it there scores near 0, and a random policy scores far below. `best_reward` is a
mean completed-episode return, not a per-step reward.

The result names the catalog reference the job trained against:

```json
{"num_envs": 256, "robot_asset": "robot/byo-urdf-arm:v1"}
```

Alongside those, the result carries the training statistics `RLTrainer.train` returns,
such as `iterations` and `best_reward`. A run takes about two minutes at the defaults
once capacity is free, including the one-time publish; a first run can take longer
while the cloud prepares the runtime.

## Inspecting results

```bash
simulo asset inspect robot/byo-urdf-arm:v1  # what the catalog recorded for the arm
simulo asset list                           # every asset in your organization's catalog
simulo jobs                                 # status and job IDs
simulo logs <job-id> --follow               # iteration and checkpoint lines
simulo result <job-id>                      # the returned dictionary
simulo policy list                          # your policies and their checkpoints
simulo policy get <policy-id>:best          # download the best checkpoint, digest-verified
simulo export <policy-id>:best              # the best checkpoint as a portable ONNX bundle
simulo cancel <job-id>                      # stop a queued or running job
```

To watch the arm move, submit with `--viewstream` and open `simulo view` while it runs;
streaming slows training, so use it to look.

`<job-id>` and `<policy-id>` are printed by `simulo run` and listed by `simulo jobs`. Every
training job makes one policy, whose ID is the job's ID with a `policy_` prefix in place of
`job_`. Stopping a log-follow session only detaches from the stream; it does not cancel the
job.

## Troubleshooting

- The job fails while building the scene, naming an asset it cannot resolve: the arm is
  not in the catalog of the organization you are signed in as. Publish it, then submit
  again.
- `simulo asset publish` cannot tell which file to start from: pass `--entry robot.urdf`.
  The directory holds meshes as well as the URDF.
- `simulo asset publish` cannot find a mesh: `<mesh filename="...">` resolves relative to
  the URDF, and every file it names must sit inside the directory you publish. Nothing is
  collected from elsewhere on your machine.
- The arm drifts or falls instead of staying bolted down: it was published with
  `--base floating`. Publish again without it; that creates a new version.
- The arm comes out a thousand times too large or too small: the URDF was authored in
  millimetres. Publish with `--scale 0.001`.
- `simulo run` prints "This wrote a local package only": you are not signed in.
- The job stays `queued`: the cloud is waiting for GPU capacity.
- The task cannot find its joints after you swap in a robot of your own: the three names
  `on_start` looks up are the ones this URDF declares. Change them to your robot's names,
  and move `observation_dim` and `action_dim` with them if the joint count changes.

## Extending it

- Publish a robot of your own: point `simulo asset publish` at your own directory, choose
  the catalog name with `--name`, and change the reference at the top of `train.py` to
  match.
- Change this arm: edit `assets/robot/byo-urdf-arm/robot.urdf` and publish again. Each
  publish creates a new immutable version, so `:v1` keeps resolving to the arm you
  already trained against until you pin the new one.
- Widen the reach: `target_amplitude` sets how far sampled targets stray from zero. Keep
  it inside the joint limits the URDF declares.
- Reshape the reward: `rew_scale_distance` and `rew_scale_velocity` weight getting to the
  target against settling once you arrive.
- Push the arm harder: `action_scale` is the effort one full-scale action commands, and
  the URDF's 30 N·m limit is the ceiling.

## Assets, licensing, attribution

This sample, including `assets/robot/byo-urdf-arm/robot.urdf` and the meshes it names,
is available under the [MIT License](../../LICENSE).

This sample installs no third-party packages.

### Attribution

Copyright (c) 2026 Simulo LLC.
