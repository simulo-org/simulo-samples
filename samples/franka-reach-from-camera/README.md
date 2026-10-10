# Franka reach from camera

## What this shows

Train a Franka Panda arm to put its fingertips over a cube it can only see. Each episode a
red cube appears at a random spot on a table. The policy gets one fixed camera image of the
scene plus the arm's own joint positions and velocities. It never receives the cube's
coordinates.

The focused lessons are:

- declaring named policy inputs with `simulo.PolicyInputSchema`: an image from a
  `simulo.Camera(purpose="policy")` and a numeric joint vector;
- giving the critic, which only exists during training, extra inputs the policy never
  sees. Here the critic also gets the cube's true position, which makes learning faster,
  and the trained policy still runs from the camera and joints alone.

## Prerequisites

- Python 3.11 or newer with Simulo 0.33.0 or newer installed from the public package index.
- A Simulo account and `simulo login`.
- Run commands from the repository root.

## Assets

The task uses the version-pinned catalog robot `simulo/robot/franka-panda:v2`,
the Franka Panda model converted to USD. Submission with
`--frozen` requires an explicit version and records the resolved content digest
on the job. Check that version 2 is available to your account before training.

## Files and APIs

- `task.py` defines `FrankaReachFromCameraTask`: the table, the cube, the policy camera,
  the actor and critic inputs, and the fingertip motion through
  `simulo.DifferentialIKController`.
- `preview.py` checks the scene, the camera input, and the action mapping before training.
- `train.py` declares the `train_franka_reach_from_camera` PPO job with a typed `seed`.
- `eval.py` evaluates deterministic actions on numbered episode starts and defines the success rule.
- `play.py` plays a saved policy and records each episode in a separate MCAP file.
- `.simuloignore` excludes local files from submitted packages.

The policy inputs are:

| Input | Shape | Contents |
| --- | --- | --- |
| `camera` | 100 x 100 x 3, uint8 | RGB image from the fixed camera in front of the table |
| `joints` | 14, float | 7 arm joint positions relative to the home pose, then 7 joint velocities |

The critic gets `joints` and `privileged`: the fingertip point, the target point, and the
vector between them.

The arm starts angled across the table, with the hand pointing down. Its seven
home joint angles are `(-1.9670, -1.0720, 0.9882, -1.0661, 1.3236, 0.8577, -0.9218)`
radians. This scene-specific pose keeps the hand and target in the policy camera,
puts the fingertips inside the command workspace, and clears the table and floor
during preview's individual joint sweeps. Reset adds up to 0.05 radians of noise
per arm joint. Each action requests at most 0.03 metres along each fingertip axis;
a zero action holds the current fingertip point while correcting the small
orientation difference from reset noise.

The table, camera, workspace, robot base and shared drive settings are unchanged.
Table, floor and self collisions remain enabled. The robot file's original home
pose drives the hand into the table during individual joint sweeps.

## Run it

### Preview

```bash
simulo run samples/franka-reach-from-camera/preview.py --frozen
simulo recordings <job-id> --all
```

### Train

```bash
simulo run samples/franka-reach-from-camera/train.py --frozen --num-envs 128 --max-iterations 600 --seed 42
simulo policy list
```

Use `--seed 43` for an independent second run.

The default configuration uses 128 environments, 600 PPO updates, 24 rollout
steps per environment, five learning epochs and eight minibatches. The job has
a 45-minute timeout. That timeout is a limit, not a measured training duration.

### Resume

```bash
simulo run samples/franka-reach-from-camera/train.py --frozen --from <policy-id>:latest --num-envs 128 --max-iterations 650 --seed 42
```

`--max-iterations` is the total update count, including the checkpoint's completed
updates. For a checkpoint at update 600, this command adds 50 updates. Use the
same environment count, task, camera and PPO settings as the original run.
The continued run creates a new policy; keep the original policy ID for the
baseline evaluation below.

### Evaluate

```bash
simulo run samples/franka-reach-from-camera/eval.py --frozen --policy <policy-id>:latest --episodes 100 --num-envs 100
simulo result <eval-job-id>
```

This evaluates numbered starts 0 through 99 with deterministic actions. Episode
numbers determine starts independently of training's seed. `:latest` selects the
final saved checkpoint; `:best` selects the checkpoint with the highest recorded
training reward. Use `:latest` when comparing complete training budgets.

An episode succeeds when the point midway between the fingertips finishes within 5 cm of
the target point, 6 cm above the cube's centre, so the open fingers end just over the
cube's top face.

### Live play

```bash
simulo run samples/franka-reach-from-camera/play.py --frozen --policy <policy-id>:latest --episodes 20 --seed 42 --record false --viewstream
```

Open the live-view link printed by submission. Playback is paced to simulation
time; 20 episodes provide about one minute of live play after setup. Live play keeps the policy camera's
training render settings. The human view is separate from the 100 x 100 camera
input; changing the human view does not move that policy camera.

### Record each episode

```bash
simulo run samples/franka-reach-from-camera/play.py --frozen --policy <policy-id>:latest --episodes 3 --seed 42
simulo recordings <play-job-id> --all
```

Recording is enabled by default. Each episode produces a separate MCAP containing
environment 0, policy actions, numeric inputs and the policy camera video. Policy
camera metadata identifies the `camera` input and `policy_camera` camera. Frames
are sampled every policy step, including the terminal frame before reset.

### Export and replay

```bash
simulo export <policy-id>:latest --format onnx -o exported
cd <downloaded-bundle-directory>
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify.py
```

Use the bundle directory printed by `simulo export` for the `cd` command. The
version-2 bundle takes `camera` as uint8 NHWC and `joints` as float32. It needs
no critic inputs. `verify.py` replays the included test vectors through the
exported model and compares its actions with the saved expected actions. It
prints a local validation result and exits 0 when they agree. This checks model
conversion on that machine; it does not measure task success.

## What to expect

_Measured in one staging run with the candidate `simulo` 0.33.0 client. These
measured results are not a guarantee._

Training (job `job_modern-chamfer-d2pda5`) completed 600 PPO updates with
128 environments and seed 42 on Tier 1 (T4) in 28.02 minutes. This wall time
includes startup and artifact sealing and excludes queue wait. The catalog
rounds the measured 28.019953 minutes to 28 minutes.

Evaluating the latest checkpoint at update 600 (job `job_patient-cupola-t007bg`)
reported 88 of 100 episodes succeeded on numbered starts 0 through 99:
the fingertips finished within 5 cm of the target point above the cube.

## Inspecting results

```bash
simulo jobs
simulo logs <job-id> --follow
simulo result <job-id>
simulo policy list
simulo recordings <job-id> --all
```

Use the training job's policy ID with either `:best` or `:latest`.

## Troubleshooting

If the preview reports a scene, joint, or action problem, follow the public
[scene troubleshooting guide](https://docs.simulo.ai/guides/troubleshoot-a-scene-with-preview/)
before training.

Each environment renders its own camera image, which uses host memory. If a job runs out
of memory, lower `--num-envs`.

## Extending it

Change the cube's sampling region, the camera pose, or the resolution in
`FrankaReachFromCameraTask`. If you change the resolution, change the `camera` input's
shape to match. A policy trained with one camera setup will not work with another.

## Assets, licensing, attribution

The Franka Panda model is licensed under the Apache License 2.0.

The PPO settings started from an Apache License 2.0 vision PPO configuration for
a Panda cube task and were then tuned for this task.

The sample code is available under the [MIT License](https://opensource.org/license/mit).

Sample code: Copyright (c) 2026 Simulo LLC.
