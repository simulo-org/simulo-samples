# Cartpole from camera

## What this shows

Train a cart to balance its pole from a fixed camera and two joint velocities.
The actor receives a 100 x 100 RGB image, cart velocity, and pole angular
velocity. It never receives cart position or pole angle as numeric inputs. A
critic used only during training receives the positions to estimate returns.

## Prerequisites

- Python 3.11 or newer with Simulo 0.33.0 or newer installed from the public package index.
- A Simulo account and `simulo login`.
- Run commands from the repository root.

## Assets

The task uses the version-pinned catalog asset `simulo/robot/cartpole:v1`.

## Files and APIs

- `task.py` defines the scene, policy camera, actor and critic inputs, reward, and resets.
- `preview.py` checks the scene and action mapping with dummy actions.
- `train.py` trains with PPO and saves `latest` and `best` checkpoints.
- `eval.py` evaluates deterministic actions on numbered episode starts and defines success.
- `play.py` plays a saved policy and can record each episode separately.
- `.simuloignore` excludes local files from submitted packages.

The main APIs are `simulo.PolicyInputSchema`, `simulo.Camera`,
`simulo.LearningEnv`, and `simulo.RLTrainer`.

## Run it

### Preview

```bash
simulo run samples/cartpole-from-camera/preview.py --seed 42
```

Preview uses dummy actions and creates no learned policy. Structured camera-input
checks cover the scene, camera input, and action mapping before training.

### Train

```bash
simulo run samples/cartpole-from-camera/train.py --num-envs 256 --max-iterations 550 --seed 42
simulo policy list
```

Use `--seed 43` for an independent second run. The default training budget is
256 environments and 550 updates. The job has a 55-minute timeout. That timeout
is a limit, not a measured training duration.

### Resume

Resume training by supplying `--from <policy-id>:latest` to the training entry,
with the same environment count and seed and a larger total update budget.

`--max-iterations` is the total update count, including updates already in the
checkpoint. For a checkpoint at update 550, a total budget of 600 adds 50 updates.
Keep the environment count, task, camera, and PPO settings the same as the original
run. The continuation creates a new policy.

### Evaluate

```bash
simulo run samples/cartpole-from-camera/eval.py --policy <policy-id>:latest --episodes 100 --num-envs 100
simulo result <eval-job-id>
```

This runs deterministic actions on numbered starts 0 through 99. `:latest`
selects the final saved checkpoint, while `:best` selects the checkpoint with
the highest recorded training reward. Use `:latest` when comparing complete
training budgets.

### Live play

```bash
simulo run samples/cartpole-from-camera/play.py --policy <policy-id>:latest --episodes 20 --seed 42 --record false --viewstream --detach
```

Submission prints the play job ID and returns without waiting for logs.
Open the viewer while that job is running:

```bash
simulo view <play-job-id>
```

Playback requests pacing to simulation time and uses deterministic actions.
The human view is separate from the policy camera, so changing it does not
change the actor input.

### Record each episode

```bash
simulo run samples/cartpole-from-camera/play.py --policy <policy-id>:latest --episodes 3 --seed 42
simulo recordings <play-job-id> --all
```

Recording is enabled by default. Each episode produces a separate recording
for environment 0, including video from the policy camera.

### Export and verify

```bash
simulo export <policy-id>:latest --format onnx -o exported
cd <downloaded-bundle-directory>
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify.py
```

Use the bundle directory printed by `simulo export` for the `cd` command. The
version-2 bundle takes `image` as uint8 NHWC and `velocities` as float32 with
shape `(2)`. It contains no critic input. `verify.py` runs the included test
vectors through the exported model and exits successfully when its actions
match the saved expected actions. This verifies conversion in the fresh virtual
environment; it does not measure task success.

## What to expect

An episode succeeds when it survives the five-second time limit and finishes
with the pole within 10 degrees of upright. It need not remain within 10
degrees for the whole episode.

The default configuration uses 256 environments, 550 updates, 32 steps per rollout,
four PPO epochs, 16 minibatches, and a learning rate of 0.0001.

_Measured in one staging training run with the candidate `simulo` 0.33.0 client.
These results are not a guarantee._

Training (job `0c9403bf-6121-4cc8-a929-e404516e97a4`) completed 550 updates
on T4 in 36.8 minutes from job start to finish, excluding queue wait.
The catalog rounds this measured time to 37 minutes.
The full-run evaluation score is not published yet. Training duration does not
establish task success.

Training retains compact uint8 images in its rollout buffer. The actor converts
images to floats and centers them inside the policy network.

## Inspecting results

```bash
simulo jobs
simulo logs <job-id> --follow
simulo result <job-id>
simulo policy list
simulo recordings <job-id> --all
```

Use the training policy ID with either `:best` or `:latest`.

## Troubleshooting

If training runs out of memory, reduce `--num-envs` and measure the new
configuration before a full run. Preview the scene again after changing camera
pose or optics: the pole must remain visible within the declared track bounds.

## Extending it

Compare a saved policy on identical seeded starts with ordinary images and with
zeroed images. A large loss of success with zeroed images checks whether the
policy depends on its camera. Keep reward, success, and reset rules unchanged
for that comparison.

## Assets, licensing, attribution

The catalog robot is supplied by Simulo.

The sample code is available under the [MIT License](https://opensource.org/license/mit).

Sample code: Copyright (c) 2026 Simulo LLC.
