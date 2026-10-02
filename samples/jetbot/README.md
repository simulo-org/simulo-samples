# JetBot

## What this shows

A small two-wheeled robot learns to turn toward a randomly chosen heading and drive along it as
fast as it can, by setting the angular velocity of its left and right wheels. The two action
values map one to one onto the two wheels, so the control problem and the actuator layout match.

You will learn how to express a heading as a unit vector so the observation stays compact and
has no angle wrap-around, how to resample the command on every reset so one policy learns all
directions, and how a task with no failure condition is written (every episode simply runs out of
time).

## Prerequisites

- Python 3.11 or newer and the Simulo client: `python -m pip install --upgrade simulo`.
- A Simulo account, signed in once with `simulo login`.
- A clone of this repository. The commands below run from its root.
- No GPU on your machine. The job asks for a Tier 1 GPU (T4) in the Simulo cloud and is
  billed to your account. Hardware describes the job's own request, not queue priority:
  it waits on the same shared GPU fleet as every other job.

## Assets

- `simulo/robot/jetbot:v2`: a version-pinned catalog reference for the robot.

The robot model is not bundled with this sample. It is fetched from the catalog when it is not
already cached. Asset preparation and simulation startup occur before training output, so any job
may have a quiet period after it reports `running`.

## Files and APIs

- `task.py`: the application, quaternion helper, reward kernel, and JetBot task.
- `train.py`: the `train_jetbot` job, declared with `@app.job(type="train", ...)`, which
  saves the policy's `best` and `latest` checkpoints automatically.
- `eval.py`: the evaluation job and its sustained-movement success rule.
- `.simuloignore`: files `simulo run` leaves out of the uploaded package.

Simulo names it uses, beyond those in the `cartpole` sample:

- `robot.state.pose` for the orientation the forward vector is derived from, and
  `robot.state.linear_velocity` for the speed term of the reward.
- `robot.set_joint_velocity_target(..., joint_ids=...)` on the two wheel joints, resolved with
  `robot.find_joints("left_wheel_joint")` and `robot.find_joints("right_wheel_joint")`.

## Run it

```bash
simulo login
simulo run samples/jetbot/train.py
```

For a quick check that the job launches, use `--max-iterations 2`. Add `--detach` to submit
without waiting for the log.

What bounds `--num-envs` is memory, and the bound is per app rather than per tier: the other
samples here ship much larger defaults. 16 is the figure measured for this app. Asking for more
memory than the job is given ends the run with an out-of-memory failure rather than running it
slowly. `simulo systems` reports a measured ceiling for Tier 1 and "not yet measured" for the
other tiers; that figure comes from one workload and is not a platform-wide limit.

The job has one configured 8-hour execution budget shared by the initial attempt and its two
retries. Dependency and asset preparation happens before that execution deadline, so this is not
an absolute billing ceiling. Run `simulo cancel <job-id>` to stop a queued or running job.

### Evaluate

From this sample directory, use `simulo policy list` to find the policy id, then run:

```bash
simulo run eval.py --policy policy_<id>:best
```

The rule checks that the robot lasts to the time limit and is moving at least 0.1 m/s at the end.

<!-- REPORT EXCERPT: filled after the GPU run -->
```text
```

## What to expect

An earlier single-GPU run of this code with `--num-envs 512` reported a best reward near 190 at
iteration 50 and near 343 from iteration 150 onward. That run used far more parallel environments
than the current default, so read its numbers as the shape of the curve rather than as figures
this sample reproduces: fewer environments means less experience per iteration, so the same reward
arrives later, if at all, within 700 iterations. `best_reward` is a mean completed-episode return
across 300 control steps (5 seconds at 60 Hz), not a per-step reward. The alignment term can
contribute at most 300 per episode; the velocity term supplies the remainder. Training is not
bit-for-bit reproducible.

The result holds `num_envs` and training statistics such as `iterations`, `best_reward`, and
checkpoint bookkeeping fields. Expect several minutes of startup before any training output
appears, whatever `--max-iterations` you pass; total time past that grows with the iteration
count. A first run can take longer still while the cloud prepares the runtime.

## Inspecting results

```bash
simulo jobs                           # status and job IDs
simulo logs <job-id> --follow         # iteration and checkpoint lines
simulo result <job-id>                # num_envs, iterations, best_reward
simulo policy list                    # your policies and their checkpoints
simulo policy get <policy-id>:best    # download the best checkpoint, digest-verified
simulo export <policy-id>:best        # the best checkpoint as a portable ONNX bundle
simulo cancel <job-id>                # stop a queued or running job
```

To see the robot drive, submit with `--viewstream` and open `simulo view` while it runs;
streaming can slow training, so use it to look.

`<job-id>` and `<policy-id>` are printed by `simulo run` and listed by `simulo jobs`. Every
training job makes one policy, whose ID is the job's ID with a `policy_` prefix in place of
`job_`; the training job saves its `best` and `latest` checkpoints automatically. Stopping a
log-follow session only detaches from the stream; it does not cancel the job.

## Troubleshooting

- The job is `running` but the log shows nothing for a while: the robot model may be fetched before
  the simulation starts. See [Assets](#assets).
- `simulo run` prints "This wrote a local package only": you are not signed in.
- The job stays `queued`: the cloud is waiting for GPU capacity.
- The reward stops improving after iteration 150 or so: that matches the cited run, which used
  many more environments than the default. Raise `--max-iterations` to push further. Raising
  `--num-envs` gives more experience per iteration, but memory bounds it: 32 has been measured
  to complete for this app and 64 has not.
- The job failed: `simulo logs` prints the platform's reason code and detail after its header.

## Extending it

- Reward shaping: `rew_scale_alignment` and `rew_scale_velocity` on `JetbotTask` weight
  pointing the right way against moving fast.
- Speed and episode length: `velocity_scale` and `episode_length_s`.
- Give it somewhere to go: [Install a PyPI dependency](../pip-install-shapely/) drives the same
  robot into a target zone and computes the reward with a third-party geometry library.
- Continue training: `--from <policy-id>:best` with a higher `--max-iterations` total.

## Assets, licensing, attribution

The sample code is available under the [MIT License](../../LICENSE).

This sample installs no third-party packages.

### Attribution

Sample code: Copyright (c) 2026 Simulo LLC.
