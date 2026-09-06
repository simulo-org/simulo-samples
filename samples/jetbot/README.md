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

- Python 3.11 or newer and the Simulo client: `python -m pip install "simulo>=0.23.1,<0.26"`.
- A Simulo account, signed in once with `simulo login`.
- A clone of this repository. The commands below run from its root.
- No GPU on your machine. The job asks for an L4-class GPU in the Simulo cloud and is billed to
  your account.

## Assets

- `simulo/robot/jetbot:v2`: a version-pinned catalog reference for the robot.

The robot model is not bundled with this sample. It is fetched from the catalog when it is not
already cached. Asset preparation and simulation startup occur before training output, so any job
may have a quiet period after it reports `running`.

## Files and APIs

- `app.py`: a quaternion helper, the reward kernel, the task, and the `train_jetbot` job.
- `.simuloignore`: files `simulo run` leaves out of the uploaded package.

Simulo names it uses, beyond those in the `cartpole` sample:

- `robot.state.pose` for the orientation the forward vector is derived from, and
  `robot.state.linear_velocity` for the speed term of the reward.
- `robot.set_joint_velocity_target(..., joint_ids=...)` on the two wheel joints, resolved with
  `robot.find_joints("left_wheel_joint")` and `robot.find_joints("right_wheel_joint")`.

## Run it

```bash
simulo login
simulo run samples/jetbot/app.py --num-envs 512 --max-iterations 700
```

For a quick check that the job launches, use `--num-envs 64 --max-iterations 2`. Add `--detach`
to submit without waiting for the log.

The job has one configured 8-hour execution budget shared by the initial attempt and its two
retries. Dependency and asset preparation happens before that execution deadline, so this is not
an absolute billing ceiling. Run `simulo cancel <job-id>` to stop a queued or running job.

## What to expect

An earlier single-GPU run of this code reported a best reward near 190 at iteration 50 and near
343 from iteration 150 onward. `best_reward` is a mean completed-episode return across 300 control
steps (5 seconds at 60 Hz), not a per-step reward. The alignment term can contribute at most 300
per episode; the velocity term supplies the remainder. Read those figures as the shape of the
curve rather than as numbers to hit; training is not bit-for-bit reproducible.

The result names the checkpoint the job saved into its volume, plus `iterations`, `best_reward`,
and checkpoint bookkeeping fields. A run takes about five minutes at the defaults once capacity is free; a first run
can take longer while the cloud prepares the runtime.

## Inspecting results

```bash
simulo jobs                           # status and job IDs
simulo logs <job-id> --follow         # iteration and checkpoint lines
simulo result <job-id>                # checkpoint, num_envs, iterations, best_reward
simulo models <job-id>                # best.pt and latest.pt
simulo models <job-id> best.pt        # download one, digest-verified
simulo cancel <job-id>                # stop a queued or running job
```

To see the robot drive, submit with `--viewstream` and open `simulo view` while it runs;
streaming can slow training, so use it to look.

`<job-id>` is printed by `simulo run` and listed by `simulo jobs`. Stopping a log-follow session
only detaches from the stream; it does not cancel the job.

## Troubleshooting

- The job is `running` but the log shows nothing for a while: the robot model may be fetched before
  the simulation starts. See [Assets](#assets).
- `simulo run` prints "This wrote a local package only": you are not signed in.
- The job stays `queued`: the cloud is waiting for GPU capacity.
- The reward stops improving after iteration 150 or so: that matches the cited run. Raise
  `--max-iterations` or `--num-envs` to push further.
- The job failed: `simulo logs` prints the platform's reason code and detail after its header.

## Extending it

- Reward shaping: `rew_scale_alignment` and `rew_scale_velocity` on `JetbotTask` weight
  pointing the right way against moving fast.
- Speed and episode length: `velocity_scale` and `episode_length_s`.
- Give it somewhere to go: [Install a PyPI dependency](../pip-install-shapely/) drives the same
  robot into a target zone and computes the reward with a third-party geometry library.
- Continue training: `--from <job-id>` with a higher `--max-iterations` total.

## Assets, licensing, attribution

The sample code is available under the [MIT License](../../LICENSE).

This sample installs no third-party packages.

### Attribution

Sample code: Copyright (c) 2026 Simulo LLC.
