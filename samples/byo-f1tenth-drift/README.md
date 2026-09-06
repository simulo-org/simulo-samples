# Bring your own F1TENTH-compatible car

## What this shows

The other bring-your-own sample, [byo-urdf-arm](../byo-urdf-arm/), publishes a robot
from a URDF and trains a fixed-base arm to reach. This one publishes a robot from a USD
package and trains a mobile, 4WD, 2-wheel-steer race car to drift around a
stadium-shaped track — the same catalog-publishing workflow, on a genuinely dynamic
driving task with a real failure mode: the car can leave the track.

You send [`assets/robot/f1tenth/`](../../assets/robot/f1tenth/) to your organization's
catalog with one command, and from then on the car behaves like any other catalog
robot: version-pinned, resolved when you submit, and mounted read-only while the job
runs. You will learn 4WD steering (turn-radius geometry driving four independently
scaled wheel targets from one steering angle), a multi-term weighted-sum reward with a
real drift band, and a two-job train-then-inspect lifecycle: `train` saves a trainer
checkpoint and exports a standalone TorchScript policy, tracking the best-so-far
checkpoint rather than trusting the final one, and `rollout` plays that policy back with
`simulo.RLPlayer` and records it to MCAP with an overhead camera video.

The task, reward shape, and race-car asset are ported from **WheeledLab**
(`UWRobotLearning/WheeledLab`), an external open-source robotics research project — see
[Assets, licensing, attribution](#assets-licensing-attribution) below. This is the only
sample in this repository whose task code and asset are derived from a third party
rather than written for it.

## Prerequisites

- Python 3.11 or newer and the Simulo client: `python -m pip install "simulo>=0.23.1,<0.25"`.
- A Simulo account, signed in once with `simulo login`.
- A clone of this repository. The commands below run from its root.
- The car published to your organization's catalog. [Run it](#run-it) does that first;
  the training job cannot start before it.
- No GPU on your machine. Both jobs ask for an L4-class GPU in the Simulo cloud and are
  billed to your account.
- To open the recording: Foxglove or Lichtblick, both free desktop applications, or the
  `mcap` Python package, which the Simulo client already depends on.

## Assets

- `assets/robot/f1tenth/f1tenth.usd`: an F1TENTH-compatible race car, already cleaned
  for publishing. This file is the input to `simulo asset publish`. It sits outside this
  sample directory, so `simulo run` never uploads it; the car reaches the job from the
  catalog instead. The [assets tree](../../assets/) explains how its directories are
  named.

  The asset started as upstream WheeledLab's own race-car package, which is not
  publishable as-is: it embeds a sensor/ROS bridge subtree (a lidar publisher, a drive
  bridge, odometry) that a publish rejects, and its collision meshes duplicate the
  visual meshes at full resolution where a simpler collider already exists. Neither is
  needed — the task itself uses a sensor-free observation space and disables every lidar
  at startup — so the shipped file has that subtree removed and its collision geometry
  simplified. Nothing about how the car drives changes.
- `robot/f1tenth:v1`: the car once you have published it, pinned to version 1. The
  reference carries no publisher segment, so it resolves against whichever organization
  you are signed in as.

The car has six joints, each resolved on its exact name in `app.py`: steering
`rotator_left` / `rotator_right`, and wheels `wheel_front_left` / `wheel_front_right` /
`wheel_back_left` / `wheel_back_right`. A car of your own with different joint names
needs those names changed to match, in `on_start`, where they're resolved.
`base_length`, `base_width`, `wheel_radius`, and `spawn_height` are this car's specific
dimensions and ride height; a different chassis needs all four adjusted too, or it
spawns clipping into or floating above the ground.

## Files and APIs

- `app.py`: the 4WD action mapping, the multi-term reward kernel, the task, and the
  `train` and `rollout` jobs.
- `.simuloignore`: files `simulo run` leaves out of the uploaded package.
- `../../assets/robot/f1tenth/f1tenth.usd`: the race-car asset.

Simulo names it uses, beyond those in [byo-urdf-arm](../byo-urdf-arm/) and
[Cartpole Eval](../cartpole-eval/):

- `simulo.Asset.from_registry("robot/f1tenth:v1")`: the same organization-scoped
  reference pattern as `byo-urdf-arm`, on a floating-base (mobile) robot instead of a
  fixed one.
- `robot.set_joint_position_target(...)` for the two steering joints and
  `robot.set_joint_velocity_target(...)` for the four wheels, both restricted to their
  own `joint_ids`.
- `robot.set_root_pose(...)` and `robot.set_root_velocity(...)`, used at reset to
  teleport the car to a random point on the track and again mid-episode to apply a small
  periodic push.
- `simulo.Camera`, `simulo.SensorOffset.look_at(...)`, and `simulo.CameraSpawnConfig`
  for a standalone, world-frame, top-down camera added directly to the scene (not
  attached to the robot, unlike `cartpole-eval`'s side-mounted one) — a link name on a
  car you bring yourself is never known in advance.
- `simulo.RLPlayer(env=..., checkpoint=...)` and `player.play(num_steps=..., record=...)`,
  the same inference-and-recording pattern `cartpole-eval`'s `rollout` job uses.

The reward is a single weighted sum of seven terms: side-slip (rewarded only inside a
real-drift band, not while crawling or spinning out), speed-tracking against a 3 m/s
cruise target, track progress, corner "turn energy", a cross-track penalty against the
racing line, a counter-steer bonus, and an out-of-bounds termination penalty. The
counter-steer bonus's weight is `0.0`: upstream ramps it in only after basic driving is
learned, on a clock a Simulo `Task` could keep for itself, but a schedule compressed to
fit this trainer's budget was measured to hold the lap less reliably and to diverge more
often, so this port fixes every weight at upstream's starting value instead. Two similar
simplifications apply to domain randomization
(only spawn pose and a periodic push are randomized per episode; `simulo.Robot(...)`
does accept `mass_scale` and `actuator_gains`, but only as constructor arguments applied
once, uniformly, to every parallel environment — there is no way to re-roll them per
episode the way a spawn pose is re-rolled, and tire friction has no setter at all) and
to the actor/critic network (a fixed `[256, 128, 64]` shape, in place of upstream's
requested `[64, 64]`). None of these change what the car is trained to do; they are
stated in full
in `app.py`'s module docstring.

## Run it

Publish the car first, from the repository root:

```bash
simulo login
simulo asset publish assets/robot/f1tenth \
  --kind robot --name f1tenth --entry f1tenth.usd --base floating
```

`--base floating` because this is a mobile vehicle, not a bolted-down manipulator.
Publishing uploads the package and validates it in the cloud (structural checks,
USD-schema checks, and a physics settle probe) before publishing version 1. Add
`--report report.json` to keep the full report, or `--json` for a machine-readable
summary.

Check what the catalog recorded:

```bash
simulo asset inspect robot/f1tenth:v1
```

Then train, and once `train` completes, play the result back:

```bash
simulo run samples/byo-f1tenth-drift/app.py --job train --num-envs 256 --max-iterations 500
simulo run samples/byo-f1tenth-drift/app.py --job rollout --num-steps 300
```

`--job` is required because the file declares two jobs; `simulo run
samples/byo-f1tenth-drift/app.py -h` lists them, and `--job train -h` lists one job's
own flags. `--num-envs` and `--max-iterations` are `train`'s; `--num-steps` is
`rollout`'s.

`train`'s execution budget is eight hours, shared across its first attempt and up to two
retries; `rollout`'s is one hour. Pressing Ctrl-C while following logs only detaches
your terminal. Stop a queued or running job explicitly with its job id:

```bash
simulo cancel <job-id>
```

## What to expect

Publishing is a one-time step; every run after that resolves `robot/f1tenth:v1` from the
catalog and uploads nothing.

`train`'s reward starts strongly negative (around -4,900 at the first update) and climbs
quickly to about 50,000 by iteration 50 and 65,000-78,000 between iterations 100 and
300 — real, non-degenerate learning. **Training on this task still diverges to `NaN`
late in a run in some attempts** (an intermittent minority, not most). That is exactly
why `train` loads the trainer's tracked best-so-far checkpoint before saving and
exporting, rather than trusting the final state, and why it independently reloads the
exported policy and verifies every value is finite before returning success — a run that
does diverge still hands back a clean, usable checkpoint from whatever iteration was
best, though one that diverges early exports a correspondingly weaker policy (an
iteration-50 checkpoint from one such run held the track in 40 of 64 evaluation
environments and lapped in 24 of them, versus the numbers below for a run that trained
further). A `train` run at the defaults (256 envs, 500 iterations) takes about 135
seconds; `rollout` (300 steps) takes about 25 seconds. Both figures are from a local RTX
3090; wall-clock in the cloud will differ with GPU class and load.

`train`'s result names both saved files and whether the best checkpoint was used:

```json
{
  "checkpoint": "<checkpoints-volume-path>/f1tenth_drift_final.pt",
  "policy": "<checkpoints-volume-path>/f1tenth_drift_policy.pt",
  "used_best_checkpoint": true,
  "num_envs": 256,
  "robot_asset": "robot/f1tenth:v1",
  "iterations": 500,
  "best_reward": 71313.2
}
```

`used_best_checkpoint` is `false` only when no episode ever finished during training —
in that case both files are the trainer's final state, and a rerun with a different seed
is worth trying. If `best_reward` comes back well under 60,000, the run likely diverged
early; a rerun is worth it there too.

Be clear-eyed about what the resulting policy does. Played back deterministically from
64 random spawn points for one full 5-second episode each, it held the track in 59 of
those 64, and 57 of those 59 completed at least one full lap (mean 1.55 laps, at close
to the 3 m/s cruise target); the 5 that left the corridor did so within 9 steps of a
spawn that pointed them straight at the wall. Its mean slip angle is below the 0.25 rad
band the side-slip reward pays for, so at these weights it drives as a fast racing-line
follower more than a dramatic drifter — `rollout`'s recording is the way to see exactly
what your own run produced, and reshaping the reward toward more slip is one of the
things to try under "Extending it" below.

`rollout` plays one full 5 s episode and returns playback statistics, including
`messages_written` and `recording_complete`, alongside the recording path:

```json
{
  "policy": "<checkpoints-volume-path>/f1tenth_drift_policy.pt",
  "mcap": "<reports-volume-path>/rollout.mcap",
  "messages_written": 2317,
  "recording_complete": true
}
```

## Inspecting results

```bash
simulo asset inspect robot/f1tenth:v1   # what the catalog recorded for the car
simulo asset list                       # every asset in your organization's catalog
simulo jobs                             # both jobs, most recent first
simulo logs <job-id> --follow           # iteration and checkpoint lines
simulo result <job-id>                  # the returned dictionary
simulo models <job-id>                  # best.pt and latest.pt from the training job
simulo recordings                       # download rollout.mcap from the rollout job
simulo cancel <job-id>                  # stop a queued or running job
```

Open `rollout.mcap` in Foxglove or Lichtblick. Add an Image panel on
`/sensors/camera/overhead_cam/video_foxglove` to watch the car from above, and a Plot
panel on `/reward.total` to see the reward per step.

To watch training itself, submit with `--viewstream` and open `simulo view` while it
runs; streaming slows training, so use it to look, not for a timed run.

`<job-id>` is printed by `simulo run` and listed by `simulo jobs`. Stopping a log-follow
session only detaches from the stream; it does not cancel the job.

## Troubleshooting

- The job fails while building the scene, naming an asset it cannot resolve: the car is
  not in the catalog of the organization you are signed in as. Publish it, then submit
  again.
- `rollout` fails with "Checkpoint not found": run `--job train` first. `rollout` reads
  the policy `train` writes, and the checkpoint volume starts empty.
- `simulo asset publish` cannot tell which file to start from: pass
  `--entry f1tenth.usd`.
- The car falls through the ground or drives strangely: it was published without
  `--base floating`. Publish again with it; that creates a new version, so update the
  reference in `app.py` and re-publish before training against it.
- `simulo run` prints "This wrote a local package only": you are not signed in.
- The job stays `queued`: the cloud is waiting for GPU capacity.
- `train` raises "produced a non-finite action ... refusing to hand back a broken
  checkpoint": this fires only when no episode ever finished during the whole run, so
  even the best-so-far checkpoint has nothing usable in it. Rerun with more iterations
  or a different seed.
- The task cannot find its joints after you swap in a car of your own: the six names
  `on_start` and `_compute_car_targets` look up are the ones this asset declares. Change
  them to your car's names.

## Extending it

- Publish a car of your own: point `simulo asset publish` at your own USD package,
  choose the catalog name with `--name`, and change the reference at the top of
  `app.py` to match.
- Train for longer: raising `--max-iterations` gives the policy more chances at a better
  peak reward, and the best-checkpoint export means a late divergence still doesn't cost
  you the run — though it does not guarantee the divergence won't happen at all.
- Reshape the reward: the `rew_scale_*` class attributes on `F1TenthDriftTask` weight
  each term against the others. `rew_scale_tlgr` in particular starts at `0.0`; raising
  it turns on the counter-steer bonus upstream normally phases in later in training.
  Raising `rew_scale_side_slip`, or lowering `slip_engage_threshold`, pushes the trained
  policy from the racing-line follower "What to expect" describes toward more visible
  drifting.
- Change the track: `straight`, `line_radius`, `corner_in_radius`, and
  `corner_out_radius` set the stadium's shape and how much drivable corridor the car has
  before an episode terminates.
- Change the shot: `simulo.SensorOffset.look_at(pos=..., target=...)` in `build` places
  the overhead camera.

## Assets, licensing, attribution

The task, reward shape, and track geometry in `app.py`, and the race-car asset in
`assets/robot/f1tenth/`, are ported from **WheeledLab** (`UWRobotLearning/WheeledLab`),
an open-source robotics research project from the University of Washington, used here
under its BSD-3-Clause license:

> Copyright (c) 2025-2027, The Wheeled Lab Project Developers.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without modification,
> are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
>
> 3. Neither the name of the copyright holder nor the names of its contributors
>    may be used to endorse or promote products derived from this software without
>    specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
> ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
> WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
> ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
> (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
> LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
> ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
> SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This robot is described as **F1TENTH-compatible** — a statement of physical
compatibility with the open F1TENTH platform, not a claim to the F1TENTH trademark. A
BSD-3-Clause license grants copying, modification, and redistribution rights; it does
not grant trademark rights, and none are asserted here.

This sample's own code (the port itself, distinct from the license text above) is
additionally available under the [MIT License](../../LICENSE), the same as every other
sample here.

This sample installs no third-party packages.

### Attribution

Sample code (the port itself): Copyright (c) 2026 Simulo LLC.
