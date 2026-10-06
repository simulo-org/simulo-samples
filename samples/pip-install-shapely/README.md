# Install a PyPI dependency

## What this shows

Learn how to install a pinned public PyPI package into a job's runtime and use it in a real
reward. This sample installs Shapely and trains a JetBot to drive into a square target zone. Every
step, Shapely computes each robot's distance to the zone and whether the zone covers it, for all
robots at once. A second runtime setting, `DEMO_ZONE_CENTER_X`, places the zone, and preview,
training, evaluation, and playback all read it.

## Prerequisites

- Python 3.11 or newer with the Simulo client installed.
- A Simulo account and `simulo login`.
- Run every command below from the repository root.

## Assets

- `simulo/robot/jetbot:v1` is the version-pinned catalog robot. The job fetches it from the
  catalog when it starts; you do not need to publish it.
- `simulo/gpu-rl:2026.06` is the base runtime.
- `shapely==2.1.2` is installed from public PyPI into that runtime.

## Files and APIs

- `task.py` creates the dedicated runtime and defines `ShapelyZoneTask`, its observations,
  single drive action, Shapely reward, and end condition.
- `preview.py` checks the task with the same zone position the other jobs use.
- `train.py` declares the `train` PPO job and logs the zone position it read.
- `eval.py` checks that each robot finishes inside the 1 m target zone.
- `play.py` plays a saved checkpoint with an overhead camera and records the rollout.
- `.simuloignore` excludes local files from submitted packages.

The focused APIs are `Runtime.from_registry(...)`, `Runtime.pip_install(...)`, and
`Runtime.env(...)`. `task.py` builds its own runtime with all three instead of changing the
default runtime that other apps share.

## Run it

### Preview

```bash
simulo run samples/pip-install-shapely/preview.py
simulo recordings <job-id>
```

### Train

```bash
simulo run samples/pip-install-shapely/train.py
simulo policy list
```

The default trains 32 environments for 700 iterations. Use the policy ID printed by training in
the remaining commands.

### Evaluate

```bash
simulo run samples/pip-install-shapely/eval.py --policy <policy-id>:best
```

An episode succeeds when the robot's final position is inside the 1 m target zone. The rule reads
the robot's pose and the task's zone, not the policy's observation.

### Play

```bash
simulo run samples/pip-install-shapely/play.py --policy <policy-id>:best --episodes 3
simulo recordings <job-id>
```

#### Try this

- Move the zone half a metre closer. In `task.py`, change `.env({"DEMO_ZONE_CENTER_X": "2.5"})`
  to `"2.0"`, rerun preview, then train a new policy, because the existing one learned the old
  distance. The training log prints the new value.
- Compare the saved checkpoints with
  `simulo run samples/pip-install-shapely/eval.py --policy <policy-id>:best --compare <policy-id>:latest`.

## What to expect

One staging run on 2026-10-06 with `simulo 0.32.0` produced the training and evaluation results
below. Preview passed 13 of 13 checks (job `job_parallel-drive-tyt9yk`), and play completed
3 episodes and 837 steps (job `job_excited-cupola-jt4r5c`), both measured 2026-10-06.

Every job prints `layer numpy==2.4.6 is shadowed by the platform runtime numpy==1.26.0`. The
platform's numpy wins and Shapely runs normally, so the line is harmless. Preview also logs that
`TargetZone` is visual-only, which means the zone is a marker and has no physics.

The default training command ran for about 4 minutes (259 seconds) and saved its best checkpoint
at iteration 350 in `policy_maize-decision-x6je0y`. The best reward was about -166. The reward is
the negative distance to the zone on every step until the robot arrives, so even a policy that
always reaches the zone finishes with a negative total; the reward of 100 for arriving is what
separates an arrival from a near miss.

Evaluation of the best checkpoint (job `job_denim-ray-kc4106`, on the policy trained by job
`job_maize-decision-x6je0y`) reported `100 of 100 episodes succeeded`, and every episode
ended by reaching the target zone.

Two more trainings in the same run changed only the seed in `train.py`. Seed 7 saved its best
checkpoint at iteration 250 and seed 123 at iteration 700, and both best checkpoints also scored
100 of 100. These are measured runs, not a guarantee that every seed learns as well.

Preview runs its checks on 16 environments and records a video. Play records three episodes from
a camera above the robot's path to the zone, so you can watch where each episode ends.

## Inspecting results

```bash
simulo jobs
simulo logs <job-id> --follow
simulo result <job-id>
simulo policy list
simulo recordings <job-id>
```

The training log prints `[demo] target-zone center X from Runtime.env(): 2.5`, which shows that the
runtime setting reached the running job. The evaluation report shows how each episode ended and
whether the robot finished inside the zone.

## Troubleshooting

If package preparation fails, confirm that the requirement in `Runtime.pip_install(...)` names a
public PyPI release and does not replace a package the base runtime already supplies. A quiet job
may still be preparing its runtime and catalog asset before the simulation starts. Use the public
[scene troubleshooting guide](https://docs.simulo.ai/guides/troubleshoot-a-scene-with-preview/)
when preview fails.

## Extending it

Try another pinned dependency in the dedicated runtime, or change the zone's shape in
`_target_zone_vertices`. The end condition in `get_dones`, the success rule in `eval.py`, and the
cuboid drawn in `build` also assume a 1 m square, so change them together, then preview before
training. Keep every job file reading the same `DEMO_ZONE_CENTER_X` value.

## Assets, licensing, attribution

The catalog robot is supplied by Simulo. The sample code is available under the
[MIT License](../../LICENSE).

[Shapely](https://pypi.org/project/shapely/) is installed from PyPI and is not included in this
repository. Shapely is available under the
[BSD 3-Clause License](https://github.com/shapely/shapely/blob/main/LICENSE.txt). Its binary
distributions may bundle GEOS, which is available under
[LGPL-2.1](https://github.com/libgeos/geos/blob/main/COPYING).

Sample code: Copyright (c) 2026 Simulo LLC. Shapely: Copyright (c) 2007, Sean C. Gillies; 2019,
Casper van der Wel; 2007-2022, Shapely Contributors.
