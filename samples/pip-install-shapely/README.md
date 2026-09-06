# Install a PyPI dependency

## What this shows

How to bring a third-party PyPI library into a Simulo job. The base runtime includes `torch`,
`numpy`, and the scientific Python stack, but not every third-party package a task may need.
`Runtime.pip_install("shapely")` asks Simulo to install Shapely from public PyPI before the job
runs, so you do not need to build or bring a separate image. The training reward genuinely
depends on the library: a JetBot drives toward a target zone defined as a `shapely` polygon, and
the reward is `Polygon.distance` and `Polygon.contains`, computed on the CPU every step.

The sample also shows `Runtime.env()`. The zone's centre is a real task parameter,
`DEMO_ZONE_CENTER_X`, set on the runtime and read back from `os.environ` inside the job, which
logs the value it read.

You will learn how to declare a dependency and an environment variable on a `Runtime`, why the
runtime must be a dedicated `Runtime.from_registry(...)` instance, how a third-party import is
deferred alongside `torch`, and why a reward that calls into a CPU library cannot be a
`torch_jit` kernel.

## Prerequisites

- Python 3.11 or newer and the Simulo client: `python -m pip install "simulo>=0.23.1,<0.26"`.
  You do not need `shapely` on your machine; `simulo run` never imports it.
- A Simulo account, signed in once with `simulo login`.
- A clone of this repository. The commands below run from its root.
- No GPU on your machine. The job asks for an L4-class GPU in the Simulo cloud and is billed to
  your account.

## Assets

- `simulo/robot/jetbot:v2`: a version-pinned catalog reference for the robot. The model is fetched
  when it is not already cached, so any run may have a quiet asset-preparation phase.
- `simulo/gpu-rl:2026.06`: the Simulo runtime the app names explicitly, so that it can layer
  `pip_install` and `env` onto it. It is the same runtime every other sample gets by default.

## Files and APIs

- `app.py`: the runtime declaration, a quaternion helper, the shapely reward, the task, and the
  `train` job.
- `.simuloignore`: files `simulo run` leaves out of the uploaded package.

Simulo names it uses, beyond those in the `jetbot` sample:

- An exact runtime and app declaration:

  ```python
  runtime = (
      simulo.Runtime.from_registry("simulo/gpu-rl:2026.06")
      .pip_install("shapely")
      .env({"DEMO_ZONE_CENTER_X": "2.5"})
  )
  app = simulo.App("pip-install-shapely", mounts={"/out": vol}, runtime=runtime)
  ```
- `with app.runtime.imports():` around both `import torch` and
  `from shapely.geometry import Point, Polygon`.
- `env.scene.env_origins` in `on_start`, to express positions relative to each environment's own
  spawn point.
- `robot.set_root_pose(...)` and `robot.internals.default_root_state` in the spawn-jitter reset.

## Run it

```bash
simulo login
simulo run samples/pip-install-shapely/app.py --num-envs 64 --max-iterations 300
```

For a quick check that the job launches, use `--num-envs 8 --max-iterations 2`. Add `--detach` to
submit without waiting for the log.

The job has one configured 8-hour execution budget shared by the initial attempt and its two
retries. Dependency and asset preparation happens before that execution deadline, so this is not
an absolute billing ceiling. Run `simulo cancel <job-id>` to stop a queued or running job.

## What to expect

Before training starts, Simulo prepares a runtime with `shapely` installed, which adds some time
before the first log line. The first line the job itself prints is the environment variable it
read:

```
[demo] target-zone center X from Runtime.env(): 2.5
```

That line is the proof that `env()` reached the job, not just that the submit recorded it. Then the
usual iteration counter, checkpoint, and best-reward lines follow. The per-step reward is minus
the distance to the polygon, which is zero for a point inside it. A point strictly inside also
receives the 2.0 reach bonus. Each episode has 600 control steps, so the theoretical episode-return
ceiling is 1200. The logged `best_reward` is a mean completed-episode return.

With the default centre X of 2.5, the one-metre square spans X=2.0 to 3.0 and Y=-0.5 to 0.5 in
each environment's local frame. Reset jitter is at most 0.3 m per axis. Both come straight from
the source, so you can read them off `_target_zone_vertices` and the reset code.

The result names the checkpoint the job saved into its volume, plus `iterations`, `best_reward`,
and checkpoint bookkeeping fields. The full run takes about three minutes at the defaults once capacity is free; a first
run can take longer while the cloud prepares the runtime, including installing shapely.

## Inspecting results

```bash
simulo jobs                  # status
simulo logs --follow         # the [demo] line, then training progress
simulo result                # checkpoint, num_envs, iterations, best_reward
simulo models                # best.pt and latest.pt uploaded by ResumableCheckpoint
simulo cancel <job-id>        # stop a queued or running job
```

`simulo logs --from-start` shows the `[demo]` line if it has scrolled out of the tail. The
`[demo]` prefix is the exact string the code emits. Stopping a log-follow session only detaches
from the stream; it does not cancel the job.

## Troubleshooting

- The job is `running` but the log is empty: the runtime and asset may still be preparing before
  the simulation starts.
- `simulo run` refuses the submit naming a package conflict: `pip_install` cannot override a
  package the base runtime already ships (`torch`, `numpy`, and the like). Only add packages the
  runtime does not have.
- `simulo run` refuses an unknown runtime name: `Runtime.from_registry` must name a runtime
  Simulo offers. This sample pins `simulo/gpu-rl:2026.06`.
- `pip_install()` seems to affect other apps: it mutates the `Runtime` it is called on. Always
  build a dedicated `Runtime.from_registry(...)` instance rather than calling it on the shared
  default `app.runtime`.
- The reward never goes positive: at these defaults that is plausible. The distance term
  saturates at zero and only `reach_bonus` lifts it above; give the run more iterations, or more
  environments, before concluding anything.

## Extending it

- Move the zone: change the `.env({"DEMO_ZONE_CENTER_X": "2.5"})` value. Check that the new zone
  is still reachable from the spawn point before training against it.
- Add a package: chain another `.pip_install("name==version")` call. Each call is an ordered
  layer installed on top of the base runtime, public PyPI only. This sample intentionally requests
  unpinned `shapely`, so the resolved release can change. For a reproducible project, use an exact
  requirement such as `shapely==<tested-version>`.
- Change the shape: `_target_zone_vertices` returns any polygon `shapely` can build.
- Use the library for more: `shapely` can also define obstacles to penalise, or a corridor to
  reward staying inside.

## Assets, licensing, attribution

The sample code is available under the [MIT License](../../LICENSE).

`shapely` is fetched from [PyPI](https://pypi.org/project/shapely/) and is not included in this
repository. The project publishes its
[BSD 3-Clause license](https://github.com/shapely/shapely/blob/main/LICENSE.txt). Shapely's binary
distributions may bundle GEOS, whose project publishes the
[LGPL-2.1 license](https://github.com/libgeos/geos/blob/main/COPYING). The exact Shapely version is
not pinned by this sample.

### Attribution

Sample code: Copyright (c) 2026 Simulo LLC. Shapely: Copyright (c) 2007, Sean C. Gillies; 2019,
Casper van der Wel; 2007-2022, Shapely Contributors; BSD 3-Clause License. GEOS: LGPL-2.1; it may
be bundled in a Shapely binary distribution.
