# Bring your own F1TENTH-compatible car

## What this shows

Learn the complete custom-robot workflow by publishing a floating-base race car to your
organization's catalog, then previewing, training, evaluating, and recording a drifting policy.
The focused lesson is keeping a version-pinned custom asset connected to every stage of the
policy lifecycle.

## Prerequisites

- Python 3.11 or newer with the Simulo client installed.
- A Simulo account and `simulo login`.
- Permission to publish assets to your organization.
- Run every command below from the repository root.

## Assets

- `assets/robot/f1tenth/f1tenth.usd` is the source package included in this repository.
- `robot/f1tenth:v1` is the organization-scoped catalog reference used by the task after
  publication.

The car has a floating base, four driven wheels, and two steering joints. If you substitute a
different chassis, update the joint names and vehicle dimensions in `task.py`.

## Files and APIs

- `task.py` defines `F1TenthDriftTask`, the track, action mapping, reward, camera, and training
  helper.
- `preview.py` checks the published robot, joints, action mapping, resets, and training loop.
- `train.py` declares the `train` job.
- `eval.py` checks that the car stays on track, keeps moving, and travels counter-clockwise.
- `play.py` plays a saved checkpoint with an overhead camera and records the rollout.
- `.simuloignore` excludes local files from submitted packages.

The policy action is `[throttle, steer]`. The task maps it to wheel velocity and steering
position targets.

## Run it

Publish and inspect the asset before using any job:

```bash
simulo asset publish assets/robot/f1tenth \
  --kind robot --name f1tenth --entry f1tenth.usd --base floating
simulo asset inspect robot/f1tenth:v1
```

Publishing the same name again creates a new version. This sample remains pinned to `:v1` unless
you update `task.py`.

### Preview

```bash
simulo run samples/byo-f1tenth-drift/preview.py
simulo recordings <job-id>
```

### Train

```bash
simulo run samples/byo-f1tenth-drift/train.py --num-envs 256 --max-iterations 500
simulo policy list
```

### Evaluate

```bash
simulo run samples/byo-f1tenth-drift/eval.py --policy <policy-id>:best
```

### Play

```bash
simulo run samples/byo-f1tenth-drift/play.py --policy <policy-id>:best
simulo recordings <job-id>
```

#### Try this

- When adapting a chassis, run
  `simulo run samples/byo-f1tenth-drift/preview.py --checks training-loop,action-map` to check
  wheel and steering mappings before training.
- Compare checkpoints with
  `simulo run samples/byo-f1tenth-drift/eval.py --policy <policy-id>:best --compare <policy-id>:latest`
  to see whether the reward-best and latest policies produce the same task outcome.

## What to expect

One staging run on 2026-10-06 with `simulo 0.32.0` produced the preview, training, and
evaluation results below. It used the existing catalog asset `robot/f1tenth:v1` without
republishing it. Preview passed all checks.

Training ran for about 4 minutes. The returned best reward was about 92,330; both `best` and
`latest` were saved at iteration 500 in `policy_dandelion-plaid-3d5tcq`.

Evaluation reported 92 of 100 successes: 92 stayed on track for 5 seconds, 97 were moving at
least `1 m/s` at the end, and 100 moved counter-clockwise. The comparison found no clear
difference: `best` and `latest` each scored 92 of 100, with zero changed outcomes, because both
labels pointed at the same checkpoint.

An earlier staging run on 2026-10-05, also with `simulo 0.32.0`, trained its own policy and
measured the rest. It recorded the asset's content digest as
`sha256:aa5fc36e5f45c1e56615371b50e201d2622d3e8e7dfd7e1951181a416a95c998`, shown as
`content_digest` by `simulo asset inspect robot/f1tenth:v1 --json`. Its preview showed
action 0 driving all four wheels and action 1 driving both steering joints, and its random and
zero-action checks recorded 51 and 50 finite matching resets, respectively. That preview
recording verified as 9,544 MCAP messages and included Lichtblick and Foxglove layout files.
Its playback completed 1 episode and 300 steps with mean reward `88869.75 +/- 0.00`. Its verified
MCAP had 2,408 messages across 15 populated topics, including policy observations and actions,
rewards, robot commands, terminations, transforms, and 300 packets on each overhead-camera video
channel. Unlike the preview retrieval, the play retrieval returned no Lichtblick or Foxglove
layout sidecars.

## Inspecting results

```bash
simulo asset inspect robot/f1tenth:v1
simulo asset list
simulo jobs
simulo logs <job-id> --follow
simulo result <job-id>
simulo policy list
simulo recordings <job-id>
```

Use the asset inspection output to verify the version used by the job. Use the play recording to
inspect the car's path around the track.

## Troubleshooting

If the asset cannot be resolved, confirm that it was published in the organization you are
currently using and that `task.py` names the published version. If the car falls or its controls
are reversed, run preview and follow the public
[scene troubleshooting guide](https://docs.simulo.ai/guides/troubleshoot-a-scene-with-preview/)
before training.

## Extending it

Publish another compatible car under a new catalog name and update the reference, joint names,
and chassis dimensions in `task.py`. You can also change the track geometry or reward weights,
then compare `:best` and `:latest` with the same evaluation rule.

## Assets, licensing, attribution

The task, reward shape, track geometry, and race-car asset are ported from
[WheeledLab](https://github.com/UWRobotLearning/WheeledLab), an open-source robotics research
project from the University of Washington, under its BSD 3-Clause license:

> Copyright (c) 2025-2027, The Wheeled Lab Project Developers.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without modification, are permitted
> provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice, this list of
>    conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice, this list of
>    conditions and the following disclaimer in the documentation and/or other materials
>    provided with the distribution.
> 3. Neither the name of the copyright holder nor the names of its contributors may be used to
>    endorse or promote products derived from this software without specific prior written
>    permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
> IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
> FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
> CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
> DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
> IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
> OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"F1TENTH-compatible" describes physical compatibility with the open F1TENTH platform. It does
not assert rights to the F1TENTH trademark.

This sample's port is also available under the repository's [MIT License](../../LICENSE).

Sample code: Copyright (c) 2026 Simulo LLC.
